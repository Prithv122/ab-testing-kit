"""CUPED: spend pre-experiment data to buy precision, not sample size.

Every other module in this toolkit changes the *decision rule* applied to an experiment.
This one changes the *metric*, and it is the only lever here that makes an experiment
genuinely cheaper rather than merely differently risky.

The idea is that a unit's behaviour before the experiment predicts its behaviour during it,
and that predictable part is noise as far as the treatment effect is concerned. If ``X`` is a
pre-experiment covariate -- last month's spend, sessions in the prior fortnight, whatever the
platform already has -- then

    Y_adjusted = Y - theta (X - Xbar)

has the same expectation as ``Y`` for any constant ``theta``, because ``E[X - Xbar] = 0`` and
``X`` was fixed before randomisation. So the estimated treatment effect stays unbiased. But
its *variance* is minimised at ``theta = Cov(Y, X) / Var(X)``, where it falls to

    Var(Y) (1 - rho^2)

with ``rho`` the correlation between covariate and outcome. That is the whole method: a
correlation of 0.7 removes about half the variance, which is worth the same as doubling the
sample size, and costs nothing but a join against data the platform already has.

Two things about that formula deserve to be checked rather than believed, and this module
checks both:

* **Is the estimator still unbiased?** ``theta`` is estimated from the same data it adjusts,
  which is the kind of thing that quietly introduces bias. :func:`cuped_study` reports the
  mean estimate against the known true effect so the answer is visible.
* **Does the reduction actually equal ``1 - rho^2``?** :func:`variance_reduction_curve` sweeps
  the covariate correlation and puts the measured reduction next to the prediction. The
  binary case is where this gets interesting -- see the note on attenuation below.

A note on binary metrics
------------------------
:mod:`ab_testing_kit.simulation` generates a binary covariate through a latent-normal copula,
so ``ExperimentSpec.covariate_corr`` is the correlation of the *latent* normals, and the
realised correlation between the two 0/1 indicators is substantially weaker. CUPED delivers
``1 - rho^2`` against the **realised** correlation, not the requested one, and that gap is
large enough to look like a bug if you are not expecting it. Everything here is therefore
reported against :meth:`~ab_testing_kit.simulation.Experiment.covariate_correlation`, which
measures what actually came out.

A note on composing this with the other modules
-----------------------------------------------
:func:`cuped_adjust` estimates ``theta`` from the whole experiment it is handed. Adjusting
first and calling :meth:`~ab_testing_kit.simulation.Experiment.truncate` afterwards would
therefore let the first look use a ``theta`` fitted on data that has not arrived yet. If CUPED
is ever combined with peeking or alpha spending, the adjustment has to happen *inside* each
look, on the truncated experiment. :func:`cuped_at_each_look` does exactly that, and exists so
the wrong composition is harder to write than the right one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ab_testing_kit.simulation import (
    Experiment,
    ExperimentSpec,
    SimulationSummary,
    TestResult,
    fixed_horizon_test,
    generate_many,
    summarize,
)

__all__ = [
    "CupedComparison",
    "CupedResult",
    "cuped_adjust",
    "cuped_at_each_look",
    "cuped_study",
    "cuped_test",
    "cuped_theta",
    "variance_reduction_curve",
]


def cuped_theta(experiment: Experiment) -> float:
    """Variance-minimising adjustment coefficient, ``Cov(Y, X) / Var(X)``.

    Estimated on **arm-centred** outcomes and pooled across arms. Centring matters: if the
    arms were pooled raw, a real treatment effect would show up as covariance between the
    outcome and nothing in particular, inflating ``theta`` in proportion to the effect being
    measured. Centring first means ``theta`` describes the covariate relationship only.

    Returns ``0.0`` for a degenerate covariate (no variation), which reduces CUPED to a no-op
    rather than dividing by zero. That case is reachable: a low-rate binary covariate on a
    small experiment can come out all zeros.
    """
    pre = np.concatenate([experiment.control_pre, experiment.treatment_pre])
    post = np.concatenate(
        [
            experiment.control - experiment.control.mean(),
            experiment.treatment - experiment.treatment.mean(),
        ]
    )
    var_pre = float(pre.var())
    if var_pre == 0.0:
        return 0.0
    return float(((pre - pre.mean()) * post).mean() / var_pre)


def cuped_adjust(experiment: Experiment, theta: float | None = None) -> Experiment:
    """Return ``experiment`` with CUPED-adjusted outcomes.

    Args:
        experiment: The experiment to adjust.
        theta: Adjustment coefficient. Estimated from ``experiment`` when omitted.

    The covariate is centred at its **pooled** mean, so the same constant is subtracted from
    both arms and cancels out of the difference in means. What does not cancel is the chance
    imbalance in the covariate between arms -- which is precisely the noise CUPED removes.

    The returned experiment's spec is marked ``continuous`` even when the input was binary,
    because adjusted outcomes are no longer 0/1 and must be analysed with a t-test rather than
    a proportion test. Its ``sd`` carries the realised pooled standard deviation of the
    adjusted outcome, so the variance reduction is legible from the spec alone.

    Warning:
        ``theta`` is fitted on the whole experiment. Do not adjust and then truncate; see
        :func:`cuped_at_each_look`.
    """
    coefficient = cuped_theta(experiment) if theta is None else theta
    pre_mean = float(np.concatenate([experiment.control_pre, experiment.treatment_pre]).mean())
    control = experiment.control - coefficient * (experiment.control_pre - pre_mean)
    treatment = experiment.treatment - coefficient * (experiment.treatment_pre - pre_mean)

    pooled_sd = float(
        np.concatenate([control - control.mean(), treatment - treatment.mean()]).std(ddof=1)
    )
    spec = experiment.spec
    return Experiment(
        spec=ExperimentSpec(
            n_per_arm=spec.n_per_arm,
            metric="continuous",
            baseline=spec.baseline,
            effect=spec.effect,
            # A perfectly predicted outcome would leave zero residual variance, which the spec
            # rejects. Floor it rather than crash: the number is descriptive, not generative.
            sd=max(pooled_sd, 1e-12),
            covariate_corr=spec.covariate_corr,
        ),
        control=control,
        treatment=treatment,
        control_pre=experiment.control_pre,
        treatment_pre=experiment.treatment_pre,
    )


@dataclass(frozen=True)
class CupedResult:
    """One experiment analysed both ways, so the comparison is exact."""

    theta: float
    correlation: float
    """Realised pooled covariate-outcome correlation. The attenuated one, for binary metrics."""

    raw: TestResult
    adjusted: TestResult

    @property
    def variance_reduction(self) -> float:
        """Fraction of the estimator's variance removed. Compare against ``correlation**2``."""
        if self.raw.se == 0.0:
            return 0.0
        return 1.0 - (self.adjusted.se / self.raw.se) ** 2

    @property
    def predicted_variance_reduction(self) -> float:
        """What the theory says this covariate should have bought: ``rho^2``."""
        return self.correlation**2


def cuped_test(experiment: Experiment, alpha: float = 0.05) -> CupedResult:
    """Analyse ``experiment`` with and without the CUPED adjustment.

    Both tests run on the same units, so any difference between them is the adjustment and
    nothing else. The raw test uses whatever is right for the original metric (a pooled
    z-test for a rate, Welch's t for a mean); the adjusted test is always Welch's t, because
    an adjusted binary outcome is no longer binary.
    """
    adjusted = cuped_adjust(experiment)
    return CupedResult(
        theta=cuped_theta(experiment),
        correlation=experiment.covariate_correlation(),
        raw=fixed_horizon_test(experiment, alpha=alpha),
        adjusted=fixed_horizon_test(adjusted, alpha=alpha),
    )


def cuped_at_each_look(experiment: Experiment, n_per_arm: int, alpha: float = 0.05) -> TestResult:
    """Truncate first, then adjust -- the safe order for sequential use.

    Fitting ``theta`` on the full experiment and then slicing would let an interim analysis
    borrow information from units that have not arrived, which inflates the apparent precision
    of early looks. Truncating first keeps every look honest about what it knew at the time.
    """
    return fixed_horizon_test(cuped_adjust(experiment.truncate(n_per_arm)), alpha=alpha)


@dataclass(frozen=True)
class CupedComparison:
    """Aggregate of one CUPED study: the same replications, analysed both ways."""

    raw: SimulationSummary
    adjusted: SimulationSummary
    mean_correlation: float
    mean_theta: float
    mean_variance_reduction: float
    predicted_variance_reduction: float
    """Mean of ``rho^2`` over replications, using the realised correlation."""

    @property
    def power_gain(self) -> float:
        """Extra power the adjustment bought at the same sample size."""
        return self.adjusted.reject_rate - self.raw.reject_rate

    @property
    def effective_sample_size_multiplier(self) -> float:
        """Sample-size factor the variance reduction is worth.

        Variance falls as ``1/n``, so removing a fraction ``r`` of it is worth the same as
        multiplying the sample size by ``1 / (1 - r)``. This is the number to quote to
        someone deciding whether the covariate join is worth building.
        """
        remaining = 1.0 - self.mean_variance_reduction
        if remaining <= 0.0:
            return float("inf")
        return 1.0 / remaining


def cuped_study(
    spec: ExperimentSpec,
    n_sims: int = 5_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> CupedComparison:
    """Replicate ``spec`` and analyse every replication both ways.

    With ``spec.effect == 0`` the two ``reject_rate`` values are Type-I errors, and CUPED must
    not inflate its own -- reducing variance is only worth having if the test still holds its
    size. With a non-zero effect they are powers, and the gap is what the covariate bought.
    """
    raw_rejected: list[bool] = []
    raw_estimates: list[float] = []
    adj_rejected: list[bool] = []
    adj_estimates: list[float] = []
    correlations: list[float] = []
    thetas: list[float] = []
    reductions: list[float] = []

    for experiment in generate_many(spec, n_sims, seed):
        result = cuped_test(experiment, alpha=alpha)
        raw_rejected.append(result.raw.significant)
        raw_estimates.append(result.raw.estimate)
        adj_rejected.append(result.adjusted.significant)
        adj_estimates.append(result.adjusted.estimate)
        correlations.append(result.correlation)
        thetas.append(result.theta)
        reductions.append(result.variance_reduction)

    ns = [spec.n_per_arm] * n_sims
    return CupedComparison(
        raw=summarize(raw_rejected, raw_estimates, ns, spec.effect),
        adjusted=summarize(adj_rejected, adj_estimates, ns, spec.effect),
        mean_correlation=float(np.mean(correlations)),
        mean_theta=float(np.mean(thetas)),
        mean_variance_reduction=float(np.mean(reductions)),
        predicted_variance_reduction=float(np.mean(np.square(correlations))),
    )


def variance_reduction_curve(
    spec: ExperimentSpec,
    correlations: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8),
    n_sims: int = 2_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[float, CupedComparison]:
    """Measured variance reduction as a function of covariate quality.

    ``spec.covariate_corr`` is replaced by each value in ``correlations`` in turn; every other
    field is held fixed, and every sweep point reuses the same ``seed``. The result is the
    curve the method's central promise lives on: reduction should track ``1 - rho^2`` against
    the **realised** correlation, which for a binary metric is not the number requested here.
    """
    return {
        rho: cuped_study(
            ExperimentSpec(
                n_per_arm=spec.n_per_arm,
                metric=spec.metric,
                baseline=spec.baseline,
                effect=spec.effect,
                sd=spec.sd,
                covariate_corr=rho,
            ),
            n_sims=n_sims,
            seed=seed,
            alpha=alpha,
        )
        for rho in correlations
    }
