"""Seeded experiment generation and the common result types.

This module is the substrate the rest of the toolkit is measured against. Every other
module answers a statistical question about experiments produced here, which is what lets
the repository *check* its claims rather than assert them: the true effect is known, so a
measured rejection rate can be compared against the rate the method promises.

Two design choices matter downstream:

* **Units are stored in arrival order.** Outcomes are i.i.d., so the first ``k`` rows of an
  arm are themselves a valid random sample of size ``k``. That is what makes
  :meth:`Experiment.truncate` an honest model of "the data as it looked on day 3", and
  therefore what makes peeking simulable at all.
* **Each replication gets its own spawned seed.** Replication ``i`` of a study is identical
  whether the study runs 100 replications or 100,000, so a surprising replication can be
  pulled out and inspected on its own.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import stats

MetricType = Literal["binary", "continuous"]

__all__ = [
    "Experiment",
    "ExperimentSpec",
    "SimulationSummary",
    "TestResult",
    "fixed_horizon_test",
    "generate",
    "generate_many",
    "summarize",
    "wilson_interval",
]


@dataclass(frozen=True)
class ExperimentSpec:
    """The ground truth of a simulated experiment.

    Args:
        n_per_arm: Units allocated to each arm (this toolkit assumes a balanced split).
        metric: ``"binary"`` for a conversion rate, ``"continuous"`` for a mean.
        baseline: Control conversion rate, or control mean.
        effect: True *absolute* treatment effect. ``0.0`` generates data under the null.
        sd: Within-arm standard deviation. Continuous metrics only; ignored for binary,
            where the variance is fixed by the rate.
        covariate_corr: Correlation between each unit's pre-experiment covariate and its
            outcome, used by CUPED. For continuous metrics this is the realised Pearson
            correlation. For binary metrics it is the correlation of the *latent* normals
            behind the two indicators; the realised correlation between the two 0/1 values
            is attenuated, so read it off :meth:`Experiment.covariate_correlation` rather
            than assuming it equals this argument.
    """

    n_per_arm: int
    metric: MetricType
    baseline: float
    effect: float = 0.0
    sd: float = 1.0
    covariate_corr: float = 0.0

    def __post_init__(self) -> None:
        if self.n_per_arm < 2:
            raise ValueError(f"n_per_arm must be at least 2, got {self.n_per_arm}")
        if self.metric not in ("binary", "continuous"):
            raise ValueError(f"metric must be 'binary' or 'continuous', got {self.metric!r}")
        if not -1.0 < self.covariate_corr < 1.0:
            raise ValueError(f"covariate_corr must be in (-1, 1), got {self.covariate_corr}")
        if self.metric == "binary":
            if not 0.0 < self.baseline < 1.0:
                raise ValueError(f"binary baseline must be in (0, 1), got {self.baseline}")
            if not 0.0 < self.treatment_mean < 1.0:
                raise ValueError(
                    f"binary baseline + effect must be in (0, 1), got {self.treatment_mean}"
                )
        elif self.sd <= 0.0:
            raise ValueError(f"sd must be positive for a continuous metric, got {self.sd}")

    @property
    def treatment_mean(self) -> float:
        """The true treatment-arm mean."""
        return self.baseline + self.effect


@dataclass(frozen=True, eq=False)
class Experiment:
    """One realised experiment. Arrays are per-unit outcomes in arrival order.

    ``eq=False`` because the fields are numpy arrays: the generated ``__eq__`` would return
    an array and blow up on any truth test.
    """

    spec: ExperimentSpec
    control: np.ndarray
    treatment: np.ndarray
    control_pre: np.ndarray
    treatment_pre: np.ndarray

    def truncate(self, n_per_arm: int) -> Experiment:
        """Return the experiment as it stood after the first ``n_per_arm`` units per arm."""
        if not 2 <= n_per_arm <= self.spec.n_per_arm:
            raise ValueError(f"n_per_arm must be in [2, {self.spec.n_per_arm}], got {n_per_arm}")
        return Experiment(
            spec=ExperimentSpec(
                n_per_arm=n_per_arm,
                metric=self.spec.metric,
                baseline=self.spec.baseline,
                effect=self.spec.effect,
                sd=self.spec.sd,
                covariate_corr=self.spec.covariate_corr,
            ),
            control=self.control[:n_per_arm],
            treatment=self.treatment[:n_per_arm],
            control_pre=self.control_pre[:n_per_arm],
            treatment_pre=self.treatment_pre[:n_per_arm],
        )

    def covariate_correlation(self) -> float:
        """Realised pooled correlation between the pre-experiment covariate and the outcome.

        Pooled across arms after centring each arm, so a treatment effect does not leak into
        the estimate.
        """
        pre = np.concatenate([self.control_pre, self.treatment_pre])
        post = np.concatenate(
            [self.control - self.control.mean(), self.treatment - self.treatment.mean()]
        )
        if pre.std() == 0.0 or post.std() == 0.0:
            return 0.0
        return float(np.corrcoef(pre, post)[0, 1])


def _bivariate_normal(n: int, rho: float, rng: np.random.Generator) -> tuple[np.ndarray, ...]:
    """Two standard normal columns with correlation ``rho``."""
    z = rng.standard_normal((n, 2))
    return z[:, 0], rho * z[:, 0] + np.sqrt(1.0 - rho**2) * z[:, 1]


def _arm(mean: float, spec: ExperimentSpec, rng: np.random.Generator) -> tuple[np.ndarray, ...]:
    """Generate one arm's (pre-experiment covariate, outcome) pair."""
    z_pre, z_out = _bivariate_normal(spec.n_per_arm, spec.covariate_corr, rng)
    if spec.metric == "binary":
        # Latent-normal copula: threshold both normals at the quantile matching each rate.
        pre = (z_pre < stats.norm.ppf(spec.baseline)).astype(float)
        out = (z_out < stats.norm.ppf(mean)).astype(float)
        return pre, out
    return spec.baseline + spec.sd * z_pre, mean + spec.sd * z_out


def generate(spec: ExperimentSpec, rng: np.random.Generator) -> Experiment:
    """Draw one experiment from ``spec``."""
    control_pre, control = _arm(spec.baseline, spec, rng)
    treatment_pre, treatment = _arm(spec.treatment_mean, spec, rng)
    return Experiment(
        spec=spec,
        control=control,
        treatment=treatment,
        control_pre=control_pre,
        treatment_pre=treatment_pre,
    )


def generate_many(spec: ExperimentSpec, n_sims: int, seed: int) -> Iterator[Experiment]:
    """Yield ``n_sims`` independent experiments.

    Each replication draws from its own spawned seed, so replication ``i`` is identical
    regardless of how many replications the caller asked for.
    """
    if n_sims < 1:
        raise ValueError(f"n_sims must be at least 1, got {n_sims}")
    for child in np.random.SeedSequence(seed).spawn(n_sims):
        yield generate(spec, np.random.default_rng(child))


@dataclass(frozen=True)
class TestResult:
    """The outcome of analysing one experiment at one point in time."""

    estimate: float
    se: float
    statistic: float
    p_value: float
    significant: bool
    n_per_arm: int
    ci: tuple[float, float]


def fixed_horizon_test(experiment: Experiment, alpha: float = 0.05) -> TestResult:
    """Two-sided fixed-horizon test of treatment minus control.

    Binary metrics get a pooled two-proportion z-test; continuous metrics get Welch's
    t-test. Both are the conventional choice, and the pooled z-test in particular is the
    test :mod:`ab_testing_kit.design` sizes for, so the design formula and the analysis
    agree by construction.

    The confidence interval deliberately uses the *unpooled* standard error while the
    p-value uses the pooled one. Pooling assumes the null is true, which is right for a
    test statistic and wrong for an interval estimate; the two can therefore disagree at
    the margin, and that is correct rather than a bug.

    A degenerate arm (every unit identical, so the standard error is zero) is reported as
    ``p_value=1.0`` and not significant rather than raising. That case turns up at the
    first look of an aggressive peeking schedule on a low-rate binary metric.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    control, treatment = experiment.control, experiment.treatment
    n_c, n_t = control.size, treatment.size
    estimate = float(treatment.mean() - control.mean())

    if experiment.spec.metric == "binary":
        p_c, p_t = float(control.mean()), float(treatment.mean())
        p_pool = (control.sum() + treatment.sum()) / (n_c + n_t)
        se_pooled = float(np.sqrt(p_pool * (1.0 - p_pool) * (1.0 / n_c + 1.0 / n_t)))
        se = float(np.sqrt(p_c * (1.0 - p_c) / n_c + p_t * (1.0 - p_t) / n_t))
        crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
        if se_pooled == 0.0:
            statistic, p_value = 0.0, 1.0
        else:
            statistic = estimate / se_pooled
            p_value = float(2.0 * stats.norm.sf(abs(statistic)))
    else:
        var_c = float(control.var(ddof=1))
        var_t = float(treatment.var(ddof=1))
        se = float(np.sqrt(var_c / n_c + var_t / n_t))
        if se == 0.0:
            statistic, p_value, crit = 0.0, 1.0, 0.0
        else:
            statistic = estimate / se
            # Welch-Satterthwaite degrees of freedom.
            df = (var_c / n_c + var_t / n_t) ** 2 / (
                (var_c / n_c) ** 2 / (n_c - 1) + (var_t / n_t) ** 2 / (n_t - 1)
            )
            p_value = float(2.0 * stats.t.sf(abs(statistic), df))
            crit = float(stats.t.ppf(1.0 - alpha / 2.0, df))

    return TestResult(
        estimate=estimate,
        se=se,
        statistic=float(statistic),
        p_value=float(p_value),
        significant=bool(p_value < alpha),
        n_per_arm=n_c,
        ci=(estimate - crit * se, estimate + crit * se),
    )


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a proportion.

    Used here on the *rejection rate itself*. A measured Type-I error of 0.0512 from 10,000
    replications carries Monte Carlo error of roughly +/- 0.004, and a claim that a method
    "controls error at 5%" only means something once that uncertainty is stated alongside it.
    """
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    z = float(stats.norm.ppf(0.5 + confidence / 2.0))
    p = successes / n
    denom = 1.0 + z**2 / n
    centre = (p + z**2 / (2.0 * n)) / denom
    half = z * np.sqrt(p * (1.0 - p) / n + z**2 / (4.0 * n**2)) / denom
    # The bounds are exactly 0 and 1 at the extremes; pin them rather than letting float
    # error leave a lower bound of 3e-18 on a study that never once rejected.
    lo = 0.0 if successes == 0 else max(0.0, float(centre - half))
    hi = 1.0 if successes == n else min(1.0, float(centre + half))
    return (lo, hi)


@dataclass(frozen=True)
class SimulationSummary:
    """Aggregate of one study: many replications of the same spec under the same method."""

    n_sims: int
    n_rejected: int
    reject_rate: float
    reject_rate_ci: tuple[float, float]
    mean_estimate: float
    mean_abs_estimate: float
    mean_n_per_arm: float
    true_effect: float

    @property
    def estimate_bias(self) -> float:
        """Mean reported effect minus the true effect, over replications that rejected.

        Non-zero here is the winner's curse: a rule that stops as soon as the estimate looks
        large reports estimates that are, on average, too large.

        This is a *signed* mean, which makes it the right summary under a true alternative
        and a misleading one under the null: a two-sided rule stops on spuriously-positive
        and spuriously-negative replications in roughly equal numbers, so the signed mean
        cancels to ~0 no matter how inflated the individual estimates are. Read
        ``mean_abs_estimate`` in that case. Session 1 nearly shipped the cancellation as a
        null finding; see README 5.3.
        """
        return self.mean_estimate - self.true_effect


def summarize(
    rejected: Sequence[bool],
    estimates: Sequence[float],
    ns: Sequence[int],
    true_effect: float,
    confidence: float = 0.95,
) -> SimulationSummary:
    """Aggregate per-replication outcomes into a :class:`SimulationSummary`.

    ``mean_estimate`` and ``mean_abs_estimate`` are both taken over the replications that
    *rejected*, because that is the subset an experimenter would actually act on and ship.
    """
    n_sims = len(rejected)
    if n_sims == 0:
        raise ValueError("cannot summarize zero replications")
    if not len(estimates) == len(ns) == n_sims:
        raise ValueError("rejected, estimates and ns must be the same length")

    rejected_arr = np.asarray(rejected, dtype=bool)
    n_rejected = int(rejected_arr.sum())
    winners = np.asarray(estimates, dtype=float)[rejected_arr]
    return SimulationSummary(
        n_sims=n_sims,
        n_rejected=n_rejected,
        reject_rate=n_rejected / n_sims,
        reject_rate_ci=wilson_interval(n_rejected, n_sims, confidence),
        mean_estimate=float(winners.mean()) if n_rejected else float("nan"),
        mean_abs_estimate=float(np.abs(winners).mean()) if n_rejected else float("nan"),
        mean_n_per_arm=float(np.asarray(ns, dtype=float).mean()),
        true_effect=true_effect,
    )
