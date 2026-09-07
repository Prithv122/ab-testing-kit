"""Bayesian decision rules on the same experiments, and what they do and do not fix.

The usual framing is that this is a philosophical choice. It mostly is not. Run a Bayesian
analysis and a frequentist one on identical data with weak priors and you get two numbers
that are, arithmetically, nearly the same number -- ``P(treatment > control)`` and
``1 - p/2`` agree to about three decimal places at any sample size a product team would
actually run. What differs is what you are entitled to *say* about it, and which decision
rule you attach to it.

So this module measures three things rather than arguing about one:

1. **How closely do the two agree on the same experiments?** :func:`agreement_study` counts
   how often the two rules reach different verdicts. The answer is "almost always", and the
   residual disagreement turns out to be a *threshold* mismatch rather than a framework one:
   the conventional 0.95 posterior threshold is a one-sided 5% rule, twice as permissive in
   the ship direction as the conventional two-sided 5% test it gets compared to.
2. **What is the frequentist error rate of a Bayesian rule?** A posterior probability is not
   a Type-I error rate, but a rule that ships whenever ``P(B > A) > 0.95`` still *has* one,
   and a team running it is still wrong some fraction of the time. :func:`bayes_study`
   measures that fraction against a known null.
3. **Does going Bayesian fix peeking?** It is widely claimed that it does -- that posteriors
   may be monitored continuously without penalty. :func:`bayes_peeking_study` runs a Bayesian
   stopping rule through the *same* peeking substrate as :mod:`ab_testing_kit.peeking` and
   reports the answer. The result is in README 5.6, and it is not the reassuring one.

The claim being tested in (3) is worth stating precisely, because it is usually stated
imprecisely. It is true that a posterior distribution is not invalidated by having been
looked at: the posterior given the data is the posterior given the data, whenever you compute
it. What is false is the inference people draw from that -- that a *stopping rule* built on a
posterior threshold therefore controls the rate at which it ships losers. It does not,
because that rate is a frequentist property and nothing about Bayesian machinery promises it.

Expected loss
-------------
The other decision rule implemented here is the one that actually justifies the Bayesian
framing in a product setting. ``P(B > A) = 0.96`` tells you the probability you are right;
it says nothing about what being wrong would cost. Expected loss does:

    E[max(control - treatment, 0)]

is the average amount of metric surrendered by shipping, integrated over the posterior. A
team can set a threshold on it in units of the metric they care about -- "ship if the
expected cost of being wrong is under 0.1 percentage points" -- which is a business
statement rather than a statistical one. That is a genuine advantage over a p-value, and it
survives everything else this module finds.

Normal posteriors
-----------------
Both metrics use a normal posterior for the arm mean: exact for a continuous metric with a
flat prior, and an approximation to the Beta-Binomial posterior for a rate. The
approximation is excellent at the sample sizes here -- ``test_normal_posterior_matches_the_
exact_beta`` pins it against direct sampling from the conjugate posterior -- and it keeps a
20,000-replication study to seconds rather than hours. Priors are weak by default (Beta(1,1),
i.e. uniform) and are exposed so the effect of strengthening them is measurable rather than
hypothetical.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

from ab_testing_kit.peeking import MIN_LOOK_SIZE, peek_schedule
from ab_testing_kit.simulation import (
    Experiment,
    ExperimentSpec,
    SimulationSummary,
    fixed_horizon_test,
    generate_many,
    summarize,
    wilson_interval,
)

__all__ = [
    "AgreementSummary",
    "BayesResult",
    "DecisionRule",
    "agreement_study",
    "analyse",
    "bayes_peeking_study",
    "bayes_study",
    "difference_posterior",
]


@dataclass(frozen=True)
class DecisionRule:
    """When to ship, in Bayesian terms.

    Args:
        prob_threshold: Ship once ``P(treatment > control)`` exceeds this. ``0.95`` is the
            conventional choice, and it is worth being precise about what it corresponds to:
            a **one-sided** 5% test, not a two-sided one. A conventional two-sided alpha of
            0.05 only ships on a positive result at 2.5%, so the industry-standard 0.95
            posterior threshold is **twice as permissive in the ship direction** as the
            industry-standard p-value threshold. Pass ``0.975`` for the genuinely
            like-for-like comparison; :func:`agreement_study` measures the gap either way.
        max_expected_loss: Ship only if the expected loss is also below this. ``None``
            disables the check, leaving a pure probability rule. Expressed in units of the
            metric, so 0.001 on a conversion rate means "0.1 percentage points".
    """

    prob_threshold: float = 0.95
    max_expected_loss: float | None = None

    def __post_init__(self) -> None:
        if not 0.5 < self.prob_threshold < 1.0:
            raise ValueError(f"prob_threshold must be in (0.5, 1), got {self.prob_threshold}")
        if self.max_expected_loss is not None and self.max_expected_loss <= 0.0:
            raise ValueError(f"max_expected_loss must be positive, got {self.max_expected_loss}")


def _arm_posterior(
    outcomes: np.ndarray, metric: str, prior_a: float, prior_b: float
) -> tuple[float, float]:
    """Posterior mean and standard deviation for one arm's metric.

    For a rate: the Beta(``prior_a`` + successes, ``prior_b`` + failures) conjugate posterior,
    summarised by its mean and standard deviation. For a mean: a flat prior, which puts the
    posterior at the sample mean with the standard error as its spread.
    """
    n = outcomes.size
    if metric == "binary":
        successes = float(outcomes.sum())
        a = prior_a + successes
        b = prior_b + (n - successes)
        total = a + b
        mean = a / total
        return mean, math.sqrt(mean * (1.0 - mean) / (total + 1.0))
    return float(outcomes.mean()), float(outcomes.std(ddof=1) / math.sqrt(n))


def difference_posterior(
    experiment: Experiment, prior_a: float = 1.0, prior_b: float = 1.0
) -> tuple[float, float]:
    """Posterior mean and standard deviation of ``treatment - control``.

    The arms are independent given the data, so the difference of two normal posteriors is
    normal with variances added. Returns ``(mean, sd)``.
    """
    if prior_a <= 0.0 or prior_b <= 0.0:
        raise ValueError(f"Beta prior parameters must be positive, got ({prior_a}, {prior_b})")
    metric = experiment.spec.metric
    control_mean, control_sd = _arm_posterior(experiment.control, metric, prior_a, prior_b)
    treatment_mean, treatment_sd = _arm_posterior(experiment.treatment, metric, prior_a, prior_b)
    return treatment_mean - control_mean, math.hypot(control_sd, treatment_sd)


@dataclass(frozen=True)
class BayesResult:
    """The posterior summary a product team would actually be shown."""

    estimate: float
    """Posterior mean of ``treatment - control``."""

    sd: float
    """Posterior standard deviation of the difference."""

    prob_better: float
    """``P(treatment > control)`` under the posterior."""

    expected_loss_ship: float
    """``E[max(control - treatment, 0)]`` -- average metric surrendered by shipping."""

    expected_loss_keep: float
    """``E[max(treatment - control, 0)]`` -- average metric forgone by not shipping."""

    credible_interval: tuple[float, float]
    ship: bool
    n_per_arm: int


def _expected_shortfall(mean: float, sd: float) -> float:
    """``E[max(-D, 0)]`` for ``D ~ Normal(mean, sd)``.

    The closed form ``sd phi(mean/sd) - mean Phi(-mean/sd)``. A degenerate posterior (zero
    spread, which a fully one-sided arm can produce at a tiny first look) collapses to the
    obvious answer rather than dividing by zero.
    """
    if sd <= 0.0:
        return max(-mean, 0.0)
    z = mean / sd
    return float(sd * stats.norm.pdf(z) - mean * stats.norm.cdf(-z))


def analyse(
    experiment: Experiment,
    rule: DecisionRule | None = None,
    prior_a: float = 1.0,
    prior_b: float = 1.0,
    credible_mass: float = 0.95,
) -> BayesResult:
    """Summarise one experiment's posterior and apply the decision rule."""
    if not 0.0 < credible_mass < 1.0:
        raise ValueError(f"credible_mass must be in (0, 1), got {credible_mass}")
    decision = DecisionRule() if rule is None else rule

    mean, sd = difference_posterior(experiment, prior_a, prior_b)
    prob_better = float(stats.norm.sf(0.0, loc=mean, scale=sd)) if sd > 0.0 else float(mean > 0.0)
    loss_ship = _expected_shortfall(mean, sd)
    # E[max(D, 0)] is the same integral with the sign flipped.
    loss_keep = _expected_shortfall(-mean, sd)

    half = float(stats.norm.isf((1.0 - credible_mass) / 2.0)) * sd
    ship = prob_better >= decision.prob_threshold
    if decision.max_expected_loss is not None:
        ship = ship and loss_ship <= decision.max_expected_loss

    return BayesResult(
        estimate=mean,
        sd=sd,
        prob_better=prob_better,
        expected_loss_ship=loss_ship,
        expected_loss_keep=loss_keep,
        credible_interval=(mean - half, mean + half),
        ship=ship,
        n_per_arm=experiment.control.size,
    )


def bayes_study(
    spec: ExperimentSpec,
    rule: DecisionRule | None = None,
    n_sims: int = 10_000,
    seed: int = 0,
    prior_a: float = 1.0,
    prior_b: float = 1.0,
) -> SimulationSummary:
    """Replicate ``spec`` and count how often the Bayesian rule ships.

    Under a true null that count is the rule's **frequentist Type-I error rate**, which is a
    perfectly meaningful thing to ask about a Bayesian procedure even though the procedure
    itself never claims to control it. A team that ships on ``P(B > A) > 0.95`` is wrong at
    some rate, and they are entitled to know what it is.
    """
    decision = DecisionRule() if rule is None else rule
    shipped: list[bool] = []
    estimates: list[float] = []
    ns: list[int] = []
    for experiment in generate_many(spec, n_sims, seed):
        result = analyse(experiment, decision, prior_a, prior_b)
        shipped.append(result.ship)
        estimates.append(result.estimate)
        ns.append(result.n_per_arm)
    return summarize(shipped, estimates, ns, spec.effect)


@dataclass(frozen=True)
class AgreementSummary:
    """How often two decision rules reach the same verdict on identical data."""

    n_sims: int
    both_ship: int
    both_hold: int
    bayes_only: int
    frequentist_only: int
    mean_abs_probability_gap: float
    """Mean ``|P(B > A) - (1 - p/2)|``. Small means the two are computing the same number."""

    @property
    def n_disagreements(self) -> int:
        return self.bayes_only + self.frequentist_only

    @property
    def agreement_rate(self) -> float:
        return (self.both_ship + self.both_hold) / self.n_sims

    @property
    def agreement_ci(self) -> tuple[float, float]:
        return wilson_interval(self.both_ship + self.both_hold, self.n_sims)


def agreement_study(
    spec: ExperimentSpec,
    rule: DecisionRule | None = None,
    n_sims: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
    prior_a: float = 1.0,
    prior_b: float = 1.0,
) -> AgreementSummary:
    """Run both frameworks on every replication and count the disagreements.

    ``mean_abs_probability_gap`` is the part that makes the point. The frequentist two-sided
    p-value maps to a one-sided posterior-style quantity as ``1 - p/2`` when the estimate is
    positive (and ``p/2`` when it is negative); with weak priors the Bayesian
    ``prob_better`` lands on essentially the same value. The frameworks are not computing
    different things here. They are computing the same thing and licensing different
    sentences about it.

    Expect the disagreements to be almost entirely ``bayes_only``, and that is a threshold
    artefact rather than a philosophical one: at the default ``prob_threshold`` of 0.95 the
    Bayesian rule ships at a one-sided 5%, while a two-sided 5% test ships on a positive
    result at 2.5%. Setting ``prob_threshold=0.975`` collapses the two onto each other, which
    is the cleanest demonstration available that the gap was never about Bayes.
    """
    decision = DecisionRule() if rule is None else rule
    counts = {"both_ship": 0, "both_hold": 0, "bayes_only": 0, "frequentist_only": 0}
    gaps: list[float] = []

    for experiment in generate_many(spec, n_sims, seed):
        posterior = analyse(experiment, decision, prior_a, prior_b)
        frequentist = fixed_horizon_test(experiment, alpha=alpha)
        # A two-sided test that rejects on a negative estimate is not a decision to ship.
        ships_frequentist = frequentist.significant and frequentist.estimate > 0.0

        if posterior.ship and ships_frequentist:
            counts["both_ship"] += 1
        elif not posterior.ship and not ships_frequentist:
            counts["both_hold"] += 1
        elif posterior.ship:
            counts["bayes_only"] += 1
        else:
            counts["frequentist_only"] += 1

        one_sided = (
            1.0 - frequentist.p_value / 2.0
            if frequentist.estimate > 0.0
            else frequentist.p_value / 2.0
        )
        gaps.append(abs(posterior.prob_better - one_sided))

    return AgreementSummary(
        n_sims=n_sims,
        both_ship=counts["both_ship"],
        both_hold=counts["both_hold"],
        bayes_only=counts["bayes_only"],
        frequentist_only=counts["frequentist_only"],
        mean_abs_probability_gap=float(np.mean(gaps)),
    )


def bayes_peeking_study(
    spec: ExperimentSpec,
    n_looks: int,
    rule: DecisionRule | None = None,
    n_sims: int = 10_000,
    seed: int = 0,
    prior_a: float = 1.0,
    prior_b: float = 1.0,
    min_look_size: int = MIN_LOOK_SIZE,
) -> SimulationSummary:
    """A Bayesian stopping rule under continuous monitoring, on the same substrate.

    Identical analysis points to :func:`~ab_testing_kit.peeking.peeking_study` and
    :func:`~ab_testing_kit.sequential.sequential_study`, because it uses the same
    :func:`~ab_testing_kit.peeking.peek_schedule`. Ships at the first look where the rule is
    satisfied.

    With ``spec.effect == 0`` the returned ``reject_rate`` is how often the rule ships a
    loser. If posterior monitoring were self-correcting, that number would not move as looks
    are added. README 5.6 reports whether it does.
    """
    decision = DecisionRule() if rule is None else rule
    schedule = peek_schedule(spec.n_per_arm, n_looks, min_look_size)

    shipped: list[bool] = []
    estimates: list[float] = []
    ns: list[int] = []
    for experiment in generate_many(spec, n_sims, seed):
        # peek_schedule is never empty, so the first look always gives a result to fall back
        # on; the loop then replaces it only while the rule has not fired.
        final = analyse(experiment.truncate(schedule[0]), decision, prior_a, prior_b)
        for n in schedule[1:]:
            if final.ship:
                break
            final = analyse(experiment.truncate(n), decision, prior_a, prior_b)
        shipped.append(final.ship)
        estimates.append(final.estimate)
        ns.append(final.n_per_arm)

    return summarize(shipped, estimates, ns, spec.effect)
