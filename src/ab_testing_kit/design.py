"""Power, sample size and minimum detectable effect, for the tests this toolkit runs.

Everything here is two-sided at the stated alpha. One-sided designs are deliberately not
offered: they would need a matching one-sided analysis path in
:func:`ab_testing_kit.simulation.fixed_horizon_test`, and shipping a parameter whose
behaviour is never validated by simulation would undercut the point of the repository.

The proportions formula is sized for a *pooled* two-proportion z-test, matching the test
:func:`~ab_testing_kit.simulation.fixed_horizon_test` actually applies. Sizing for one test
and analysing with another is a quiet and common way to end up under-powered; here the two
are kept in step on purpose, and ``validate_power`` checks that they are.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy import optimize, stats

from ab_testing_kit.simulation import (
    ExperimentSpec,
    MetricType,
    SimulationSummary,
    fixed_horizon_test,
    generate_many,
    summarize,
)

__all__ = [
    "PowerCheck",
    "analytic_power",
    "mde",
    "sample_size",
    "validate_power",
]


def _z(alpha: float, power: float) -> tuple[float, float]:
    """Critical value and power quantile for a two-sided test."""
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if not 0.0 < power < 1.0:
        raise ValueError(f"power must be in (0, 1), got {power}")
    return float(stats.norm.ppf(1.0 - alpha / 2.0)), float(stats.norm.ppf(power))


def sample_size(
    metric: MetricType,
    baseline: float,
    mde_abs: float,
    sd: float = 1.0,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """Units required *per arm* to detect ``mde_abs`` with the stated power.

    Args:
        metric: ``"binary"`` or ``"continuous"``.
        baseline: Control rate or control mean.
        mde_abs: Absolute effect to detect. Sign is ignored; the test is two-sided.
        sd: Within-arm standard deviation. Continuous metrics only.
        alpha: Two-sided significance level.
        power: Target power.

    Returns:
        Sample size per arm, rounded up.
    """
    z_a, z_b = _z(alpha, power)
    delta = abs(mde_abs)
    if delta <= 0.0:
        raise ValueError(f"mde_abs must be non-zero, got {mde_abs}")

    if metric == "binary":
        p0 = baseline
        p1 = baseline + delta
        if not 0.0 < p0 < 1.0:
            raise ValueError(f"binary baseline must be in (0, 1), got {p0}")
        if not 0.0 < p1 < 1.0:
            raise ValueError(f"baseline + mde must be in (0, 1), got {p1}")
        p_bar = (p0 + p1) / 2.0
        pooled = z_a * math.sqrt(2.0 * p_bar * (1.0 - p_bar))
        unpooled = z_b * math.sqrt(p0 * (1.0 - p0) + p1 * (1.0 - p1))
        return math.ceil((pooled + unpooled) ** 2 / delta**2)

    if metric != "continuous":
        raise ValueError(f"metric must be 'binary' or 'continuous', got {metric!r}")
    if sd <= 0.0:
        raise ValueError(f"sd must be positive, got {sd}")
    return math.ceil(2.0 * (z_a + z_b) ** 2 * sd**2 / delta**2)


def analytic_power(
    metric: MetricType,
    baseline: float,
    effect: float,
    n_per_arm: int,
    sd: float = 1.0,
    alpha: float = 0.05,
) -> float:
    """Power of the two-sided fixed-horizon test at this sample size.

    Uses the normal approximation, so it ignores the t-distribution's heavier tails for
    continuous metrics. At small ``n_per_arm`` that makes it mildly optimistic;
    :func:`validate_power` is what surfaces the size of that gap rather than hiding it.
    """
    z_a, _ = _z(alpha, 0.5)
    delta = abs(effect)
    if n_per_arm < 2:
        raise ValueError(f"n_per_arm must be at least 2, got {n_per_arm}")
    if delta == 0.0:
        return alpha

    if metric == "binary":
        p0, p1 = baseline, baseline + effect
        if not 0.0 < p0 < 1.0 or not 0.0 < p1 < 1.0:
            raise ValueError("baseline and baseline + effect must both be in (0, 1)")
        p_bar = (p0 + p1) / 2.0
        null_se = math.sqrt(2.0 * p_bar * (1.0 - p_bar) / n_per_arm)
        alt_se = math.sqrt((p0 * (1.0 - p0) + p1 * (1.0 - p1)) / n_per_arm)
    elif metric == "continuous":
        if sd <= 0.0:
            raise ValueError(f"sd must be positive, got {sd}")
        null_se = alt_se = sd * math.sqrt(2.0 / n_per_arm)
    else:
        raise ValueError(f"metric must be 'binary' or 'continuous', got {metric!r}")

    return float(stats.norm.cdf((delta - z_a * null_se) / alt_se))


def mde(
    metric: MetricType,
    baseline: float,
    n_per_arm: int,
    sd: float = 1.0,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """Smallest absolute effect detectable at this sample size, with the stated power.

    The continuous case inverts the sample-size formula in closed form. The binary case
    cannot: the effect appears inside the variance as well as in the numerator, so it is
    solved numerically against :func:`analytic_power`.
    """
    z_a, z_b = _z(alpha, power)
    if n_per_arm < 2:
        raise ValueError(f"n_per_arm must be at least 2, got {n_per_arm}")

    if metric == "continuous":
        if sd <= 0.0:
            raise ValueError(f"sd must be positive, got {sd}")
        return (z_a + z_b) * sd * math.sqrt(2.0 / n_per_arm)

    if metric != "binary":
        raise ValueError(f"metric must be 'binary' or 'continuous', got {metric!r}")
    if not 0.0 < baseline < 1.0:
        raise ValueError(f"binary baseline must be in (0, 1), got {baseline}")

    headroom = 1.0 - baseline
    lo, hi = 1e-9, headroom * (1.0 - 1e-9)

    def shortfall(delta: float) -> float:
        return analytic_power("binary", baseline, delta, n_per_arm, alpha=alpha) - power

    if shortfall(hi) < 0.0:
        raise ValueError(
            f"no detectable effect at n_per_arm={n_per_arm} for baseline={baseline}: "
            f"even a lift to 1.0 gives power {shortfall(hi) + power:.3f} < {power}"
        )
    return float(optimize.brentq(shortfall, lo, hi, xtol=1e-12))


@dataclass(frozen=True)
class PowerCheck:
    """Analytic power next to the power actually observed in simulation."""

    spec: ExperimentSpec
    alpha: float
    analytic: float
    simulated: SimulationSummary

    @property
    def gap(self) -> float:
        """Simulated power minus analytic power. Negative means the formula over-promises."""
        return self.simulated.reject_rate - self.analytic

    @property
    def agrees(self) -> bool:
        """Whether the analytic value falls inside the simulation's Monte Carlo interval."""
        lo, hi = self.simulated.reject_rate_ci
        return lo <= self.analytic <= hi


def validate_power(
    spec: ExperimentSpec,
    n_sims: int = 5_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> PowerCheck:
    """Run ``spec`` many times and compare observed rejection rate to the formula.

    This is the check the sample-size calculator is worth having: size an experiment, run
    it thousands of times against a known true effect, and confirm the fraction of runs
    that reach significance is the power that was asked for.
    """
    rejected: list[bool] = []
    estimates: list[float] = []
    ns: list[int] = []
    for experiment in generate_many(spec, n_sims, seed):
        result = fixed_horizon_test(experiment, alpha=alpha)
        rejected.append(result.significant)
        estimates.append(result.estimate)
        ns.append(result.n_per_arm)

    return PowerCheck(
        spec=spec,
        alpha=alpha,
        analytic=analytic_power(
            spec.metric, spec.baseline, spec.effect, spec.n_per_arm, spec.sd, alpha
        ),
        simulated=summarize(rejected, estimates, ns, spec.effect),
    )
