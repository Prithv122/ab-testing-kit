"""What continuous monitoring does to a fixed-horizon test, measured rather than asserted.

The failure this module demonstrates is not that peeking is imprecise. It is that a
fixed-horizon test applied repeatedly to accumulating data no longer has the error rate
printed on the tin: each look is another chance for noise to cross the boundary, and the
chances accumulate. Nominal alpha stays 0.05 while the real rate climbs.

There is a second, quieter cost that gets far less attention than the first, so this module
measures it too: **the estimates that survive are inflated**. A rule that stops the moment
the effect looks big enough selects for replications where noise happened to run in the
treatment's favour, so the effect an experimenter reports is biased upward even when the
decision to ship was correct. That is the winner's curse, and it is why
:class:`~ab_testing_kit.simulation.SimulationSummary` averages estimates over the
replications that rejected rather than over all of them.

Both effects are reported against the same seeded experiments, so the peeking and
fixed-horizon numbers in the README differ only in the decision rule applied to them.
"""

from __future__ import annotations

from dataclasses import dataclass

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
    "PeekOutcome",
    "peek_schedule",
    "peeking_study",
    "run_with_peeking",
    "type_i_error_curve",
]

# Below roughly this many units per arm a two-proportion z-test is too unstable for its
# nominal alpha to mean anything, so an aggressive schedule on a small experiment would
# measure the normal approximation breaking down rather than the peeking effect itself.
MIN_LOOK_SIZE = 30


def peek_schedule(
    n_per_arm: int, n_looks: int, min_look_size: int = MIN_LOOK_SIZE
) -> tuple[int, ...]:
    """Evenly spaced interim analysis points, ending at the full sample.

    Args:
        n_per_arm: Planned final sample size per arm.
        n_looks: Number of analyses, including the final one. ``1`` is no peeking.
        min_look_size: Drop any look smaller than this. Looks are dropped rather than
            clamped, because clamping would silently produce duplicate analysis points.

    Returns:
        Increasing sample sizes per arm. Always ends at ``n_per_arm``.
    """
    if n_per_arm < 2:
        raise ValueError(f"n_per_arm must be at least 2, got {n_per_arm}")
    if n_looks < 1:
        raise ValueError(f"n_looks must be at least 1, got {n_looks}")

    raw = [round(n_per_arm * (i + 1) / n_looks) for i in range(n_looks)]
    # The final look is exactly n_per_arm, and the floor is capped at n_per_arm, so the
    # horizon always survives the filter -- the schedule cannot come back empty or short.
    return tuple(sorted({n for n in raw if n >= min(min_look_size, n_per_arm)}))


@dataclass(frozen=True)
class PeekOutcome:
    """What happened to one experiment run under a stop-at-first-significance rule."""

    looks: tuple[TestResult, ...]
    stopped_at_look: int | None
    """1-indexed look at which the boundary was first crossed; ``None`` if never."""

    @property
    def rejected(self) -> bool:
        return self.stopped_at_look is not None

    @property
    def final(self) -> TestResult:
        """The result an experimenter would report: at the stop, or at the full sample."""
        if self.stopped_at_look is None:
            return self.looks[-1]
        return self.looks[self.stopped_at_look - 1]

    @property
    def n_at_stop(self) -> int:
        """Units per arm actually consumed before the decision was made."""
        return self.final.n_per_arm


def run_with_peeking(
    experiment: Experiment,
    n_looks: int,
    alpha: float = 0.05,
    min_look_size: int = MIN_LOOK_SIZE,
) -> PeekOutcome:
    """Analyse ``experiment`` repeatedly, stopping at the first significant look.

    Each look applies the ordinary fixed-horizon test at the full nominal ``alpha`` -- no
    correction whatsoever. That is the point: this is what an experimenter does when they
    refresh the dashboard and call it as soon as it goes green.
    """
    schedule = peek_schedule(experiment.spec.n_per_arm, n_looks, min_look_size)
    looks: list[TestResult] = []
    stopped_at: int | None = None

    for i, n in enumerate(schedule, start=1):
        result = fixed_horizon_test(experiment.truncate(n), alpha=alpha)
        looks.append(result)
        if result.significant:
            stopped_at = i
            break  # An experimenter who has called it stops collecting data.

    return PeekOutcome(looks=tuple(looks), stopped_at_look=stopped_at)


def peeking_study(
    spec: ExperimentSpec,
    n_looks: int,
    n_sims: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
    min_look_size: int = MIN_LOOK_SIZE,
) -> SimulationSummary:
    """Replicate ``spec`` many times under a peeking rule with ``n_looks`` analyses.

    With ``spec.effect == 0`` the returned ``reject_rate`` is the realised Type-I error, to
    be read against the nominal ``alpha``. With a non-zero effect it is power, and
    ``estimate_bias`` is the winner's curse.
    """
    rejected: list[bool] = []
    estimates: list[float] = []
    ns: list[int] = []
    for experiment in generate_many(spec, n_sims, seed):
        outcome = run_with_peeking(experiment, n_looks, alpha=alpha, min_look_size=min_look_size)
        rejected.append(outcome.rejected)
        estimates.append(outcome.final.estimate)
        ns.append(outcome.n_at_stop)

    return summarize(rejected, estimates, ns, spec.effect)


def type_i_error_curve(
    spec: ExperimentSpec,
    look_counts: tuple[int, ...] = (1, 2, 3, 5, 10, 20),
    n_sims: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
    min_look_size: int = MIN_LOOK_SIZE,
) -> dict[int, SimulationSummary]:
    """Measured error rate as a function of how many times the experiment is checked.

    ``spec`` should carry ``effect=0.0``; anything else measures power, not Type-I error.
    Every look count reuses the same ``seed``, so the curve varies only in the decision
    rule and not in the underlying data.
    """
    if spec.effect != 0.0:
        raise ValueError(
            f"type_i_error_curve needs a null spec (effect=0.0), got effect={spec.effect}. "
            "Use peeking_study directly to measure power under an alternative."
        )
    return {
        k: peeking_study(
            spec, k, n_sims=n_sims, seed=seed, alpha=alpha, min_look_size=min_look_size
        )
        for k in look_counts
    }
