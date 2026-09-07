"""Group-sequential testing: keep the looks, put the alpha on a budget.

:mod:`ab_testing_kit.peeking` measures what unrestricted monitoring costs -- a nominal 5%
test that actually fires 24.7% of the time at twenty looks. This module fixes it, and the
fix is not "look less often". It is to decide **in advance** how much of the 5% each look is
allowed to consume, and to raise the significance bar at every look so that the total,
across the whole schedule, still comes to 5%.

That framing is the whole idea. An alpha *spending function* ``alpha*(t)`` maps information
fraction ``t = n_k / n_max`` to the cumulative Type-I error the design has permitted itself
to spend by that point, with ``alpha*(1) = alpha``. Given the schedule, the critical values
follow by construction: ``z_k`` is whatever value makes the probability of having crossed
*by look k* -- accounting for the fact that the earlier looks are correlated with this one,
because they share data -- equal exactly ``alpha*(t_k)``.

Two spending functions are implemented, because the contrast is the interesting part:

* **O'Brien-Fleming** spends almost nothing early. The first look of a five-look design is
  tested at roughly ``z = 4.4``, so stopping there requires an overwhelming result. In
  exchange the final look is tested at close to the ordinary 1.96, which is why an OBF
  design costs very little power relative to a fixed-horizon test.
* **Pocock** spends evenly, so every look uses nearly the same critical value. Early
  stopping is much more likely, but the final look is penalised: an experiment that runs to
  its horizon and lands at ``z = 2.1`` is significant under a fixed-horizon test and *not*
  significant under Pocock.

Neither is free, and the cost is measured rather than asserted: :func:`sequential_study`
runs against the same seeded experiments the peeking study uses, so the README's sequential
rows and peeking rows differ only in the decision rule applied to them.

Implementation note
-------------------
The boundaries come from the standard recursive numerical integration (Armitage-McPherson-
Rowe, as used for Lan-DeMets designs), not from a closed form -- none exists past the first
look. Under the null the cumulative score statistic ``B(t)`` behaves as Brownian motion, so
the continuation region after each look carries a *sub-density* that the next look convolves
forward with a Gaussian increment. Crossing probabilities are integrals of that sub-density,
and each critical value is found by root-finding against the spending schedule. See
:func:`alpha_spending_boundary`.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from scipy import optimize, stats

from ab_testing_kit.peeking import MIN_LOOK_SIZE, peek_schedule
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
    "SPENDING_FUNCTIONS",
    "SequentialBoundary",
    "SequentialDesign",
    "SequentialOutcome",
    "alpha_spending_boundary",
    "obrien_fleming_spend",
    "pocock_spend",
    "run_sequential",
    "sequential_design",
    "sequential_error_curve",
    "sequential_study",
    "spending_function",
]

# Quadrature nodes per look. Must be odd (composite Simpson). The continuation region is
# only a couple of units wide on the B scale, while the sub-density at an early look has a
# standard deviation of sqrt(t) -- a sharp spike inside a wide interval -- so the grid has to
# resolve the spike, not merely the interval. 1001 nodes puts the absolute quadrature error
# near 1e-12, comfortably below the smallest alpha increment the solver will chase.
_GRID_POINTS = 1001

# Truncate each look's carried-forward grid at this many standard deviations. The exact
# continuation region is [-b, b]; beyond +/- 8 sqrt(t) the sub-density is below 1e-15 and
# spending grid resolution out there buys nothing. Crossing probabilities themselves are
# always evaluated over the exact region, so this only ever affects propagation.
_GRID_SD = 8.0

# Root-finding bracket for a critical value, on the z scale.
_MAX_Z = 15.0

# An alpha increment below this is smaller than the quadrature's own error, so solving for it
# would be fitting noise. O'Brien-Fleming's first look of a twenty-look design budgets about
# 2e-18, which no numerical scheme can honour. Such looks get an infinite boundary -- they
# simply cannot reject -- and their unspent alpha rolls forward, because the recursion targets
# *cumulative* spend. Conservative by construction: the design spends less than its schedule
# permits, never more.
_MIN_SPEND = 1e-9


def obrien_fleming_spend(t: float, alpha: float) -> float:
    """O'Brien-Fleming-style cumulative alpha spent by information fraction ``t``.

    ``alpha*(t) = 2 (1 - Phi(z_{alpha/2} / sqrt(t)))``, the Lan-DeMets spending function that
    reproduces the shape of O'Brien and Fleming's original boundary while allowing the looks
    to fall wherever the data actually arrives, rather than at pre-committed equal spacing.

    The ``1/sqrt(t)`` inflation of the critical value is what makes it so conservative early:
    at ``t = 0.2`` the argument is ``1.96 / 0.447 = 4.38``, so the design has permitted itself
    roughly one part in 4,000 of its error budget by the first of five looks.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return alpha
    return float(2.0 * stats.norm.sf(float(stats.norm.isf(alpha / 2.0)) / math.sqrt(t)))


def pocock_spend(t: float, alpha: float) -> float:
    """Pocock-style cumulative alpha spent by information fraction ``t``.

    ``alpha*(t) = alpha ln(1 + (e - 1) t)``, which spends nearly evenly across the schedule
    and so produces a near-constant critical value. Included as the foil to O'Brien-Fleming
    rather than as a recommendation: it buys realistic early stopping at the price of a final
    look that is meaningfully harder to pass than 1.96.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return alpha
    return float(alpha * math.log1p((math.e - 1.0) * t))


SPENDING_FUNCTIONS: dict[str, Callable[[float, float], float]] = {
    "obrien-fleming": obrien_fleming_spend,
    "pocock": pocock_spend,
}


def spending_function(name: str) -> Callable[[float, float], float]:
    """Look up a spending function by name."""
    try:
        return SPENDING_FUNCTIONS[name]
    except KeyError:
        known = ", ".join(sorted(SPENDING_FUNCTIONS))
        raise ValueError(f"unknown spending function {name!r}; known: {known}") from None


@dataclass(frozen=True)
class SequentialBoundary:
    """Critical values for one analysis schedule, and the alpha they consume.

    Attributes:
        information: Information fraction at each look, strictly increasing, ending at 1.0
            for a schedule that runs to its planned horizon.
        z: Critical value at each look. ``inf`` marks a look whose alpha budget was too small
            to resolve numerically, which therefore cannot reject.
        budgeted: Cumulative alpha the spending function permits by each look.
        achieved: Cumulative crossing probability the computed boundary actually delivers.
            Equals ``budgeted`` to solver tolerance except at ``inf`` looks, where it lags
            because the unspent budget has rolled forward.
    """

    information: tuple[float, ...]
    z: tuple[float, ...]
    budgeted: tuple[float, ...]
    achieved: tuple[float, ...]
    alpha: float
    spending: str

    def __len__(self) -> int:
        return len(self.z)

    @property
    def total_alpha(self) -> float:
        """Type-I error the whole boundary spends. Should land on ``alpha``."""
        return self.achieved[-1]

    @property
    def increments(self) -> tuple[float, ...]:
        """Alpha actually spent *at* each look, rather than cumulatively."""
        previous = (0.0, *self.achieved[:-1])
        return tuple(now - before for now, before in zip(self.achieved, previous, strict=True))


def _simpson_grid(half_width: float, n_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Composite-Simpson nodes and weights on ``[-half_width, half_width]``."""
    nodes = np.linspace(-half_width, half_width, n_points)
    weights = np.ones(n_points)
    weights[1:-1:2] = 4.0
    weights[2:-1:2] = 2.0
    return nodes, weights * (2.0 * half_width / (n_points - 1)) / 3.0


def _survival(
    nodes: np.ndarray, density: np.ndarray, weights: np.ndarray, b: float, sqrt_dt: float
) -> float:
    """Probability of surviving a look at ``+/- b``, given the sub-density before it.

    The Brownian increment is Gaussian and independent of the past, so the double integral
    over the new continuation region collapses to a single integral over the *previous* grid
    with an analytic inner factor. That is what makes root-finding cheap: each trial boundary
    costs one O(n) pass instead of a fresh convolution.
    """
    upper = stats.norm.cdf((b - nodes) / sqrt_dt)
    lower = stats.norm.cdf((-b - nodes) / sqrt_dt)
    return float(weights @ (density * (upper - lower)))


def _crossed_by(
    trial_z: float,
    nodes: np.ndarray,
    density: np.ndarray,
    weights: np.ndarray,
    sqrt_t: float,
    sqrt_dt: float,
    target: float = 0.0,
) -> float:
    """Cumulative crossing probability at critical value ``trial_z``, minus ``target``.

    Decreasing in ``trial_z``, so root-finding it against a spending target is what produces
    each critical value. Called with the default ``target`` it simply reports the probability,
    which is what gets recorded as the boundary's ``achieved`` alpha.
    """
    return 1.0 - _survival(nodes, density, weights, trial_z * sqrt_t, sqrt_dt) - target


def _propagate(
    nodes: np.ndarray,
    density: np.ndarray,
    weights: np.ndarray,
    new_nodes: np.ndarray,
    sqrt_dt: float,
) -> np.ndarray:
    """Convolve the surviving sub-density forward onto ``new_nodes``."""
    kernel = stats.norm.pdf((new_nodes[:, None] - nodes[None, :]) / sqrt_dt) / sqrt_dt
    return kernel @ (weights * density)


def alpha_spending_boundary(
    information: tuple[float, ...],
    alpha: float = 0.05,
    spending: str = "obrien-fleming",
    grid_points: int = _GRID_POINTS,
) -> SequentialBoundary:
    """Critical values that spend exactly ``alpha`` across the whole schedule.

    Args:
        information: Strictly increasing information fractions in ``(0, 1]``, one per look.
            A final fraction below 1.0 is allowed, and simply leaves alpha unspent.
        alpha: Total two-sided Type-I error for the schedule.
        spending: Key into :data:`SPENDING_FUNCTIONS`.
        grid_points: Odd number of quadrature nodes per look.

    Returns:
        The boundary, carrying the alpha it budgets and the alpha it achieves side by side --
        the second is what :func:`sequential_study` then checks against simulation.

    The recursion carries the sub-density of the cumulative score ``B(t)`` restricted to "has
    not crossed yet". It starts as a point mass at the origin at ``t = 0``, which is why the
    loop needs no special case for the first look: a one-node grid holding weight 1 *is* that
    point mass, and the survival integral through it collapses to the closed-form
    ``Phi(b/sqrt(t_1)) - Phi(-b/sqrt(t_1))``.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if grid_points < 5 or grid_points % 2 == 0:
        raise ValueError(f"grid_points must be an odd integer >= 5, got {grid_points}")
    fractions = tuple(float(t) for t in information)
    if not fractions:
        raise ValueError("information must contain at least one look")
    if not all(0.0 < t <= 1.0 for t in fractions):
        raise ValueError(f"information fractions must be in (0, 1], got {fractions}")
    if any(later <= earlier for earlier, later in pairwise(fractions)):
        raise ValueError(f"information fractions must be strictly increasing, got {fractions}")

    spend = spending_function(spending)
    nodes, density, weights = np.zeros(1), np.ones(1), np.ones(1)
    t_previous = 0.0
    achieved_so_far = 0.0
    z_values: list[float] = []
    budgeted: list[float] = []
    achieved: list[float] = []

    for t in fractions:
        sqrt_dt = math.sqrt(t - t_previous)
        sqrt_t = math.sqrt(t)
        target = spend(t, alpha)

        if target - achieved_so_far <= _MIN_SPEND:
            # Budget below the quadrature's own noise floor. Cannot reject here; the unspent
            # alpha carries forward, because the next look's target is cumulative.
            z = math.inf
            half_width = _GRID_SD * sqrt_t
        else:
            state = (nodes, density, weights, sqrt_t, sqrt_dt)
            z = float(optimize.brentq(_crossed_by, 0.0, _MAX_Z, args=(*state, target), xtol=1e-13))
            achieved_so_far = _crossed_by(z, *state)
            half_width = min(z * sqrt_t, _GRID_SD * sqrt_t)

        z_values.append(z)
        budgeted.append(target)
        achieved.append(achieved_so_far)

        new_nodes, new_weights = _simpson_grid(half_width, grid_points)
        density = _propagate(nodes, density, weights, new_nodes, sqrt_dt)
        nodes, weights = new_nodes, new_weights
        t_previous = t

    return SequentialBoundary(
        information=fractions,
        z=tuple(z_values),
        budgeted=tuple(budgeted),
        achieved=tuple(achieved),
        alpha=alpha,
        spending=spending,
    )


@dataclass(frozen=True)
class SequentialDesign:
    """A boundary bound to the sample sizes it will be evaluated at."""

    schedule: tuple[int, ...]
    """Units per arm at each look, ending at the planned horizon."""

    boundary: SequentialBoundary

    def __len__(self) -> int:
        return len(self.schedule)


def sequential_design(
    n_per_arm: int,
    n_looks: int,
    alpha: float = 0.05,
    spending: str = "obrien-fleming",
    min_look_size: int = MIN_LOOK_SIZE,
) -> SequentialDesign:
    """Build the analysis schedule and its boundary for an experiment of ``n_per_arm``.

    Deliberately reuses :func:`~ab_testing_kit.peeking.peek_schedule`, so a sequential study
    and a peeking study at the same ``n_looks`` analyse the data at *exactly* the same points.
    The two then differ in one thing only: the critical value applied.
    """
    schedule = peek_schedule(n_per_arm, n_looks, min_look_size)
    information = tuple(n / n_per_arm for n in schedule)
    return SequentialDesign(
        schedule=schedule,
        boundary=alpha_spending_boundary(information, alpha=alpha, spending=spending),
    )


@dataclass(frozen=True)
class SequentialOutcome:
    """What happened to one experiment run against a spending boundary."""

    looks: tuple[TestResult, ...]
    design: SequentialDesign
    stopped_at_look: int | None
    """1-indexed look at which the boundary was crossed; ``None`` if it never was."""

    @property
    def rejected(self) -> bool:
        """The sequential verdict. Not the same as ``final.significant`` -- see below."""
        return self.stopped_at_look is not None

    @property
    def final(self) -> TestResult:
        """The result an experimenter would report: at the stop, or at the full sample.

        ``final.significant`` is the *naive fixed-horizon* verdict at that look, kept
        deliberately so the two rules can be compared on identical data. It is not the
        sequential decision; read :attr:`rejected` for that. Under O'Brien-Fleming the two
        routinely disagree at early looks, which is the entire point of the module.
        """
        if self.stopped_at_look is None:
            return self.looks[-1]
        return self.looks[self.stopped_at_look - 1]

    @property
    def n_at_stop(self) -> int:
        """Units per arm consumed before the decision was made."""
        return self.final.n_per_arm

    @property
    def boundary_at_stop(self) -> float:
        """Critical value the reported look was judged against."""
        index = len(self.looks) - 1 if self.stopped_at_look is None else self.stopped_at_look - 1
        return self.design.boundary.z[index]


def run_sequential(
    experiment: Experiment, design: SequentialDesign, alpha: float = 0.05
) -> SequentialOutcome:
    """Analyse ``experiment`` at each look, stopping the first time the boundary is crossed.

    ``alpha`` is passed through to :func:`~ab_testing_kit.simulation.fixed_horizon_test` only
    so each :class:`~ab_testing_kit.simulation.TestResult` carries a comparable naive verdict
    and confidence interval. The sequential decision uses the test *statistic* against
    ``design.boundary.z`` and ignores that verdict entirely.

    A continuous metric produces a Welch t-statistic, which is compared against a normal
    boundary. The boundary theory is asymptotic in the first place, and by the smallest look
    this toolkit permits (30 units per arm) the two quantiles differ in the third decimal; the
    simulated Type-I error in the README is what confirms the approximation does no damage.
    """
    if len(design.schedule) != len(design.boundary):
        raise ValueError("design schedule and boundary must have the same number of looks")

    looks: list[TestResult] = []
    stopped_at: int | None = None

    for i, (n, z) in enumerate(zip(design.schedule, design.boundary.z, strict=True), start=1):
        result = fixed_horizon_test(experiment.truncate(n), alpha=alpha)
        looks.append(result)
        if abs(result.statistic) >= z:
            stopped_at = i
            break  # An experimenter who has called it stops collecting data.

    return SequentialOutcome(looks=tuple(looks), design=design, stopped_at_look=stopped_at)


def sequential_study(
    spec: ExperimentSpec,
    n_looks: int,
    n_sims: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
    spending: str = "obrien-fleming",
    min_look_size: int = MIN_LOOK_SIZE,
) -> SimulationSummary:
    """Replicate ``spec`` under a spending boundary with ``n_looks`` analyses.

    The direct counterpart of :func:`~ab_testing_kit.peeking.peeking_study`: same spec, same
    seed, same analysis points, corrected critical values. With ``spec.effect == 0`` the
    returned ``reject_rate`` is the realised Type-I error, and it should sit on ``alpha``
    however many looks are taken -- the claim this whole module exists to make checkable.

    ``mean_n_per_arm`` is the other half of the story. Alpha spending is not free, and what it
    costs is measured here: how much data the design consumes on average before it decides.
    """
    design = sequential_design(
        spec.n_per_arm, n_looks, alpha=alpha, spending=spending, min_look_size=min_look_size
    )
    rejected: list[bool] = []
    estimates: list[float] = []
    ns: list[int] = []
    for experiment in generate_many(spec, n_sims, seed):
        outcome = run_sequential(experiment, design, alpha=alpha)
        rejected.append(outcome.rejected)
        estimates.append(outcome.final.estimate)
        ns.append(outcome.n_at_stop)

    return summarize(rejected, estimates, ns, spec.effect)


def sequential_error_curve(
    spec: ExperimentSpec,
    look_counts: tuple[int, ...] = (1, 2, 3, 5, 10, 20),
    n_sims: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
    spending: str = "obrien-fleming",
    min_look_size: int = MIN_LOOK_SIZE,
) -> dict[int, SimulationSummary]:
    """Measured Type-I error against look count, under alpha spending.

    Deliberately the same signature and the same default ``look_counts`` as
    :func:`~ab_testing_kit.peeking.type_i_error_curve`, so the two curves can be printed
    beside each other. That one climbs to 24.7%; this one should stay flat.
    """
    if spec.effect != 0.0:
        raise ValueError(
            f"sequential_error_curve needs a null spec (effect=0.0), got effect={spec.effect}. "
            "Use sequential_study directly to measure power under an alternative."
        )
    return {
        k: sequential_study(
            spec,
            k,
            n_sims=n_sims,
            seed=seed,
            alpha=alpha,
            spending=spending,
            min_look_size=min_look_size,
        )
        for k in look_counts
    }
