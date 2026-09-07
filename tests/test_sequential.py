"""Tests for group-sequential alpha spending.

Two headline assertions here, and they are different in kind.

``test_crossing_probability_matches_a_brownian_simulation`` checks the *numerics*: the
recursive integration that produces the boundary is validated against a direct Monte Carlo
of the Brownian motion it models. That is a check on the solver, and it deliberately does
not go through the A/B substrate at all -- if the quadrature were wrong, this fails first
and the failure points at the right place.

``test_error_rate_stays_flat_with_more_looks`` checks the *claim*: the same null
experiments that drive a naive rule to 24.7% leave a spending boundary sitting on 5%.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest

from ab_testing_kit.peeking import peek_schedule, peeking_study
from ab_testing_kit.sequential import (
    SequentialDesign,
    alpha_spending_boundary,
    obrien_fleming_spend,
    pocock_spend,
    run_sequential,
    sequential_design,
    sequential_error_curve,
    sequential_study,
    spending_function,
)
from ab_testing_kit.simulation import ExperimentSpec, generate_many


def _equal_looks(k: int) -> tuple[float, ...]:
    return tuple((i + 1) / k for i in range(k))


class TestSpendingFunctions:
    def test_nothing_spent_before_the_experiment_starts(self) -> None:
        assert obrien_fleming_spend(0.0, 0.05) == 0.0
        assert pocock_spend(0.0, 0.05) == 0.0

    def test_everything_spent_by_the_horizon(self) -> None:
        assert obrien_fleming_spend(1.0, 0.05) == 0.05
        assert pocock_spend(1.0, 0.05) == 0.05

    @pytest.mark.parametrize("spend", [obrien_fleming_spend, pocock_spend])
    def test_is_strictly_increasing(self, spend: Callable[[float, float], float]) -> None:
        values = [spend(t, 0.05) for t in (0.1, 0.25, 0.5, 0.75, 0.9)]
        assert values == sorted(values)
        assert len(set(values)) == len(values)

    def test_obrien_fleming_is_the_more_miserly_early(self) -> None:
        """The whole character of the two functions, in one assertion."""
        assert obrien_fleming_spend(0.2, 0.05) < pocock_spend(0.2, 0.05) / 1000.0

    def test_pocock_spends_roughly_evenly(self) -> None:
        """Half the information, appreciably more than half the alpha -- but the same order."""
        assert 0.5 < pocock_spend(0.5, 0.05) / 0.05 < 0.85

    @pytest.mark.parametrize("spend", [obrien_fleming_spend, pocock_spend])
    def test_rejects_invalid_alpha(self, spend: Callable[[float, float], float]) -> None:
        with pytest.raises(ValueError, match="alpha"):
            spend(0.5, 1.0)

    def test_lookup_by_name(self) -> None:
        assert spending_function("obrien-fleming") is obrien_fleming_spend
        assert spending_function("pocock") is pocock_spend

    def test_unknown_name_lists_what_is_available(self) -> None:
        with pytest.raises(ValueError, match="unknown spending function"):
            spending_function("haybittle-peto")


class TestBoundary:
    def test_one_look_is_exactly_the_fixed_horizon_critical_value(self) -> None:
        """A one-look 'sequential' design is a fixed-horizon test. It had better say so."""
        boundary = alpha_spending_boundary((1.0,), alpha=0.05)
        assert boundary.z[0] == pytest.approx(1.959964, abs=1e-6)

    def test_crossing_probability_matches_a_brownian_simulation(self) -> None:
        """Validates the quadrature against the process it claims to integrate.

        Under the null the sequence of z-statistics is standardised Brownian motion observed
        at the information times. Simulating that motion directly and counting boundary
        crossings is a completely independent route to the same number, sharing no code with
        the recursion.
        """
        rng = np.random.default_rng(11)
        n_paths = 200_000
        for n_looks in (2, 5, 20):
            information = np.array(_equal_looks(n_looks))
            increments = np.diff(np.concatenate([[0.0], information]))
            brownian = np.cumsum(
                rng.standard_normal((n_paths, n_looks)) * np.sqrt(increments), axis=1
            )
            z_paths = brownian / np.sqrt(information)
            for spending in ("obrien-fleming", "pocock"):
                boundary = alpha_spending_boundary(tuple(information), 0.05, spending)
                crossed = float((np.abs(z_paths) >= np.array(boundary.z)).any(axis=1).mean())
                # 200k paths puts the Monte Carlo standard error near 0.0005.
                assert crossed == pytest.approx(0.05, abs=0.002)

    def test_spends_the_whole_budget_and_no_more(self) -> None:
        for n_looks in (1, 2, 5, 20):
            boundary = alpha_spending_boundary(_equal_looks(n_looks), alpha=0.05)
            assert boundary.total_alpha == pytest.approx(0.05, abs=1e-9)

    def test_achieved_tracks_what_was_budgeted(self) -> None:
        boundary = alpha_spending_boundary(_equal_looks(5), alpha=0.05)
        for achieved, budgeted in zip(boundary.achieved, boundary.budgeted, strict=True):
            assert achieved == pytest.approx(budgeted, abs=1e-9)

    def test_obrien_fleming_boundary_relaxes_towards_the_horizon(self) -> None:
        boundary = alpha_spending_boundary(_equal_looks(5), alpha=0.05)
        assert list(boundary.z) == sorted(boundary.z, reverse=True)
        assert boundary.z[0] > 4.0
        # The final look is only slightly stricter than an uncorrected test. That is the
        # entire selling point of O'Brien-Fleming: the correction is nearly free at the end.
        assert 1.96 < boundary.z[-1] < 2.15

    def test_pocock_boundary_is_nearly_flat(self) -> None:
        boundary = alpha_spending_boundary(_equal_looks(5), alpha=0.05, spending="pocock")
        assert max(boundary.z) - min(boundary.z) < 0.1
        # ...and the price of that flatness is a final look well above 1.96.
        assert boundary.z[-1] > 2.3

    def test_increments_sum_to_the_total(self) -> None:
        boundary = alpha_spending_boundary(_equal_looks(5), alpha=0.05)
        assert sum(boundary.increments) == pytest.approx(boundary.total_alpha)
        assert all(step >= 0.0 for step in boundary.increments)
        assert len(boundary) == 5

    def test_unresolvably_small_budget_gives_an_unusable_boundary(self) -> None:
        """Twenty O'Brien-Fleming looks budget ~2e-18 for the first. That is not solvable.

        The look is marked ``inf`` rather than fudged, so it cannot reject; the alpha it did
        not spend rolls into the next look and the total still lands on 0.05.
        """
        boundary = alpha_spending_boundary(_equal_looks(20), alpha=0.05)
        assert math.isinf(boundary.z[0])
        assert boundary.achieved[0] == 0.0
        assert all(math.isfinite(z) for z in boundary.z[3:])
        assert boundary.total_alpha == pytest.approx(0.05, abs=1e-9)

    def test_stopping_short_of_the_horizon_leaves_alpha_unspent(self) -> None:
        boundary = alpha_spending_boundary((0.3, 0.6), alpha=0.05)
        assert boundary.total_alpha < 0.05

    def test_is_insensitive_to_grid_refinement(self) -> None:
        """If the answer moved when the quadrature got finer, it would not be an answer."""
        coarse = alpha_spending_boundary(_equal_looks(5), grid_points=401)
        fine = alpha_spending_boundary(_equal_looks(5), grid_points=1601)
        for a, b in zip(coarse.z, fine.z, strict=True):
            assert a == pytest.approx(b, abs=1e-5)

    def test_rejects_invalid_alpha(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            alpha_spending_boundary((1.0,), alpha=0.0)

    @pytest.mark.parametrize("grid_points", [4, 1000])
    def test_rejects_unusable_grids(self, grid_points: int) -> None:
        with pytest.raises(ValueError, match="grid_points"):
            alpha_spending_boundary((1.0,), grid_points=grid_points)

    def test_rejects_an_empty_schedule(self) -> None:
        with pytest.raises(ValueError, match="at least one look"):
            alpha_spending_boundary(())

    @pytest.mark.parametrize("information", [(0.0, 1.0), (0.5, 1.5), (-0.2, 1.0)])
    def test_rejects_impossible_information_fractions(self, information: tuple[float, ...]) -> None:
        with pytest.raises(ValueError, match=r"in \(0, 1\]"):
            alpha_spending_boundary(information)

    @pytest.mark.parametrize("information", [(0.5, 0.5, 1.0), (0.8, 0.4, 1.0)])
    def test_rejects_a_schedule_that_does_not_move_forward(
        self, information: tuple[float, ...]
    ) -> None:
        with pytest.raises(ValueError, match="strictly increasing"):
            alpha_spending_boundary(information)


class TestSequentialDesign:
    def test_analyses_at_exactly_the_same_points_as_the_peeking_study(self) -> None:
        """Non-negotiable: the two rules must differ in the rule, not in the data."""
        design = sequential_design(3841, 20)
        assert design.schedule == peek_schedule(3841, 20)
        assert len(design) == len(design.boundary)

    def test_information_fractions_end_at_the_horizon(self) -> None:
        design = sequential_design(2000, 5)
        assert design.boundary.information[-1] == 1.0
        assert design.boundary.information[0] == pytest.approx(0.2)


class TestRunSequential:
    def test_one_look_reproduces_the_fixed_horizon_decision(self) -> None:
        spec = ExperimentSpec(n_per_arm=2000, metric="binary", baseline=0.1, effect=0.02)
        design = sequential_design(2000, 1)
        for experiment in generate_many(spec, 25, seed=41):
            outcome = run_sequential(experiment, design)
            assert outcome.rejected == outcome.final.significant

    def test_stops_collecting_once_the_boundary_is_crossed(self) -> None:
        spec = ExperimentSpec(n_per_arm=4000, metric="continuous", baseline=0.0, effect=0.4, sd=1.0)
        design = sequential_design(4000, 10)
        outcome = run_sequential(next(generate_many(spec, 1, seed=42)), design)
        assert outcome.rejected
        assert len(outcome.looks) == outcome.stopped_at_look
        assert outcome.n_at_stop < 4000
        assert outcome.n_at_stop == design.schedule[outcome.stopped_at_look - 1]
        assert abs(outcome.final.statistic) >= outcome.boundary_at_stop

    def test_never_crossing_reports_the_full_sample(self) -> None:
        spec = ExperimentSpec(n_per_arm=1000, metric="continuous", baseline=0.0, sd=1.0)
        design = sequential_design(1000, 5)
        for experiment in generate_many(spec, 40, seed=43):
            outcome = run_sequential(experiment, design)
            if not outcome.rejected:
                assert outcome.stopped_at_look is None
                assert outcome.n_at_stop == 1000
                assert outcome.boundary_at_stop == design.boundary.z[-1]
                return
        pytest.fail("expected at least one non-rejecting replication under the null")

    def test_a_naively_significant_look_need_not_cross_the_boundary(self) -> None:
        """The documented footgun, pinned down.

        ``final.significant`` is the uncorrected per-look verdict and ``rejected`` is the
        sequential one. Under the null with many looks they disagree constantly -- that
        disagreement *is* the correction doing its job.
        """
        spec = ExperimentSpec(n_per_arm=4000, metric="continuous", baseline=0.0, sd=1.0)
        design = sequential_design(4000, 20)
        for experiment in generate_many(spec, 60, seed=44):
            outcome = run_sequential(experiment, design)
            if any(look.significant for look in outcome.looks) and not outcome.rejected:
                return
        pytest.fail("expected a replication the naive rule would have called and this one did not")

    def test_rejects_a_design_whose_parts_disagree(self) -> None:
        spec = ExperimentSpec(n_per_arm=200, metric="continuous", baseline=0.0, sd=1.0)
        broken = SequentialDesign(schedule=(100, 200), boundary=alpha_spending_boundary((1.0,)))
        with pytest.raises(ValueError, match="same number of looks"):
            run_sequential(next(generate_many(spec, 1, seed=45)), broken)


class TestTypeIControl:
    def test_error_rate_stays_flat_with_more_looks(self) -> None:
        """The headline. Peeking's curve climbs; this one does not move.

        Tolerance is set by Monte Carlo error at this replication count, not by taste: 500
        replications put one standard error near 0.0097, so 0.025 is about 2.5 SE. The
        headline numbers come from the CLI at 20,000 replications instead.
        """
        spec = ExperimentSpec(n_per_arm=1500, metric="continuous", baseline=0.0, sd=1.0)
        curve = sequential_error_curve(spec, look_counts=(1, 5, 20), n_sims=500, seed=301)
        for summary in curve.values():
            assert summary.reject_rate == pytest.approx(0.05, abs=0.025)

    def test_beats_the_naive_rule_on_identical_data(self) -> None:
        """Same spec, same seed, same analysis points. Only the critical values differ."""
        spec = ExperimentSpec(n_per_arm=1500, metric="continuous", baseline=0.0, sd=1.0)
        naive = peeking_study(spec, n_looks=20, n_sims=400, seed=302)
        spent = sequential_study(spec, n_looks=20, n_sims=400, seed=302)
        assert naive.reject_rate > 0.15
        assert spent.reject_rate < 0.10
        assert spent.reject_rate_ci[1] < naive.reject_rate_ci[0]

    def test_pocock_also_holds_its_rate(self) -> None:
        """A different spending shape, the same guarantee. The guarantee is not OBF-specific."""
        spec = ExperimentSpec(n_per_arm=1500, metric="continuous", baseline=0.0, sd=1.0)
        summary = sequential_study(spec, n_looks=10, n_sims=500, seed=303, spending="pocock")
        assert summary.reject_rate == pytest.approx(0.05, abs=0.025)


class TestWhatItCosts:
    def test_reaches_a_decision_on_less_data_than_the_horizon(self) -> None:
        spec = ExperimentSpec(
            n_per_arm=2500, metric="continuous", baseline=0.0, effect=0.12, sd=1.0
        )
        summary = sequential_study(spec, n_looks=10, n_sims=400, seed=304)
        assert summary.mean_n_per_arm < 2500

    def test_pocock_stops_sooner_but_pays_for_it_in_power(self) -> None:
        """The tradeoff between the two spending shapes, measured rather than described."""
        spec = ExperimentSpec(
            n_per_arm=2500, metric="continuous", baseline=0.0, effect=0.09, sd=1.0
        )
        obf = sequential_study(spec, n_looks=10, n_sims=400, seed=305)
        pocock = sequential_study(spec, n_looks=10, n_sims=400, seed=305, spending="pocock")
        assert pocock.mean_n_per_arm < obf.mean_n_per_arm
        assert pocock.reject_rate < obf.reject_rate

    def test_the_winners_curse_is_smaller_than_under_naive_peeking(self) -> None:
        """Stopping early still inflates the estimate -- just far less, because it stops later."""
        spec = ExperimentSpec(
            n_per_arm=3000, metric="continuous", baseline=0.0, effect=0.07, sd=1.0
        )
        naive = peeking_study(spec, n_looks=20, n_sims=350, seed=306)
        spent = sequential_study(spec, n_looks=20, n_sims=350, seed=306)
        assert 0.0 < spent.estimate_bias < naive.estimate_bias

    def test_magnitude_inflation_under_the_null_is_smaller_too(self) -> None:
        """Read magnitude, not sign, under a two-sided null. See ``mean_abs_estimate``."""
        spec = ExperimentSpec(n_per_arm=1500, metric="continuous", baseline=0.0, sd=1.0)
        naive = peeking_study(spec, n_looks=20, n_sims=350, seed=307)
        spent = sequential_study(spec, n_looks=20, n_sims=350, seed=307)
        assert spent.mean_abs_estimate < naive.mean_abs_estimate


class TestErrorCurve:
    def test_rejects_a_non_null_spec(self) -> None:
        spec = ExperimentSpec(n_per_arm=1000, metric="continuous", baseline=0.0, effect=0.1)
        with pytest.raises(ValueError, match="null spec"):
            sequential_error_curve(spec, look_counts=(1, 2), n_sims=10)
