"""Tests for the peeking demonstration.

The headline assertion is ``test_error_rate_inflates_with_more_looks``: the same null data,
analysed once, holds 5%; analysed repeatedly with no correction, does not.
"""

from __future__ import annotations

import pytest

from ab_testing_kit.design import validate_power
from ab_testing_kit.peeking import (
    peek_schedule,
    peeking_study,
    run_with_peeking,
    type_i_error_curve,
)
from ab_testing_kit.simulation import ExperimentSpec, generate_many


class TestPeekSchedule:
    def test_single_look_is_the_full_sample(self) -> None:
        assert peek_schedule(1000, 1) == (1000,)

    def test_evenly_spaced_and_ends_at_the_horizon(self) -> None:
        assert peek_schedule(1000, 4) == (250, 500, 750, 1000)

    def test_is_strictly_increasing(self) -> None:
        schedule = peek_schedule(3000, 7)
        assert list(schedule) == sorted(set(schedule))

    def test_drops_looks_below_the_minimum(self) -> None:
        """20 looks at 200 units would start at 10 per arm, where the z-test is junk."""
        schedule = peek_schedule(200, 20, min_look_size=30)
        assert min(schedule) >= 30
        assert schedule[-1] == 200

    def test_always_reaches_the_horizon_even_when_all_looks_are_dropped(self) -> None:
        assert peek_schedule(40, 5, min_look_size=100) == (40,)

    def test_rejects_invalid_inputs(self) -> None:
        with pytest.raises(ValueError, match="n_per_arm"):
            peek_schedule(1, 3)
        with pytest.raises(ValueError, match="n_looks"):
            peek_schedule(1000, 0)


class TestRunWithPeeking:
    def test_single_look_equals_the_fixed_horizon_test(self) -> None:
        spec = ExperimentSpec(n_per_arm=2000, metric="binary", baseline=0.1, effect=0.02)
        exp = next(generate_many(spec, 1, seed=5))
        outcome = run_with_peeking(exp, n_looks=1)
        assert len(outcome.looks) == 1
        assert outcome.final.n_per_arm == 2000

    def test_stops_collecting_once_it_calls_a_winner(self) -> None:
        """A schedule of 10 looks that stops at look 3 must record exactly 3 looks."""
        spec = ExperimentSpec(n_per_arm=4000, metric="continuous", baseline=0.0, effect=0.3, sd=1.0)
        exp = next(generate_many(spec, 1, seed=6))
        outcome = run_with_peeking(exp, n_looks=10)
        assert outcome.rejected
        assert len(outcome.looks) == outcome.stopped_at_look
        assert outcome.n_at_stop < 4000

    def test_final_is_the_stopping_look(self) -> None:
        spec = ExperimentSpec(n_per_arm=4000, metric="continuous", baseline=0.0, effect=0.3, sd=1.0)
        outcome = run_with_peeking(next(generate_many(spec, 1, seed=6)), n_looks=10)
        assert outcome.final is outcome.looks[-1]
        assert outcome.final.significant

    def test_never_rejecting_reports_the_full_sample(self) -> None:
        spec = ExperimentSpec(n_per_arm=300, metric="continuous", baseline=0.0, sd=1.0)
        for exp in generate_many(spec, 40, seed=7):
            outcome = run_with_peeking(exp, n_looks=5)
            if not outcome.rejected:
                assert outcome.stopped_at_look is None
                assert outcome.n_at_stop == 300
                assert not outcome.final.significant
                return
        pytest.fail("expected at least one non-rejecting replication under the null")


class TestTypeIErrorInflation:
    def test_one_look_holds_nominal_alpha(self) -> None:
        spec = ExperimentSpec(n_per_arm=2000, metric="binary", baseline=0.10)
        summary = peeking_study(spec, n_looks=1, n_sims=3_000, seed=201)
        assert summary.reject_rate == pytest.approx(0.05, abs=0.012)

    def test_error_rate_inflates_with_more_looks(self) -> None:
        """The headline result. Same null, same seed, only the decision rule changes."""
        spec = ExperimentSpec(n_per_arm=4000, metric="continuous", baseline=0.0, sd=1.0)
        curve = type_i_error_curve(spec, look_counts=(1, 5, 20), n_sims=2_000, seed=202)
        assert curve[1].reject_rate == pytest.approx(0.05, abs=0.015)
        assert curve[5].reject_rate > curve[1].reject_rate
        assert curve[20].reject_rate > curve[5].reject_rate
        # Twenty unadjusted looks should roughly triple the nominal rate.
        assert curve[20].reject_rate > 0.12

    def test_inflation_is_outside_monte_carlo_noise(self) -> None:
        """Guards against reading a sampling wobble as a real effect."""
        spec = ExperimentSpec(n_per_arm=4000, metric="continuous", baseline=0.0, sd=1.0)
        curve = type_i_error_curve(spec, look_counts=(1, 10), n_sims=2_500, seed=203)
        assert curve[10].reject_rate_ci[0] > curve[1].reject_rate_ci[1]

    def test_curve_rejects_a_non_null_spec(self) -> None:
        spec = ExperimentSpec(n_per_arm=1000, metric="continuous", baseline=0.0, effect=0.1)
        with pytest.raises(ValueError, match="null spec"):
            type_i_error_curve(spec, look_counts=(1, 2), n_sims=10)


class TestWinnersCurse:
    def test_peeking_inflates_the_reported_effect(self) -> None:
        """The quieter cost: stopping early biases the estimate you ship with."""
        spec = ExperimentSpec(
            n_per_arm=6000, metric="continuous", baseline=0.0, effect=0.05, sd=1.0
        )
        peeked = peeking_study(spec, n_looks=20, n_sims=1_200, seed=204)
        fixed = validate_power(spec, n_sims=1_200, seed=204)
        assert peeked.estimate_bias > fixed.simulated.estimate_bias
        assert peeked.estimate_bias > 0.0

    def test_peeking_stops_earlier_on_average(self) -> None:
        spec = ExperimentSpec(
            n_per_arm=6000, metric="continuous", baseline=0.0, effect=0.05, sd=1.0
        )
        peeked = peeking_study(spec, n_looks=20, n_sims=500, seed=205)
        assert peeked.mean_n_per_arm < 6000
