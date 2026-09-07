"""Tests for CUPED variance reduction.

The headline is ``test_measured_reduction_matches_the_theory``: the method promises to remove
a fraction ``rho^2`` of the variance, and that is checked against the correlation actually
realised in the data rather than the one requested from the generator. For binary metrics
those two numbers are very different, which is its own test.

The second thing worth checking is that the adjustment is *free* -- that buying precision has
not quietly cost unbiasedness or size. ``TestItStaysHonest`` covers both.
"""

from __future__ import annotations

import numpy as np
import pytest

from ab_testing_kit.cuped import (
    CupedComparison,
    cuped_adjust,
    cuped_at_each_look,
    cuped_study,
    cuped_test,
    cuped_theta,
    variance_reduction_curve,
)
from ab_testing_kit.simulation import (
    Experiment,
    ExperimentSpec,
    SimulationSummary,
    fixed_horizon_test,
    generate_many,
    summarize,
)


def _spec(rho: float, effect: float = 0.05, n: int = 2000) -> ExperimentSpec:
    return ExperimentSpec(
        n_per_arm=n,
        metric="continuous",
        baseline=0.0,
        effect=effect,
        sd=1.0,
        covariate_corr=rho,
    )


def _degenerate(values: np.ndarray, pre: np.ndarray, metric: str = "continuous") -> Experiment:
    """An experiment built by hand, for the edge cases the generator will not produce."""
    return Experiment(
        spec=ExperimentSpec(
            n_per_arm=values.size,
            metric=metric,  # type: ignore[arg-type]
            baseline=0.1 if metric == "binary" else 0.0,
            sd=1.0,
        ),
        control=values.copy(),
        treatment=values.copy(),
        control_pre=pre.copy(),
        treatment_pre=pre.copy(),
    )


class TestTheta:
    def test_recovers_the_generating_slope(self) -> None:
        """With unit variances, the optimal theta is just the correlation."""
        experiment = next(generate_many(_spec(0.6), 1, seed=10))
        assert cuped_theta(experiment) == pytest.approx(0.6, abs=0.05)

    def test_is_zero_when_the_covariate_carries_no_signal(self) -> None:
        experiment = next(generate_many(_spec(0.0), 1, seed=11))
        assert cuped_theta(experiment) == pytest.approx(0.0, abs=0.05)

    def test_is_not_inflated_by_a_large_treatment_effect(self) -> None:
        """Centring each arm first is what makes this true; pooling raw would not be.

        Two experiments identical but for the size of the true effect must produce the same
        theta, because theta describes the covariate relationship and nothing else.
        """
        small = cuped_theta(next(generate_many(_spec(0.6, effect=0.0), 1, seed=12)))
        large = cuped_theta(next(generate_many(_spec(0.6, effect=5.0), 1, seed=12)))
        assert small == pytest.approx(large, abs=1e-12)

    def test_a_constant_covariate_disables_the_adjustment(self) -> None:
        """Reachable: an all-zero binary covariate on a small, low-rate experiment."""
        experiment = _degenerate(np.zeros(10), np.zeros(10), metric="binary")
        assert cuped_theta(experiment) == 0.0


class TestAdjust:
    def test_leaves_the_effect_estimate_alone_on_average(self) -> None:
        """The adjustment removes noise, not signal."""
        estimates = []
        for experiment in generate_many(_spec(0.7), 300, seed=13):
            adjusted = cuped_adjust(experiment)
            estimates.append(adjusted.treatment.mean() - adjusted.control.mean())
        assert float(np.mean(estimates)) == pytest.approx(0.05, abs=0.01)

    def test_shrinks_the_outcome_variance(self) -> None:
        experiment = next(generate_many(_spec(0.8), 1, seed=14))
        adjusted = cuped_adjust(experiment)
        assert adjusted.control.var() < 0.5 * experiment.control.var()

    def test_relabels_an_adjusted_binary_metric_as_continuous(self) -> None:
        """Adjusted 0/1 outcomes are no longer 0/1, so a proportion test would be wrong."""
        spec = ExperimentSpec(n_per_arm=500, metric="binary", baseline=0.2, covariate_corr=0.6)
        adjusted = cuped_adjust(next(generate_many(spec, 1, seed=15)))
        assert adjusted.spec.metric == "continuous"
        assert not np.isin(adjusted.control, (0.0, 1.0)).all()

    def test_accepts_an_externally_supplied_theta(self) -> None:
        experiment = next(generate_many(_spec(0.6), 1, seed=16))
        assert not np.allclose(
            cuped_adjust(experiment, theta=0.0).control, cuped_adjust(experiment).control
        )
        np.testing.assert_allclose(cuped_adjust(experiment, theta=0.0).control, experiment.control)

    def test_keeps_the_covariate_so_the_result_stays_inspectable(self) -> None:
        experiment = next(generate_many(_spec(0.6), 1, seed=17))
        adjusted = cuped_adjust(experiment)
        np.testing.assert_array_equal(adjusted.control_pre, experiment.control_pre)

    def test_survives_a_perfectly_predicted_outcome(self) -> None:
        """Residual variance of exactly zero would be rejected by the spec. Floor it instead."""
        pre = np.tile([0.0, 1.0], 10)
        experiment = _degenerate(3.0 * pre, pre)
        adjusted = cuped_adjust(experiment)
        assert adjusted.spec.sd > 0.0
        assert adjusted.control.std() == pytest.approx(0.0, abs=1e-12)


class TestMeasuredAgainstTheory:
    def test_measured_reduction_matches_the_theory(self) -> None:
        """The headline: reduction tracks rho^2, measured against the realised correlation."""
        for rho in (0.0, 0.3, 0.6, 0.8):
            comparison = cuped_study(_spec(rho), n_sims=250, seed=18)
            assert comparison.mean_variance_reduction == pytest.approx(
                comparison.predicted_variance_reduction, abs=0.01
            )

    def test_a_better_covariate_removes_more_variance(self) -> None:
        curve = variance_reduction_curve(_spec(0.0), correlations=(0.2, 0.5, 0.8), n_sims=200)
        reductions = [curve[rho].mean_variance_reduction for rho in (0.2, 0.5, 0.8)]
        assert reductions == sorted(reductions)
        assert reductions[-1] > 0.55

    def test_binary_covariates_are_attenuated_by_the_copula(self) -> None:
        """A real gotcha, not a bug: the realised correlation is well below the requested one.

        The generator correlates *latent normals* and then thresholds them, which costs a lot
        of correlation. CUPED still delivers exactly ``1 - rho^2`` -- against the realised
        rho, which is the only one the data actually has.
        """
        spec = ExperimentSpec(n_per_arm=3000, metric="binary", baseline=0.10, covariate_corr=0.8)
        comparison = cuped_study(spec, n_sims=150, seed=19)
        assert comparison.mean_correlation < 0.65
        assert comparison.mean_variance_reduction == pytest.approx(
            comparison.predicted_variance_reduction, abs=0.02
        )

    def test_a_useless_covariate_buys_nothing(self) -> None:
        comparison = cuped_study(_spec(0.0), n_sims=250, seed=20)
        assert comparison.mean_variance_reduction == pytest.approx(0.0, abs=0.01)
        assert abs(comparison.power_gain) < 0.06


class TestItStaysHonest:
    def test_does_not_inflate_the_type_i_error(self) -> None:
        """Variance reduction is only worth having if the test still holds its size."""
        comparison = cuped_study(_spec(0.7, effect=0.0), n_sims=1500, seed=21)
        assert comparison.adjusted.reject_rate == pytest.approx(0.05, abs=0.02)

    def test_the_adjusted_estimate_is_unbiased_over_all_replications(self) -> None:
        """Theta is fitted on the same data it adjusts, so this is worth checking, not assuming.

        Averaged over *every* replication rather than the ones that reached significance --
        ``SimulationSummary.estimate_bias`` conditions on rejection and so carries the
        winner's curse on top, which would mask the question being asked here.
        """
        estimates = []
        for experiment in generate_many(_spec(0.7), 400, seed=22):
            estimates.append(cuped_test(experiment).adjusted.estimate)
        assert float(np.mean(estimates)) == pytest.approx(0.05, abs=0.006)

    def test_buys_real_power_at_the_same_sample_size(self) -> None:
        comparison = cuped_study(_spec(0.8), n_sims=400, seed=23)
        assert comparison.power_gain > 0.20
        assert comparison.effective_sample_size_multiplier > 2.0


class TestCupedResult:
    def test_reports_both_arms_of_the_comparison(self) -> None:
        result = cuped_test(next(generate_many(_spec(0.7), 1, seed=24)))
        assert result.adjusted.se < result.raw.se
        assert result.variance_reduction == pytest.approx(
            result.predicted_variance_reduction, abs=0.1
        )
        assert result.adjusted.n_per_arm == result.raw.n_per_arm

    def test_a_degenerate_experiment_reduces_nothing_rather_than_dividing_by_zero(self) -> None:
        result = cuped_test(_degenerate(np.zeros(20), np.zeros(20), metric="binary"))
        assert result.raw.se == 0.0
        assert result.variance_reduction == 0.0


class TestSequentialComposition:
    def test_fits_theta_on_the_truncated_data_only(self) -> None:
        """The safe order. Adjusting first and slicing after would leak the future into look 1."""
        experiment = next(generate_many(_spec(0.7, n=4000), 1, seed=25))
        early = cuped_at_each_look(experiment, 500)
        assert early.n_per_arm == 500
        # Adjusting the full experiment and slicing gives a different -- and dishonest -- answer.
        leaked = fixed_horizon_test(cuped_adjust(experiment).truncate(500))
        assert early.estimate != leaked.estimate

    def test_still_reduces_variance_at_an_interim_look(self) -> None:
        experiment = next(generate_many(_spec(0.8, n=4000), 1, seed=26))
        assert (
            cuped_at_each_look(experiment, 1000).se
            < fixed_horizon_test(experiment.truncate(1000)).se
        )


class TestComparisonArithmetic:
    def _summary(self, rate: float) -> SimulationSummary:
        return summarize(
            [True] * int(rate * 100) + [False] * (100 - int(rate * 100)),
            [1.0] * 100,
            [10] * 100,
            1.0,
        )

    def test_a_total_reduction_would_be_worth_unbounded_sample_size(self) -> None:
        """Unreachable in practice, but the arithmetic should not divide by zero on the way."""
        comparison = CupedComparison(
            raw=self._summary(0.5),
            adjusted=self._summary(0.9),
            mean_correlation=1.0,
            mean_theta=1.0,
            mean_variance_reduction=1.0,
            predicted_variance_reduction=1.0,
        )
        assert comparison.effective_sample_size_multiplier == float("inf")
        assert comparison.power_gain == pytest.approx(0.4)

    def test_half_the_variance_is_worth_twice_the_sample(self) -> None:
        comparison = CupedComparison(
            raw=self._summary(0.5),
            adjusted=self._summary(0.7),
            mean_correlation=0.7071,
            mean_theta=0.7,
            mean_variance_reduction=0.5,
            predicted_variance_reduction=0.5,
        )
        assert comparison.effective_sample_size_multiplier == pytest.approx(2.0)
