"""Tests for the shared simulation substrate."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from ab_testing_kit.simulation import (
    Experiment,
    ExperimentSpec,
    fixed_horizon_test,
    generate,
    generate_many,
    summarize,
    wilson_interval,
)


class TestExperimentSpec:
    def test_rejects_tiny_sample(self) -> None:
        with pytest.raises(ValueError, match="n_per_arm"):
            ExperimentSpec(n_per_arm=1, metric="binary", baseline=0.1)

    def test_rejects_unknown_metric(self) -> None:
        with pytest.raises(ValueError, match="metric"):
            ExperimentSpec(n_per_arm=100, metric="poisson", baseline=1.0)  # type: ignore[arg-type]

    @pytest.mark.parametrize("baseline", [0.0, 1.0, -0.1, 1.5])
    def test_rejects_out_of_range_binary_baseline(self, baseline: float) -> None:
        with pytest.raises(ValueError, match="baseline"):
            ExperimentSpec(n_per_arm=100, metric="binary", baseline=baseline)

    def test_rejects_effect_pushing_rate_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="baseline \\+ effect"):
            ExperimentSpec(n_per_arm=100, metric="binary", baseline=0.9, effect=0.2)

    def test_rejects_nonpositive_sd(self) -> None:
        with pytest.raises(ValueError, match="sd"):
            ExperimentSpec(n_per_arm=100, metric="continuous", baseline=0.0, sd=0.0)

    def test_rejects_impossible_correlation(self) -> None:
        with pytest.raises(ValueError, match="covariate_corr"):
            ExperimentSpec(n_per_arm=100, metric="continuous", baseline=0.0, covariate_corr=1.0)

    def test_treatment_mean(self) -> None:
        spec = ExperimentSpec(n_per_arm=100, metric="binary", baseline=0.10, effect=0.02)
        assert spec.treatment_mean == pytest.approx(0.12)


class TestGeneration:
    def test_same_seed_gives_identical_data(self) -> None:
        spec = ExperimentSpec(n_per_arm=500, metric="continuous", baseline=10.0, sd=2.0)
        a = next(generate_many(spec, 1, seed=7))
        b = next(generate_many(spec, 1, seed=7))
        np.testing.assert_array_equal(a.control, b.control)
        np.testing.assert_array_equal(a.treatment, b.treatment)

    def test_different_seeds_give_different_data(self) -> None:
        spec = ExperimentSpec(n_per_arm=500, metric="continuous", baseline=10.0, sd=2.0)
        a = next(generate_many(spec, 1, seed=7))
        b = next(generate_many(spec, 1, seed=8))
        assert not np.array_equal(a.control, b.control)

    def test_replication_identity_is_independent_of_study_size(self) -> None:
        """Replication 3 of a 5-run study equals replication 3 of a 500-run study.

        This is the property that makes a surprising replication inspectable on its own.
        """
        spec = ExperimentSpec(n_per_arm=200, metric="binary", baseline=0.2)
        short = list(generate_many(spec, 5, seed=42))[3]
        long = list(generate_many(spec, 500, seed=42))[3]
        np.testing.assert_array_equal(short.treatment, long.treatment)

    def test_rejects_zero_replications(self) -> None:
        spec = ExperimentSpec(n_per_arm=100, metric="binary", baseline=0.2)
        with pytest.raises(ValueError, match="n_sims"):
            list(generate_many(spec, 0, seed=0))

    def test_generate_accepts_a_caller_supplied_rng(self) -> None:
        """The single-experiment entry point, for callers driving their own generator."""
        spec = ExperimentSpec(n_per_arm=300, metric="continuous", baseline=1.0, sd=2.0)
        a = generate(spec, np.random.default_rng(99))
        b = generate(spec, np.random.default_rng(99))
        np.testing.assert_array_equal(a.control, b.control)
        assert a.control.size == a.treatment.size == 300
        assert a.spec is spec

    def test_binary_arms_are_zero_one(self) -> None:
        spec = ExperimentSpec(n_per_arm=1000, metric="binary", baseline=0.3)
        exp = next(generate_many(spec, 1, seed=0))
        assert set(np.unique(exp.control)) <= {0.0, 1.0}

    def test_binary_rate_matches_baseline(self) -> None:
        spec = ExperimentSpec(n_per_arm=200_000, metric="binary", baseline=0.30, effect=0.05)
        exp = next(generate_many(spec, 1, seed=0))
        assert exp.control.mean() == pytest.approx(0.30, abs=0.01)
        assert exp.treatment.mean() == pytest.approx(0.35, abs=0.01)

    def test_continuous_mean_and_sd_match_spec(self) -> None:
        spec = ExperimentSpec(
            n_per_arm=200_000, metric="continuous", baseline=10.0, effect=0.5, sd=3.0
        )
        exp = next(generate_many(spec, 1, seed=0))
        assert exp.control.mean() == pytest.approx(10.0, abs=0.05)
        assert exp.treatment.mean() == pytest.approx(10.5, abs=0.05)
        assert exp.control.std(ddof=1) == pytest.approx(3.0, abs=0.05)

    @pytest.mark.parametrize("rho", [-0.6, 0.0, 0.3, 0.8])
    def test_continuous_covariate_correlation_is_as_requested(self, rho: float) -> None:
        spec = ExperimentSpec(
            n_per_arm=100_000, metric="continuous", baseline=5.0, sd=2.0, covariate_corr=rho
        )
        exp = next(generate_many(spec, 1, seed=0))
        assert exp.covariate_correlation() == pytest.approx(rho, abs=0.02)

    def test_covariate_correlation_is_zero_for_a_degenerate_arm(self) -> None:
        """A constant covariate has no correlation to report; 0.0 rather than a nan."""
        spec = ExperimentSpec(n_per_arm=10, metric="continuous", baseline=1.0)
        exp = Experiment(
            spec=spec,
            control=np.arange(10.0),
            treatment=np.arange(10.0),
            control_pre=np.zeros(10),
            treatment_pre=np.zeros(10),
        )
        assert exp.covariate_correlation() == 0.0

    def test_binary_covariate_correlation_is_attenuated_but_monotone(self) -> None:
        """Documents the latent-normal caveat rather than pretending it away."""
        spec = ExperimentSpec(n_per_arm=100_000, metric="binary", baseline=0.3, covariate_corr=0.8)
        realised = next(generate_many(spec, 1, seed=0)).covariate_correlation()
        assert 0.0 < realised < 0.8

        weaker = ExperimentSpec(
            n_per_arm=100_000, metric="binary", baseline=0.3, covariate_corr=0.4
        )
        assert next(generate_many(weaker, 1, seed=0)).covariate_correlation() < realised


class TestTruncate:
    def test_returns_a_prefix(self) -> None:
        spec = ExperimentSpec(n_per_arm=1000, metric="continuous", baseline=0.0)
        exp = next(generate_many(spec, 1, seed=1))
        cut = exp.truncate(250)
        assert cut.spec.n_per_arm == 250
        np.testing.assert_array_equal(cut.control, exp.control[:250])
        np.testing.assert_array_equal(cut.control_pre, exp.control_pre[:250])

    def test_preserves_ground_truth(self) -> None:
        spec = ExperimentSpec(n_per_arm=1000, metric="binary", baseline=0.2, effect=0.05)
        cut = next(generate_many(spec, 1, seed=1)).truncate(100)
        assert cut.spec.baseline == 0.2
        assert cut.spec.effect == 0.05

    def test_full_length_is_a_no_op(self) -> None:
        spec = ExperimentSpec(n_per_arm=100, metric="continuous", baseline=0.0)
        exp = next(generate_many(spec, 1, seed=1))
        np.testing.assert_array_equal(exp.truncate(100).control, exp.control)

    @pytest.mark.parametrize("n", [1, 0, 101, -5])
    def test_rejects_out_of_range(self, n: int) -> None:
        spec = ExperimentSpec(n_per_arm=100, metric="continuous", baseline=0.0)
        with pytest.raises(ValueError, match="n_per_arm"):
            next(generate_many(spec, 1, seed=1)).truncate(n)


class TestFixedHorizonTest:
    def test_welch_matches_scipy(self) -> None:
        """The hand-rolled Welch test exists so SE and df are inspectable; it must still
        agree with the reference implementation to floating-point precision."""
        spec = ExperimentSpec(n_per_arm=400, metric="continuous", baseline=10.0, effect=0.4, sd=2.0)
        exp = next(generate_many(spec, 1, seed=3))
        result = fixed_horizon_test(exp)
        expected = stats.ttest_ind(exp.treatment, exp.control, equal_var=False)
        assert result.statistic == pytest.approx(expected.statistic)
        assert result.p_value == pytest.approx(expected.pvalue)

    def test_binary_matches_manual_pooled_z(self) -> None:
        spec = ExperimentSpec(n_per_arm=2000, metric="binary", baseline=0.20, effect=0.03)
        exp = next(generate_many(spec, 1, seed=3))
        result = fixed_horizon_test(exp)

        n = spec.n_per_arm
        p_c, p_t = exp.control.mean(), exp.treatment.mean()
        p_pool = (exp.control.sum() + exp.treatment.sum()) / (2 * n)
        z = (p_t - p_c) / np.sqrt(p_pool * (1 - p_pool) * (2 / n))
        assert result.statistic == pytest.approx(z)
        assert result.p_value == pytest.approx(2 * stats.norm.sf(abs(z)))

    def test_estimate_is_treatment_minus_control(self) -> None:
        spec = ExperimentSpec(n_per_arm=500, metric="continuous", baseline=5.0, effect=1.0)
        exp = next(generate_many(spec, 1, seed=2))
        result = fixed_horizon_test(exp)
        assert result.estimate == pytest.approx(exp.treatment.mean() - exp.control.mean())

    def test_ci_brackets_the_estimate(self) -> None:
        spec = ExperimentSpec(n_per_arm=500, metric="binary", baseline=0.2, effect=0.05)
        result = fixed_horizon_test(next(generate_many(spec, 1, seed=2)))
        assert result.ci[0] < result.estimate < result.ci[1]

    def test_significance_agrees_with_alpha(self) -> None:
        spec = ExperimentSpec(n_per_arm=5000, metric="continuous", baseline=0.0, effect=0.1, sd=1.0)
        result = fixed_horizon_test(next(generate_many(spec, 1, seed=4)))
        assert result.significant == (result.p_value < 0.05)

    def test_degenerate_binary_arm_does_not_raise(self) -> None:
        """Every unit zero: SE is zero. Reported as p=1, not a ZeroDivisionError."""
        spec = ExperimentSpec(n_per_arm=10, metric="binary", baseline=0.001)
        exp = Experiment(
            spec=spec,
            control=np.zeros(10),
            treatment=np.zeros(10),
            control_pre=np.zeros(10),
            treatment_pre=np.zeros(10),
        )
        result = fixed_horizon_test(exp)
        assert result.p_value == 1.0
        assert not result.significant

    def test_degenerate_continuous_arm_does_not_raise(self) -> None:
        spec = ExperimentSpec(n_per_arm=10, metric="continuous", baseline=1.0)
        exp = Experiment(
            spec=spec,
            control=np.ones(10),
            treatment=np.ones(10),
            control_pre=np.zeros(10),
            treatment_pre=np.zeros(10),
        )
        result = fixed_horizon_test(exp)
        assert result.p_value == 1.0
        assert not result.significant

    @pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1])
    def test_rejects_invalid_alpha(self, alpha: float) -> None:
        spec = ExperimentSpec(n_per_arm=100, metric="binary", baseline=0.2)
        with pytest.raises(ValueError, match="alpha"):
            fixed_horizon_test(next(generate_many(spec, 1, seed=0)), alpha=alpha)

    def test_unbiased_under_the_null(self) -> None:
        """Averaged over replications with no selection rule, the estimate is centred on
        zero. This is the contrast the winner's curse in peeking.py is measured against."""
        spec = ExperimentSpec(n_per_arm=1000, metric="continuous", baseline=0.0, sd=1.0)
        estimates = [fixed_horizon_test(e).estimate for e in generate_many(spec, 2000, seed=11)]
        assert float(np.mean(estimates)) == pytest.approx(0.0, abs=0.005)


class TestWilsonInterval:
    def test_brackets_the_point_estimate(self) -> None:
        lo, hi = wilson_interval(500, 10_000)
        assert lo < 0.05 < hi

    def test_narrows_with_more_replications(self) -> None:
        narrow = wilson_interval(5_000, 100_000)
        wide = wilson_interval(50, 1_000)
        assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])

    def test_stays_within_zero_one_at_the_boundary(self) -> None:
        assert wilson_interval(0, 100)[0] == 0.0
        assert wilson_interval(100, 100)[1] == 1.0

    def test_rejects_invalid_inputs(self) -> None:
        with pytest.raises(ValueError, match="n must be"):
            wilson_interval(0, 0)
        with pytest.raises(ValueError, match="confidence"):
            wilson_interval(5, 100, confidence=1.5)


class TestSummarize:
    def test_counts_and_rate(self) -> None:
        summary = summarize([True, False, True, False], [1.0, 0.0, 3.0, 0.0], [10] * 4, 2.0)
        assert summary.n_sims == 4
        assert summary.n_rejected == 2
        assert summary.reject_rate == 0.5

    def test_mean_estimate_uses_only_rejections(self) -> None:
        """Averaging over all replications would hide the winner's curse entirely."""
        summary = summarize([True, False, True], [4.0, -99.0, 6.0], [10] * 3, 2.0)
        assert summary.mean_estimate == pytest.approx(5.0)
        assert summary.estimate_bias == pytest.approx(3.0)

    def test_mean_estimate_is_nan_when_nothing_rejected(self) -> None:
        summary = summarize([False, False], [1.0, 2.0], [10, 10], 0.0)
        assert np.isnan(summary.mean_estimate)

    def test_mean_n_averages_stopping_points(self) -> None:
        summary = summarize([True, True], [1.0, 1.0], [100, 300], 1.0)
        assert summary.mean_n_per_arm == pytest.approx(200.0)

    def test_rejects_empty_and_ragged_input(self) -> None:
        with pytest.raises(ValueError, match="zero replications"):
            summarize([], [], [], 0.0)
        with pytest.raises(ValueError, match="same length"):
            summarize([True], [1.0, 2.0], [10], 0.0)
