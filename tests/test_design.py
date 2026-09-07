"""Tests for power, sample size and MDE.

The important tests here are not the ones that check the formula against itself. They are
``TestSimulatedPower``, which sizes an experiment with the formula, runs it thousands of
times against a known true effect, and checks that the fraction reaching significance is
the power that was requested.
"""

from __future__ import annotations

import pytest

from ab_testing_kit.design import analytic_power, mde, sample_size, validate_power
from ab_testing_kit.simulation import ExperimentSpec


class TestSampleSize:
    def test_matches_published_value_for_proportions(self) -> None:
        """p0=0.10, absolute MDE=0.02, alpha=0.05, power=0.80.

        Standard references put this near 3,840 per arm for the pooled two-proportion
        z-test without continuity correction. Pinned so a refactor cannot quietly drift.
        """
        assert sample_size("binary", baseline=0.10, mde_abs=0.02) == pytest.approx(3840, abs=10)

    def test_matches_closed_form_for_means(self) -> None:
        """n = 2 (z_a + z_b)^2 sd^2 / delta^2 -> 16 sd^2 / delta^2 at 80% power."""
        n = sample_size("continuous", baseline=0.0, mde_abs=0.5, sd=1.0)
        assert n == pytest.approx(63, abs=1)

    def test_quartering_the_effect_multiplies_n_by_sixteen(self) -> None:
        big = sample_size("continuous", baseline=0.0, mde_abs=0.4, sd=1.0)
        small = sample_size("continuous", baseline=0.0, mde_abs=0.1, sd=1.0)
        assert small / big == pytest.approx(16.0, rel=0.01)

    def test_more_power_needs_more_units(self) -> None:
        assert sample_size("binary", 0.10, 0.02, power=0.90) > sample_size(
            "binary", 0.10, 0.02, power=0.80
        )

    def test_stricter_alpha_needs_more_units(self) -> None:
        assert sample_size("binary", 0.10, 0.02, alpha=0.01) > sample_size(
            "binary", 0.10, 0.02, alpha=0.05
        )

    def test_sign_of_effect_is_ignored(self) -> None:
        assert sample_size("continuous", 0.0, 0.5) == sample_size("continuous", 0.0, -0.5)

    def test_rejects_zero_effect(self) -> None:
        with pytest.raises(ValueError, match="mde_abs"):
            sample_size("continuous", 0.0, 0.0)

    def test_rejects_impossible_binary_lift(self) -> None:
        with pytest.raises(ValueError, match="baseline \\+ mde"):
            sample_size("binary", baseline=0.95, mde_abs=0.10)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"alpha": 0.0}, "alpha"),
            ({"power": 1.0}, "power"),
        ],
    )
    def test_rejects_invalid_rates(self, kwargs: dict, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            sample_size("continuous", 0.0, 0.5, **kwargs)

    def test_rejects_unknown_metric(self) -> None:
        with pytest.raises(ValueError, match="metric"):
            sample_size("poisson", 1.0, 0.5)  # type: ignore[arg-type]

    def test_rejects_nonpositive_sd(self) -> None:
        with pytest.raises(ValueError, match="sd"):
            sample_size("continuous", 0.0, 0.5, sd=-1.0)


class TestAnalyticPower:
    def test_hits_target_at_the_computed_sample_size(self) -> None:
        """The formula must be self-consistent: size for 80%, get back 80%."""
        n = sample_size("binary", baseline=0.10, mde_abs=0.02, power=0.80)
        assert analytic_power("binary", 0.10, 0.02, n) == pytest.approx(0.80, abs=0.005)

    def test_hits_target_for_means(self) -> None:
        n = sample_size("continuous", baseline=0.0, mde_abs=0.3, sd=1.0, power=0.90)
        assert analytic_power("continuous", 0.0, 0.3, n) == pytest.approx(0.90, abs=0.01)

    def test_zero_effect_gives_power_equal_to_alpha(self) -> None:
        assert analytic_power("continuous", 0.0, 0.0, 1000, alpha=0.05) == 0.05

    def test_grows_with_sample_size(self) -> None:
        small = analytic_power("continuous", 0.0, 0.2, 200)
        large = analytic_power("continuous", 0.0, 0.2, 2000)
        assert small < large

    def test_rejects_tiny_sample(self) -> None:
        with pytest.raises(ValueError, match="n_per_arm"):
            analytic_power("continuous", 0.0, 0.2, 1)

    def test_rejects_out_of_range_binary_rates(self) -> None:
        with pytest.raises(ValueError, match="must both be in"):
            analytic_power("binary", 0.9, 0.2, 1000)

    def test_rejects_unknown_metric(self) -> None:
        with pytest.raises(ValueError, match="metric"):
            analytic_power("poisson", 1.0, 0.2, 1000)  # type: ignore[arg-type]


class TestMde:
    def test_round_trips_with_sample_size_for_means(self) -> None:
        n = sample_size("continuous", baseline=0.0, mde_abs=0.25, sd=1.5)
        assert mde("continuous", 0.0, n, sd=1.5) == pytest.approx(0.25, rel=0.01)

    def test_round_trips_with_sample_size_for_proportions(self) -> None:
        n = sample_size("binary", baseline=0.20, mde_abs=0.03)
        assert mde("binary", 0.20, n) == pytest.approx(0.03, rel=0.01)

    def test_shrinks_with_more_units(self) -> None:
        assert mde("binary", 0.2, 20_000) < mde("binary", 0.2, 2_000)

    def test_raises_when_no_effect_is_detectable(self) -> None:
        """At 5 units per arm on a 90% base rate there is only 10 points of headroom, and
        even consuming all of it tops out near 10% power -- so no MDE exists at 80%."""
        with pytest.raises(ValueError, match="no detectable effect"):
            mde("binary", baseline=0.90, n_per_arm=5)

    def test_rejects_invalid_inputs(self) -> None:
        with pytest.raises(ValueError, match="n_per_arm"):
            mde("continuous", 0.0, 1)
        with pytest.raises(ValueError, match="metric"):
            mde("poisson", 1.0, 100)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="baseline"):
            mde("binary", 1.5, 100)
        with pytest.raises(ValueError, match="sd"):
            mde("continuous", 0.0, 100, sd=0.0)


class TestSimulatedPower:
    """Size an experiment with the formula, then measure whether it delivers."""

    def test_binary_design_delivers_requested_power(self) -> None:
        n = sample_size("binary", baseline=0.10, mde_abs=0.02, power=0.80)
        spec = ExperimentSpec(n_per_arm=n, metric="binary", baseline=0.10, effect=0.02)
        check = validate_power(spec, n_sims=2_500, seed=101)
        assert check.simulated.reject_rate == pytest.approx(0.80, abs=0.02)
        assert check.agrees, (
            f"analytic {check.analytic:.4f} outside {check.simulated.reject_rate_ci}"
        )

    def test_continuous_design_delivers_requested_power(self) -> None:
        n = sample_size("continuous", baseline=0.0, mde_abs=0.2, sd=1.0, power=0.80)
        spec = ExperimentSpec(n_per_arm=n, metric="continuous", baseline=0.0, effect=0.2, sd=1.0)
        check = validate_power(spec, n_sims=4_000, seed=102)
        assert check.simulated.reject_rate == pytest.approx(0.80, abs=0.02)

    def test_type_i_error_matches_alpha_under_the_null(self) -> None:
        """A single look at the planned horizon: this is the 5% that peeking destroys."""
        spec = ExperimentSpec(n_per_arm=2_000, metric="binary", baseline=0.10, effect=0.0)
        check = validate_power(spec, n_sims=4_000, seed=103)
        assert check.simulated.reject_rate == pytest.approx(0.05, abs=0.01)

    def test_fixed_horizon_estimates_are_unbiased(self) -> None:
        spec = ExperimentSpec(
            n_per_arm=1_000, metric="continuous", baseline=0.0, effect=0.15, sd=1.0
        )
        check = validate_power(spec, n_sims=3_000, seed=104)
        # Averaged over rejections only, a well-powered fixed-horizon design is still
        # mildly optimistic -- but nothing like the bias peeking introduces.
        assert abs(check.simulated.estimate_bias) < 0.02

    def test_normal_approximation_over_promises_at_small_n(self) -> None:
        """A real limitation, surfaced rather than hidden: analytic_power ignores the
        t-distribution, so at small samples simulated power lands below the formula."""
        spec = ExperimentSpec(n_per_arm=12, metric="continuous", baseline=0.0, effect=1.2, sd=1.0)
        check = validate_power(spec, n_sims=4_000, seed=105)
        assert check.gap < 0.0
