"""Tests for the Bayesian decision rules.

Three claims are load-bearing here and each has a test that would fail loudly if it broke:

* ``test_posterior_probability_is_the_one_sided_p_value`` -- with weak priors the two
  frameworks compute the same number, so any later disagreement is about thresholds or
  language, not arithmetic.
* ``test_a_matched_threshold_collapses_the_disagreement`` -- and here is the proof that it
  *is* thresholds: matching them takes the disagreement rate from roughly 1 in 8 to roughly
  2 in 10,000, and the residual traces to two named approximations rather than to Bayes.
* ``test_posterior_monitoring_does_not_control_the_error_rate`` -- the claim that Bayesian
  analysis is immune to peeking, measured on the same substrate that produced §5.2.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from ab_testing_kit.bayes import (
    AgreementSummary,
    DecisionRule,
    agreement_study,
    analyse,
    bayes_peeking_study,
    bayes_study,
    difference_posterior,
)
from ab_testing_kit.peeking import peeking_study
from ab_testing_kit.simulation import (
    Experiment,
    ExperimentSpec,
    fixed_horizon_test,
    generate_many,
)


def _binary(effect: float = 0.0, n: int = 3000) -> ExperimentSpec:
    return ExperimentSpec(n_per_arm=n, metric="binary", baseline=0.10, effect=effect)


def _continuous(effect: float = 0.0, n: int = 2000) -> ExperimentSpec:
    return ExperimentSpec(n_per_arm=n, metric="continuous", baseline=0.0, effect=effect, sd=1.0)


def _flat(value: float, n: int = 50) -> Experiment:
    """A continuous experiment with no variation at all, so the posterior has no spread."""
    return Experiment(
        spec=ExperimentSpec(n_per_arm=n, metric="continuous", baseline=0.0, sd=1.0),
        control=np.zeros(n),
        treatment=np.full(n, value),
        control_pre=np.zeros(n),
        treatment_pre=np.zeros(n),
    )


class TestDecisionRule:
    def test_defaults_to_the_conventional_threshold(self) -> None:
        assert DecisionRule().prob_threshold == 0.95
        assert DecisionRule().max_expected_loss is None

    @pytest.mark.parametrize("threshold", [0.5, 1.0, 0.2, 1.5])
    def test_rejects_a_threshold_that_is_not_a_decision(self, threshold: float) -> None:
        with pytest.raises(ValueError, match="prob_threshold"):
            DecisionRule(prob_threshold=threshold)

    def test_rejects_a_non_positive_loss_budget(self) -> None:
        with pytest.raises(ValueError, match="max_expected_loss"):
            DecisionRule(max_expected_loss=0.0)


class TestPosterior:
    def test_recovers_the_true_difference(self) -> None:
        mean, sd = difference_posterior(next(generate_many(_binary(0.02), 1, seed=30)))
        assert mean == pytest.approx(0.02, abs=0.02)
        assert 0.0 < sd < 0.02

    def test_a_stronger_prior_pulls_the_estimate_towards_it(self) -> None:
        """Beta(50, 50) says 'I believe both arms convert at 50%' -- and it should show."""
        experiment = next(generate_many(_binary(0.02), 1, seed=31))
        weak, _ = difference_posterior(experiment, prior_a=1.0, prior_b=1.0)
        strong_mean, strong_sd = difference_posterior(experiment, prior_a=50.0, prior_b=50.0)
        assert abs(strong_mean) < abs(weak)
        assert strong_sd > 0.0

    def test_rejects_an_improper_prior(self) -> None:
        experiment = next(generate_many(_binary(), 1, seed=32))
        with pytest.raises(ValueError, match="Beta prior"):
            difference_posterior(experiment, prior_a=0.0)

    def test_handles_a_continuous_metric(self) -> None:
        mean, sd = difference_posterior(next(generate_many(_continuous(0.1), 1, seed=33)))
        assert mean == pytest.approx(0.1, abs=0.1)
        assert sd > 0.0

    def test_posterior_probability_is_the_one_sided_p_value(self) -> None:
        """The claim the whole module rests on: same arithmetic, different sentence.

        Under a flat prior and a normal posterior these are not merely similar, they are the
        same computation approached from two directions. Anything above a few parts in 10,000
        would mean one of the two implementations is wrong.
        """
        for experiment in generate_many(_binary(0.02), 40, seed=34):
            posterior = analyse(experiment)
            frequentist = fixed_horizon_test(experiment)
            one_sided = (
                1.0 - frequentist.p_value / 2.0
                if frequentist.estimate > 0.0
                else frequentist.p_value / 2.0
            )
            assert posterior.prob_better == pytest.approx(one_sided, abs=0.003)


class TestAnalyse:
    def test_expected_loss_falls_as_the_evidence_strengthens(self) -> None:
        weak = analyse(next(generate_many(_continuous(0.0, n=400), 1, seed=35)))
        strong = analyse(next(generate_many(_continuous(0.5, n=4000), 1, seed=35)))
        assert strong.expected_loss_ship < weak.expected_loss_ship
        assert strong.ship

    def test_the_two_losses_bracket_the_estimate(self) -> None:
        """``E[max(D,0)] - E[max(-D,0)] = E[D]`` is an identity, and a good arithmetic check."""
        result = analyse(next(generate_many(_continuous(0.1), 1, seed=36)))
        assert result.expected_loss_keep - result.expected_loss_ship == pytest.approx(
            result.estimate, abs=1e-12
        )

    def test_expected_loss_matches_direct_integration(self) -> None:
        """The closed form, checked against sampling from the posterior it summarises."""
        result = analyse(next(generate_many(_continuous(0.05, n=1000), 1, seed=37)))
        draws = stats.norm.rvs(
            loc=result.estimate,
            scale=result.sd,
            size=400_000,
            random_state=np.random.default_rng(5),
        )
        assert result.expected_loss_ship == pytest.approx(
            float(np.maximum(-draws, 0.0).mean()), abs=result.sd * 0.02
        )

    def test_a_loss_budget_can_veto_a_confident_result(self) -> None:
        """The second rule is a real veto, not decoration.

        The budget is set relative to the loss this experiment actually carries rather than
        to a fixed constant: expected loss shrinks super-exponentially in the z-statistic
        (1.9e-131 for a five-sigma result), so any hard-coded threshold either never bites or
        always does, depending on the effect size chosen.
        """
        experiment = next(generate_many(_continuous(0.1, n=2000), 1, seed=38))
        permissive = analyse(experiment, DecisionRule(prob_threshold=0.95))
        assert permissive.ship
        assert permissive.expected_loss_ship > 0.0
        strict = DecisionRule(
            prob_threshold=0.95, max_expected_loss=permissive.expected_loss_ship / 2.0
        )
        assert not analyse(experiment, strict).ship

    def test_credible_interval_widens_with_the_mass_requested(self) -> None:
        experiment = next(generate_many(_continuous(0.1), 1, seed=39))
        narrow = analyse(experiment, credible_mass=0.50).credible_interval
        wide = analyse(experiment, credible_mass=0.99).credible_interval
        assert wide[0] < narrow[0] and narrow[1] < wide[1]

    def test_rejects_an_impossible_credible_mass(self) -> None:
        with pytest.raises(ValueError, match="credible_mass"):
            analyse(next(generate_many(_continuous(), 1, seed=40)), credible_mass=1.0)

    def test_a_posterior_with_no_spread_still_decides(self) -> None:
        """Reachable at a tiny first look on a degenerate arm. Must not divide by zero."""
        assert analyse(_flat(1.0)).prob_better == 1.0
        assert analyse(_flat(-1.0)).prob_better == 0.0
        assert analyse(_flat(-1.0)).expected_loss_ship == pytest.approx(1.0)
        assert analyse(_flat(1.0)).expected_loss_ship == 0.0


class TestAgreement:
    @pytest.mark.parametrize("effect_spec", ["continuous", "binary"])
    def test_a_matched_threshold_collapses_the_disagreement(self, effect_spec: str) -> None:
        """The punchline. 0.975 is the posterior threshold that matches two-sided alpha=0.05.

        Moving from the conventional 0.95 to the matched 0.975 takes the disagreement rate
        from roughly 1 in 8 to roughly 2 in 10,000. What is left is *not* the two frameworks
        differing. It is two approximations, both named and both tiny:

        * a rate metric keeps its Beta(1,1) prior, which shrinks the estimate a hair;
        * a continuous metric meets Welch's **t** on the frequentist side and a **normal**
          posterior on the Bayesian side, and those differ slightly in the tail.

        Asserted as a rate rather than an exact count, because the residual is a handful of
        replications in tens of thousands and pinning it to zero would only be pinning this
        seed. The rates measured at 20,000 replications are in README 5.6.
        """
        spec = _continuous(0.05) if effect_spec == "continuous" else _binary(0.02)
        matched = agreement_study(spec, DecisionRule(prob_threshold=0.975), n_sims=500, seed=41)
        conventional = agreement_study(spec, n_sims=500, seed=41)
        assert matched.agreement_rate > 0.99
        assert matched.n_disagreements < conventional.n_disagreements / 10

    def test_the_conventional_threshold_is_the_more_permissive_one(self) -> None:
        """And the disagreements run one way only, which is the diagnostic."""
        summary = agreement_study(_binary(0.02), n_sims=400, seed=41)
        assert summary.bayes_only > 0
        assert summary.frequentist_only == 0
        assert summary.agreement_rate < 1.0

    def test_the_probability_gap_is_negligible(self) -> None:
        summary = agreement_study(_binary(0.02), n_sims=300, seed=42)
        assert summary.mean_abs_probability_gap < 0.005

    def test_counts_add_up_and_the_interval_covers_the_rate(self) -> None:
        summary = agreement_study(_binary(0.02), n_sims=200, seed=43)
        total = (
            summary.both_ship + summary.both_hold + summary.bayes_only + summary.frequentist_only
        )
        assert total == summary.n_sims
        lo, hi = summary.agreement_ci
        assert lo <= summary.agreement_rate <= hi

    def test_records_both_kinds_of_shipping_decision(self) -> None:
        """A real effect gives both_ship; a null gives mostly both_hold."""
        assert agreement_study(_binary(0.02), n_sims=200, seed=44).both_ship > 0
        assert agreement_study(_binary(0.0), n_sims=200, seed=44).both_hold > 100

    def test_a_frequentist_only_verdict_is_possible_at_a_strict_threshold(self) -> None:
        """Push the posterior threshold past 0.975 and the asymmetry reverses."""
        summary = agreement_study(
            _binary(0.02), DecisionRule(prob_threshold=0.999), n_sims=400, seed=45
        )
        assert summary.frequentist_only > 0


class TestFrequentistPropertiesOfABayesianRule:
    def test_ships_a_loser_at_about_the_threshold_rate(self) -> None:
        """A 0.95 posterior rule is a one-sided 5% rule, so this should land near 0.05."""
        summary = bayes_study(_binary(0.0), n_sims=1500, seed=46)
        assert summary.reject_rate == pytest.approx(0.05, abs=0.02)

    def test_has_power_against_a_real_effect(self) -> None:
        summary = bayes_study(_binary(0.02, n=3841), n_sims=600, seed=47)
        assert summary.reject_rate > 0.80

    def test_posterior_monitoring_does_not_control_the_error_rate(self) -> None:
        """The claim that going Bayesian fixes peeking, tested on the peeking substrate.

        Same generated experiments, same analysis points, same stopping-at-first-green shape.
        If posterior monitoring were self-correcting this curve would be flat, as
        ``sequential_study``'s is. It is not.
        """
        spec = _continuous(0.0, n=2000)
        one = bayes_peeking_study(spec, 1, n_sims=400, seed=48)
        many = bayes_peeking_study(spec, 20, n_sims=400, seed=48)
        assert one.reject_rate == pytest.approx(0.05, abs=0.03)
        assert many.reject_rate > 0.15
        assert many.reject_rate_ci[0] > one.reject_rate_ci[1]

    def test_it_inflates_at_much_the_same_rate_as_the_frequentist_rule(self) -> None:
        """Because they are the same arithmetic, they degrade the same way."""
        spec = _continuous(0.0, n=2000)
        bayesian = bayes_peeking_study(spec, 20, n_sims=400, seed=49)
        frequentist = peeking_study(spec, 20, n_sims=400, seed=49)
        assert bayesian.reject_rate == pytest.approx(frequentist.reject_rate, abs=0.08)

    def test_a_loss_budget_curbs_the_inflation(self) -> None:
        """Not a fix, but the one Bayesian lever that actually bites here."""
        spec = _continuous(0.0, n=2000)
        loose = bayes_peeking_study(spec, 20, n_sims=400, seed=50)
        strict = bayes_peeking_study(
            spec, 20, DecisionRule(max_expected_loss=1e-4), n_sims=400, seed=50
        )
        assert strict.reject_rate < loose.reject_rate

    def test_stops_earlier_when_it_stops(self) -> None:
        summary = bayes_peeking_study(_continuous(0.15, n=2000), 10, n_sims=300, seed=51)
        assert summary.mean_n_per_arm < 2000


class TestAgreementSummaryArithmetic:
    def test_rate_and_disagreement_count_are_consistent(self) -> None:
        summary = AgreementSummary(
            n_sims=100,
            both_ship=40,
            both_hold=50,
            bayes_only=7,
            frequentist_only=3,
            mean_abs_probability_gap=1e-5,
        )
        assert summary.n_disagreements == 10
        assert summary.agreement_rate == pytest.approx(0.90)
        lo, hi = summary.agreement_ci
        assert lo < 0.90 < hi
