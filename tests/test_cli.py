"""Tests for the CLI.

The CLI is the interface every README number is generated through, so the properties that
matter are that it is deterministic under a fixed seed, that it fails with a readable
message rather than a traceback, and that its exit codes are honest.
"""

from __future__ import annotations

import pytest

from ab_testing_kit.cli import build_parser, main


class TestParser:
    def test_requires_a_subcommand(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args([])

    def test_rejects_unknown_metric(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args(
                ["design", "--metric", "poisson", "--baseline", "0.1", "--mde", "0.02"]
            )

    def test_parses_look_counts(self) -> None:
        args = build_parser().parse_args(
            [
                "peeking",
                "--metric",
                "binary",
                "--baseline",
                "0.1",
                "--n",
                "1000",
                "--looks",
                "1,4,9",
            ]
        )
        assert args.looks == (1, 4, 9)

    def test_rejects_non_integer_look_counts(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args(
                [
                    "peeking",
                    "--metric",
                    "binary",
                    "--baseline",
                    "0.1",
                    "--n",
                    "1000",
                    "--looks",
                    "1,two",
                ]
            )

    def test_rejects_zero_look_count(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args(
                [
                    "peeking",
                    "--metric",
                    "binary",
                    "--baseline",
                    "0.1",
                    "--n",
                    "1000",
                    "--looks",
                    "0,5",
                ]
            )


class TestDesignCommand:
    def test_reports_sample_size(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["design", "--metric", "binary", "--baseline", "0.10", "--mde", "0.02"]) == 0
        out = capsys.readouterr().out
        assert "n per arm            3,841" in out
        assert "total n              7,682" in out
        assert "relative MDE         20.0%" in out

    def test_continuous_reports_sd_and_omits_relative_mde(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert (
            main(
                ["design", "--metric", "continuous", "--baseline", "0", "--mde", "0.5", "--sd", "2"]
            )
            == 0
        )
        out = capsys.readouterr().out
        assert "within-arm sd        2" in out
        assert "relative MDE" not in out

    def test_bad_input_exits_two_with_a_readable_message(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(["design", "--metric", "binary", "--baseline", "0.95", "--mde", "0.10"])
        assert code == 2
        assert "error: baseline + mde must be in (0, 1)" in capsys.readouterr().out


class TestMdeCommand:
    def test_round_trips_against_design(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["mde", "--metric", "binary", "--baseline", "0.10", "--n", "3841"]) == 0
        out = capsys.readouterr().out
        assert "absolute MDE         0.0199" in out or "absolute MDE         0.02" in out

    def test_reports_failure_when_nothing_is_detectable(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(["mde", "--metric", "binary", "--baseline", "0.90", "--n", "5"])
        assert code == 2
        assert "no detectable effect" in capsys.readouterr().out


class TestValidatePowerCommand:
    def test_reports_analytic_and_simulated_power(self, capsys: pytest.CaptureFixture[str]) -> None:
        code = main(
            [
                "validate-power",
                "--metric",
                "continuous",
                "--baseline",
                "0",
                "--n",
                "400",
                "--effect",
                "0.2",
                "--sims",
                "300",
                "--seed",
                "1",
            ]
        )
        assert code == 0
        out = capsys.readouterr().out
        assert "analytic power" in out
        assert "simulated power" in out
        assert "formula inside CI" in out

    def test_is_deterministic_under_a_fixed_seed(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Every README number is quoted next to its command; the two must agree forever."""
        argv = [
            "validate-power",
            "--metric",
            "binary",
            "--baseline",
            "0.2",
            "--n",
            "500",
            "--effect",
            "0.05",
            "--sims",
            "200",
            "--seed",
            "77",
        ]
        main(argv)
        first = capsys.readouterr().out
        main(argv)
        assert capsys.readouterr().out == first

    def test_different_seeds_give_different_output(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        base = [
            "validate-power",
            "--metric",
            "binary",
            "--baseline",
            "0.2",
            "--n",
            "500",
            "--effect",
            "0.05",
            "--sims",
            "200",
            "--seed",
        ]
        main([*base, "1"])
        first = capsys.readouterr().out
        main([*base, "2"])
        assert capsys.readouterr().out != first


class TestPeekingCommand:
    def test_prints_one_row_per_look_count(self, capsys: pytest.CaptureFixture[str]) -> None:
        code = main(
            [
                "peeking",
                "--metric",
                "continuous",
                "--baseline",
                "0",
                "--n",
                "600",
                "--looks",
                "1,5",
                "--sims",
                "300",
                "--seed",
                "3",
            ]
        )
        assert code == 0
        lines = capsys.readouterr().out.strip().splitlines()
        rows = [
            ln for ln in lines if ln.strip().startswith(("1 ", "5 ")) or ln.strip()[:1].isdigit()
        ]
        assert len([r for r in rows if r.split()[0] in {"1", "5"}]) == 2

    def test_labels_the_null_column_as_type_i_error(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(
            [
                "peeking",
                "--metric",
                "continuous",
                "--baseline",
                "0",
                "--n",
                "600",
                "--looks",
                "1",
                "--sims",
                "100",
                "--seed",
                "3",
            ]
        )
        out = capsys.readouterr().out
        assert "Type-I error" in out
        assert "(null)" in out

    def test_labels_the_alternative_column_as_power(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(
            [
                "peeking",
                "--metric",
                "continuous",
                "--baseline",
                "0",
                "--n",
                "600",
                "--effect",
                "0.2",
                "--looks",
                "1",
                "--sims",
                "100",
                "--seed",
                "3",
            ]
        )
        out = capsys.readouterr().out
        assert "power" in out
        assert "(alternative)" in out


class TestSequentialCommand:
    def test_prints_one_row_per_look_count(self, capsys: pytest.CaptureFixture[str]) -> None:
        code = main(
            [
                "sequential",
                "--metric",
                "continuous",
                "--baseline",
                "0",
                "--n",
                "600",
                "--looks",
                "1,5",
                "--sims",
                "200",
                "--seed",
                "3",
            ]
        )
        assert code == 0
        out = capsys.readouterr().out
        rows = [ln for ln in out.splitlines() if ln.split()[:1] in (["1"], ["5"])]
        assert len(rows) == 2
        assert "spending function    obrien-fleming" in out

    def test_reports_the_magnitude_column_alongside_the_signed_bias(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Both are needed under the null; see _print_study_table."""
        main(
            [
                "sequential",
                "--metric",
                "continuous",
                "--baseline",
                "0",
                "--n",
                "600",
                "--looks",
                "1",
                "--sims",
                "100",
                "--seed",
                "3",
            ]
        )
        out = capsys.readouterr().out
        assert "Type-I error" in out
        assert "est. bias" in out
        assert "mean |est|" in out

    def test_labels_the_alternative_column_as_power(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(
            [
                "sequential",
                "--metric",
                "continuous",
                "--baseline",
                "0",
                "--effect",
                "0.2",
                "--n",
                "600",
                "--looks",
                "1",
                "--sims",
                "100",
                "--seed",
                "3",
            ]
        )
        out = capsys.readouterr().out
        assert "power" in out
        assert "(alternative)" in out

    def test_accepts_an_alternative_spending_function(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(
            [
                "sequential",
                "--metric",
                "continuous",
                "--baseline",
                "0",
                "--n",
                "600",
                "--looks",
                "1",
                "--sims",
                "100",
                "--seed",
                "3",
                "--spending",
                "pocock",
            ]
        )
        assert code == 0
        assert "spending function    pocock" in capsys.readouterr().out

    def test_rejects_an_unknown_spending_function(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args(
                [
                    "sequential",
                    "--metric",
                    "binary",
                    "--baseline",
                    "0.1",
                    "--n",
                    "1000",
                    "--spending",
                    "haybittle-peto",
                ]
            )


class TestBoundaryCommand:
    def test_prints_one_row_per_look_and_the_budget_it_spends(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(["boundary", "--n", "3841", "--looks", "5"])
        assert code == 0
        out = capsys.readouterr().out
        rows = [ln for ln in out.splitlines() if ln.split()[:1] in ([str(i)] for i in range(1, 6))]
        assert len(rows) == 5
        assert "total alpha spent    0.050000" in out

    def test_shows_the_boundary_tightening_towards_the_horizon(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The point of the subcommand: the correction should be visible, not just asserted."""
        main(["boundary", "--n", "3841", "--looks", "5"])
        criticals = [
            float(ln.split()[3])
            for ln in capsys.readouterr().out.splitlines()
            if ln.split()[:1] in ([str(i)] for i in range(1, 6))
        ]
        assert criticals == sorted(criticals, reverse=True)
        assert criticals[0] > 4.0

    def test_marks_an_unusable_look_as_never(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Twenty O'Brien-Fleming looks cannot spend anything at the first one."""
        main(["boundary", "--n", "3841", "--looks", "20"])
        out = capsys.readouterr().out
        assert "never" in out
        assert "total alpha spent    0.050000" in out

    def test_pocock_boundary_is_flat_by_comparison(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(["boundary", "--n", "3841", "--looks", "5", "--spending", "pocock"])
        out = capsys.readouterr().out
        criticals = [
            float(ln.split()[3])
            for ln in out.splitlines()
            if ln.split()[:1] in ([str(i)] for i in range(1, 6))
        ]
        assert max(criticals) - min(criticals) < 0.1
        assert "spending function    pocock" in out


class TestCupedCommand:
    def test_prints_one_row_per_correlation(self, capsys: pytest.CaptureFixture[str]) -> None:
        code = main(
            [
                "cuped",
                "--metric",
                "continuous",
                "--baseline",
                "0",
                "--n",
                "800",
                "--effect",
                "0.1",
                "--corr",
                "0,0.6",
                "--sims",
                "120",
                "--seed",
                "4",
            ]
        )
        assert code == 0
        out = capsys.readouterr().out
        rows = [ln for ln in out.splitlines() if ln.strip().startswith(("0.00", "0.60"))]
        assert len(rows) == 2
        assert "realised rho" in out

    def test_measured_reduction_sits_next_to_the_prediction(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A number is only checkable if what it should have been is printed beside it."""
        main(
            [
                "cuped",
                "--metric",
                "continuous",
                "--baseline",
                "0",
                "--n",
                "800",
                "--corr",
                "0.6",
                "--sims",
                "120",
                "--seed",
                "4",
            ]
        )
        row = next(
            ln for ln in capsys.readouterr().out.splitlines() if ln.strip().startswith("0.60")
        ).split()
        measured, predicted = float(row[3]), float(row[4])
        assert measured == pytest.approx(predicted, abs=0.02)

    def test_warns_about_attenuation_for_binary_covariates(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(
            [
                "cuped",
                "--metric",
                "binary",
                "--baseline",
                "0.2",
                "--n",
                "800",
                "--corr",
                "0.8",
                "--sims",
                "60",
                "--seed",
                "4",
            ]
        )
        assert "attenuated" in capsys.readouterr().out

    def test_rejects_impossible_correlations(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args(
                [
                    "cuped",
                    "--metric",
                    "continuous",
                    "--baseline",
                    "0",
                    "--n",
                    "800",
                    "--corr",
                    "0.5,1.0",
                ]
            )

    def test_rejects_non_numeric_correlations(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args(
                [
                    "cuped",
                    "--metric",
                    "continuous",
                    "--baseline",
                    "0",
                    "--n",
                    "800",
                    "--corr",
                    "0.5,high",
                ]
            )


class TestBayesCommand:
    def test_prints_one_row_per_look_count(self, capsys: pytest.CaptureFixture[str]) -> None:
        code = main(
            [
                "bayes",
                "--metric",
                "continuous",
                "--baseline",
                "0",
                "--n",
                "600",
                "--looks",
                "1,5",
                "--sims",
                "150",
                "--seed",
                "6",
            ]
        )
        assert code == 0
        out = capsys.readouterr().out
        rows = [ln for ln in out.splitlines() if ln.split()[:1] in (["1"], ["5"])]
        assert len(rows) == 2
        assert "ships a loser" in out
        assert "P(B>A) >= 0.95" in out

    def test_labels_the_alternative_case(self, capsys: pytest.CaptureFixture[str]) -> None:
        main(
            [
                "bayes",
                "--metric",
                "continuous",
                "--baseline",
                "0",
                "--effect",
                "0.3",
                "--n",
                "600",
                "--looks",
                "1",
                "--sims",
                "100",
                "--seed",
                "6",
            ]
        )
        assert "ships a winner" in capsys.readouterr().out

    def test_reports_a_loss_budget_when_one_is_set(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(
            [
                "bayes",
                "--metric",
                "continuous",
                "--baseline",
                "0",
                "--n",
                "600",
                "--looks",
                "1",
                "--sims",
                "100",
                "--seed",
                "6",
                "--max-loss",
                "0.001",
            ]
        )
        assert code == 0
        assert "E[loss] <= 0.001" in capsys.readouterr().out


class TestAgreementCommand:
    def test_prints_one_row_per_threshold(self, capsys: pytest.CaptureFixture[str]) -> None:
        code = main(
            [
                "agreement",
                "--metric",
                "continuous",
                "--baseline",
                "0",
                "--effect",
                "0.1",
                "--n",
                "800",
                "--sims",
                "200",
                "--seed",
                "6",
            ]
        )
        assert code == 0
        out = capsys.readouterr().out
        rows = [
            ln
            for ln in out.splitlines()
            if ln.startswith("    ") and ln.strip().startswith(("0.950", "0.975", "0.990"))
        ]
        assert len(rows) == 3
        assert "compute the same number" in out

    def test_the_matched_threshold_is_the_one_that_agrees(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A flat prior at 0.975 should agree with the two-sided test on every replication."""
        main(
            [
                "agreement",
                "--metric",
                "continuous",
                "--baseline",
                "0",
                "--effect",
                "0.1",
                "--n",
                "800",
                "--thresholds",
                "0.95,0.975",
                "--sims",
                "200",
                "--seed",
                "6",
            ]
        )
        rates = {
            ln.split()[0]: float(ln.split()[1])
            for ln in capsys.readouterr().out.splitlines()
            if ln.startswith("    ") and ln.strip().startswith(("0.950", "0.975"))
        }
        assert rates["0.975"] == 1.0
        assert rates["0.950"] < 1.0

    def test_rejects_a_threshold_that_is_not_a_decision(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args(
                [
                    "agreement",
                    "--metric",
                    "continuous",
                    "--baseline",
                    "0",
                    "--n",
                    "800",
                    "--thresholds",
                    "0.95,0.4",
                ]
            )

    def test_rejects_non_numeric_thresholds(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args(
                [
                    "agreement",
                    "--metric",
                    "continuous",
                    "--baseline",
                    "0",
                    "--n",
                    "800",
                    "--thresholds",
                    "0.95,most",
                ]
            )
