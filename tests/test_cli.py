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
