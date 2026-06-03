from pathlib import Path

from texthunt import __version__
from texthunt.cli import main


def test_cli_runs_and_reports_success():
    assert main([]) == 0


def test_version_is_exposed():
    assert __version__ == "0.1.0"


def test_evaluate_command_runs_end_to_end(rich_slack_export: Path, capsys):
    exit_code = main(["evaluate", "--data", str(rich_slack_export), "--min-chars", "20"])

    assert exit_code == 0
    assert "top-1 accuracy" in capsys.readouterr().out


def test_identify_command_prints_a_verdict(rich_slack_export: Path, capsys):
    exit_code = main(
        ["identify", "--data", str(rich_slack_export), "--min-chars", "20", "WOW BIG WIN TODAY"]
    )

    assert exit_code == 0
    assert "U_shout" in capsys.readouterr().out
