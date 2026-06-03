from texthunt import __version__
from texthunt.cli import main


def test_cli_runs_and_reports_success():
    assert main([]) == 0


def test_version_is_exposed():
    assert __version__ == "0.1.0"
