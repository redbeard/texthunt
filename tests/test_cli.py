from pathlib import Path
from types import SimpleNamespace

from pydantic import SecretStr

from texthunt import __version__
from texthunt.cli import main
from texthunt.slack_export import Channel, Page


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


class _StubSlackClient:
    def list_channels(self, cursor):
        return Page(items=[Channel("C1", "general")], next_cursor=None)

    def history(self, channel_id, oldest, latest, cursor):
        message = {"type": "message", "user": "U1", "ts": f"{oldest + 1}", "text": "hello team"}
        return Page(items=[message], next_cursor=None)


def test_export_command_writes_a_corpus(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        "texthunt.cli.load_settings", lambda: SimpleNamespace(slack_token=SecretStr("x"))
    )
    monkeypatch.setattr("texthunt.cli.build_slack_client", lambda token: _StubSlackClient())

    exit_code = main(
        ["export", "--out", str(tmp_path), "--days", "30", "--windows", "3", "--delay", "0"]
    )

    assert exit_code == 0
    assert "exported" in capsys.readouterr().out
    assert (tmp_path / "general" / "export.json").exists()
