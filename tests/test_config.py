import pytest

from texthunt.config import load_settings


def test_reads_slack_token_from_environment(monkeypatch):
    monkeypatch.setenv("SLACK_TOKEN", "xoxp-secret")

    assert load_settings().slack_token.get_secret_value() == "xoxp-secret"


def test_missing_token_raises_a_clear_error(monkeypatch, tmp_path):
    monkeypatch.delenv("SLACK_TOKEN", raising=False)
    monkeypatch.chdir(tmp_path)  # no .env here, so the loader has nothing to fall back on

    with pytest.raises(ValueError, match="slack_token"):
        load_settings()


def test_token_is_not_exposed_in_repr(monkeypatch):
    monkeypatch.setenv("SLACK_TOKEN", "xoxp-secret")

    assert "xoxp-secret" not in repr(load_settings())
