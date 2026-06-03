from pathlib import Path

from texthunt.ingest import load_messages


def test_loads_human_messages_with_channel_and_timestamp(slack_export: Path):
    messages = load_messages(slack_export)

    assert [m.author_id for m in messages] == ["U1", "U2"]
    assert all(m.channel == "general" for m in messages)
    assert messages[0].text == "hey there"
    assert messages[0].timestamp == 1672531200.0001


def test_drops_bot_system_and_empty_messages(slack_export: Path):
    messages = load_messages(slack_export)

    authors = {m.author_id for m in messages}
    assert "B1" not in authors  # bot
    assert "U3" not in authors  # channel_join system message
    assert all(m.text.strip() for m in messages)


def test_returns_messages_in_timestamp_order(slack_export: Path):
    messages = load_messages(slack_export)

    timestamps = [m.timestamp for m in messages]
    assert timestamps == sorted(timestamps)
