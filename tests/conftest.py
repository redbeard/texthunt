import json
from pathlib import Path

import pytest


@pytest.fixture
def slack_export(tmp_path: Path) -> Path:
    """A minimal Slack export: per-channel folders of dated JSON message files."""
    general = tmp_path / "general"
    general.mkdir()
    (general / "2023-01-01.json").write_text(
        json.dumps(
            [
                {"type": "message", "user": "U1", "ts": "1672531200.000100", "text": "hey there"},
                {"type": "message", "user": "U2", "ts": "1672531201.000200", "text": "morning all"},
                # bot and system messages must be dropped
                {"type": "message", "subtype": "bot_message", "bot_id": "B1", "text": "deploy ok"},
                {
                    "type": "message",
                    "subtype": "channel_join",
                    "user": "U3",
                    "ts": "1672531202.0",
                    "text": "has joined the channel",
                },
                # empty text must be dropped
                {"type": "message", "user": "U1", "ts": "1672531203.0", "text": "   "},
            ]
        )
    )
    return tmp_path


_STYLES = {
    "U_shout": "WOW THIS IS GREAT NEWS EVERYONE LETS CELEBRATE THE BIG WIN TODAY",
    "U_whisper": "hmm, i'm really not so sure about any of this, maybe later, idk, we'll see",
    "U_formal": "I would respectfully suggest that we reconsider the proposal in due course",
    "U_ramble": "so anyway like i was just saying and then also another thing came up right",
}


@pytest.fixture
def rich_slack_export(tmp_path: Path) -> Path:
    """A larger export: several authors with distinct styles across two channels."""
    timestamp = 1_700_000_000
    for channel in ("general", "random"):
        channel_dir = tmp_path / channel
        channel_dir.mkdir()
        messages = []
        for author, style in _STYLES.items():
            for i in range(8):
                timestamp += 1
                messages.append(
                    {
                        "type": "message",
                        "user": author,
                        "ts": f"{timestamp}.0",
                        "text": f"{style} number {i}",
                    }
                )
        (channel_dir / "2023-11-14.json").write_text(json.dumps(messages))
    return tmp_path
