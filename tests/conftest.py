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
