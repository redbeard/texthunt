"""Load a Slack workspace export into normalised :class:`Message` objects.

A Slack export is a directory of per-channel folders, each holding dated JSON files that contain an
array of raw message objects. Only genuine human messages are kept: bot posts and system events
(joins, topic changes, …) carry a ``subtype`` and are discarded.
"""

import json
from pathlib import Path

from texthunt.models import Message


def load_messages(export_dir: Path) -> list[Message]:
    """Return every human message in ``export_dir``, ordered by timestamp."""
    messages = [
        message
        for channel_dir in sorted(p for p in export_dir.iterdir() if p.is_dir())
        for day_file in sorted(channel_dir.glob("*.json"))
        for raw in json.loads(day_file.read_text())
        if (message := _to_message(raw, channel=channel_dir.name)) is not None
    ]
    messages.sort(key=lambda m: m.timestamp)
    return messages


def is_human_message(raw: dict) -> bool:
    """True for a real person's message — not a bot post or a system event (join, topic change…)."""
    return (
        raw.get("type") == "message"
        and "subtype" not in raw
        and bool(raw.get("user"))
        and bool(raw.get("text", "").strip())
    )


def _to_message(raw: dict, channel: str) -> Message | None:
    if not is_human_message(raw):
        return None
    return Message(
        author_id=raw["user"],
        channel=channel,
        timestamp=float(raw["ts"]),
        text=raw["text"],
    )
