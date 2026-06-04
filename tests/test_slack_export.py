from pathlib import Path

from texthunt.ingest import load_messages
from texthunt.slack_export import (
    Channel,
    Page,
    balance_by_author,
    export,
    time_windows,
)


def test_time_windows_tile_the_range_without_gaps_or_overlap():
    windows = time_windows(0.0, 100.0, count=4)

    assert windows == [(0.0, 25.0), (25.0, 50.0), (50.0, 75.0), (75.0, 100.0)]


def test_balance_by_author_caps_each_person_but_keeps_others():
    messages = [{"user": "a"}, {"user": "a"}, {"user": "a"}, {"user": "b"}]

    balanced = balance_by_author(messages, max_per_author=2)

    assert [m["user"] for m in balanced] == ["a", "a", "b"]


class FakeSlackClient:
    """An in-memory Slack stand-in: serves canned channels and time-filtered history."""

    def __init__(self, channels: list[Channel], messages: dict[str, list[dict]]) -> None:
        self._channels = channels
        self._messages = messages
        self.history_queries: list[tuple[str, float, float]] = []

    def list_channels(self, cursor: str | None) -> Page:
        return Page(items=list(self._channels), next_cursor=None)

    def history(self, channel_id: str, oldest: float, latest: float, cursor: str | None) -> Page:
        self.history_queries.append((channel_id, oldest, latest))
        within = [m for m in self._messages[channel_id] if oldest <= float(m["ts"]) < latest]
        return Page(items=within, next_cursor=None)


def _message(user: str, ts: float, text: str = "a real human sentence") -> dict:
    return {"type": "message", "user": user, "ts": f"{ts}", "text": text}


def _fake_client() -> FakeSlackClient:
    general = [_message(user, ts) for ts in (10, 35, 60, 85) for user in ("u1", "u2")]
    general += [_message("u1", 11), _message("u1", 12)]  # u1 is chatty
    general.append({"type": "message", "subtype": "bot_message", "ts": "40", "text": "deploy ok"})
    random = [_message("u3", ts) for ts in (20, 70)]
    return FakeSlackClient(
        channels=[Channel("C1", "general"), Channel("C2", "random")],
        messages={"C1": general, "C2": random},
    )


def test_export_writes_a_corpus_that_ingest_can_read_back(tmp_path: Path):
    calls: list[int] = []
    client = _fake_client()

    export(
        client,
        tmp_path,
        start=0.0,
        end=100.0,
        n_windows=4,
        max_per_author=2,
        max_messages_per_window=100,
        throttle=lambda: calls.append(1),
    )

    messages = load_messages(tmp_path)
    assert {m.author_id for m in messages} == {"u1", "u2", "u3"}
    assert all(m.text for m in messages)  # bot message was filtered out


def test_export_caps_messages_per_author(tmp_path: Path):
    client = _fake_client()

    export(
        client,
        tmp_path,
        start=0.0,
        end=100.0,
        n_windows=4,
        max_per_author=2,
        max_messages_per_window=100,
        throttle=lambda: None,
    )

    messages = load_messages(tmp_path)
    assert sum(m.author_id == "u1" for m in messages) == 2


def test_export_queries_every_window_and_throttles_between_calls(tmp_path: Path):
    calls: list[int] = []
    client = _fake_client()

    summary = export(
        client,
        tmp_path,
        start=0.0,
        end=100.0,
        n_windows=4,
        max_per_author=10,
        max_messages_per_window=100,
        throttle=lambda: calls.append(1),
    )

    assert len(client.history_queries) == 2 * 4  # 2 channels x 4 windows
    assert len(calls) >= len(client.history_queries)  # throttled at least once per call
    assert summary.n_channels == 2
    assert summary.n_authors == 3
