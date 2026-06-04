from collections import Counter
from pathlib import Path

from texthunt.ingest import load_messages
from texthunt.slack_export import (
    Channel,
    Page,
    balance_by_author,
    export,
    export_balanced,
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
    """An in-memory Slack stand-in: serves canned channels, time-filtered history, and threads."""

    def __init__(
        self,
        channels: list[Channel],
        messages: dict[str, list[dict]],
        replies: dict[tuple[str, str], list[dict]] | None = None,
    ) -> None:
        self._channels = channels
        self._messages = messages
        self._replies = replies or {}
        self.history_queries: list[tuple[str, float, float]] = []

    def list_channels(self, cursor: str | None) -> Page:
        return Page(items=list(self._channels), next_cursor=None)

    def history(self, channel_id: str, oldest: float, latest: float, cursor: str | None) -> Page:
        self.history_queries.append((channel_id, oldest, latest))
        within = [m for m in self._messages[channel_id] if oldest <= float(m["ts"]) < latest]
        return Page(items=within, next_cursor=None)

    def replies(self, channel_id: str, thread_ts: str, cursor: str | None) -> Page:
        return Page(items=list(self._replies.get((channel_id, thread_ts), [])), next_cursor=None)


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


def test_export_can_restrict_to_named_channels(tmp_path: Path):
    client = _fake_client()

    export(
        client,
        tmp_path,
        start=0.0,
        end=100.0,
        n_windows=2,
        max_per_author=10,
        max_messages_per_window=100,
        throttle=lambda: None,
        channel_names=["general"],
    )

    assert (tmp_path / "general").exists()
    assert not (tmp_path / "random").exists()


def test_export_caps_the_number_of_channels_scanned(tmp_path: Path):
    client = _fake_client()

    summary = export(
        client,
        tmp_path,
        start=0.0,
        end=100.0,
        n_windows=2,
        max_per_author=10,
        max_messages_per_window=100,
        throttle=lambda: None,
        max_channels=1,
    )

    assert summary.n_channels == 1
    assert {channel for channel, *_ in client.history_queries} == {"C1"}


def test_export_ignores_channels_the_token_has_not_joined(tmp_path: Path):
    client = FakeSlackClient(
        channels=[Channel("C1", "joined"), Channel("C2", "not-joined", is_member=False)],
        messages={"C1": [_message("u1", 10)], "C2": [_message("u9", 10)]},
    )

    export(
        client,
        tmp_path,
        start=0.0,
        end=100.0,
        n_windows=1,
        max_per_author=10,
        max_messages_per_window=100,
        throttle=lambda: None,
    )

    assert client.history_queries == [("C1", 0.0, 100.0)]  # C2 never queried


class _FlakyClient(FakeSlackClient):
    def history(self, channel_id, oldest, latest, cursor):
        if channel_id == "C2":
            raise RuntimeError("not_in_channel")
        return super().history(channel_id, oldest, latest, cursor)


def test_export_skips_channels_that_error_without_aborting(tmp_path: Path):
    client = _FlakyClient(
        channels=[Channel("C1", "good"), Channel("C2", "broken")],
        messages={"C1": [_message("u1", 10)], "C2": [_message("u9", 10)]},
    )

    summary = export(
        client,
        tmp_path,
        start=0.0,
        end=100.0,
        n_windows=1,
        max_per_author=10,
        max_messages_per_window=100,
        throttle=lambda: None,
    )

    assert summary.n_skipped == 1
    assert (tmp_path / "good").exists()
    assert not (tmp_path / "broken").exists()


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


def test_export_includes_thread_replies_when_enabled(tmp_path: Path):
    parent = {"type": "message", "user": "u1", "ts": "10", "text": "question?", "reply_count": 2}
    reply1 = {"type": "message", "user": "u2", "ts": "11", "text": "one answer", "thread_ts": "10"}
    reply2 = {"type": "message", "user": "u3", "ts": "12", "text": "another", "thread_ts": "10"}
    client = FakeSlackClient(
        channels=[Channel("C1", "general")],
        messages={"C1": [parent]},
        replies={("C1", "10"): [parent, reply1, reply2]},
    )

    export(
        client,
        tmp_path,
        start=0.0,
        end=100.0,
        n_windows=1,
        max_per_author=10,
        max_messages_per_window=100,
        throttle=lambda: None,
        include_threads=True,
    )

    assert {m.author_id for m in load_messages(tmp_path)} == {"u1", "u2", "u3"}


def test_export_omits_thread_replies_when_disabled(tmp_path: Path):
    parent = {"type": "message", "user": "u1", "ts": "10", "text": "question?", "reply_count": 1}
    reply = {"type": "message", "user": "u2", "ts": "11", "text": "an answer", "thread_ts": "10"}
    client = FakeSlackClient(
        channels=[Channel("C1", "general")],
        messages={"C1": [parent]},
        replies={("C1", "10"): [parent, reply]},
    )

    export(
        client,
        tmp_path,
        start=0.0,
        end=100.0,
        n_windows=1,
        max_per_author=10,
        max_messages_per_window=100,
        throttle=lambda: None,
        include_threads=False,
    )

    assert {m.author_id for m in load_messages(tmp_path)} == {"u1"}


def test_export_balanced_fills_to_target_drops_thin_authors_and_stops_early(tmp_path: Path):
    busy = [_message("rich", ts) for ts in range(0, 60)]
    more_busy = [_message("rich", ts) for ts in range(0, 60)]
    client = FakeSlackClient(
        channels=[
            Channel("C1", "big", num_members=100),
            Channel("C2", "mid", num_members=50),
            Channel("C3", "small", num_members=10),
        ],
        messages={
            "C1": busy + [_message("thin", ts) for ts in range(60, 65)],
            "C2": more_busy,
            "C3": [_message("never_scanned", 1)],
        },
    )

    summary = export_balanced(
        client,
        tmp_path,
        start=-1.0,
        end=100.0,
        n_windows=1,
        target_per_author=100,
        floor_per_author=20,
        max_messages_per_window=1000,
        throttle=lambda: None,
        include_threads=False,
    )

    counts = Counter(m.author_id for m in load_messages(tmp_path))
    assert counts["rich"] == 100  # capped at target, accumulated across two channels
    assert "thin" not in counts  # below the floor, dropped
    assert "never_scanned" not in counts  # C3 never scanned: target met after C2
    assert summary.n_authors == 1
    assert summary.dropped_authors == ("thin",)  # a dropped author is always reported


def test_export_balanced_scans_channels_most_active_first(tmp_path: Path):
    client = FakeSlackClient(
        channels=[
            Channel("C_small", "small", num_members=5),
            Channel("C_big", "big", num_members=500),
        ],
        messages={"C_small": [_message("u1", 10)], "C_big": [_message("u2", 10)]},
    )

    export_balanced(
        client,
        tmp_path,
        start=0.0,
        end=100.0,
        n_windows=1,
        target_per_author=100,
        floor_per_author=1,
        max_messages_per_window=100,
        throttle=lambda: None,
        include_threads=False,
        max_channels=1,
    )

    assert client.history_queries[0][0] == "C_big"  # largest channel scanned first
