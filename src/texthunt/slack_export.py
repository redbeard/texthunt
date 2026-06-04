"""Export a balanced, throttle-friendly sample of Slack history to disk.

The exporter optimises for a *representative* corpus rather than a complete one:

- **across time** — the date range is split into windows and a capped slice is pulled from each, so
  the sample spans months rather than the last busy afternoon.
- **across people** — messages are capped per author, so one prolific poster cannot dominate.

Two layers keep us under Slack's rate limits: an injected ``throttle`` that paces every request, and
(in the real client) ``slack_sdk``'s retry handler that honours ``Retry-After`` on HTTP 429.

The output directory matches the layout :func:`texthunt.ingest.load_messages` reads, so an export
feeds straight into the rest of the pipeline.
"""

import json
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from texthunt.ingest import is_human_message

Throttle = Callable[[], None]


@dataclass(frozen=True)
class Channel:
    id: str
    name: str
    is_member: bool = True  # history is only readable for channels the token has joined
    num_members: int = 0  # a cheap activity proxy used to scan the busiest channels first


@dataclass(frozen=True)
class Page:
    items: list
    next_cursor: str | None


@dataclass(frozen=True)
class ExportSummary:
    n_channels: int
    n_messages: int
    n_authors: int
    n_skipped: int = 0
    dropped_authors: tuple[str, ...] = ()  # seen but too thin to keep — never hidden


class SlackClient(Protocol):
    def list_channels(self, cursor: str | None) -> Page: ...

    def history(
        self, channel_id: str, oldest: float, latest: float, cursor: str | None
    ) -> Page: ...

    def replies(self, channel_id: str, thread_ts: str, cursor: str | None) -> Page: ...


def time_windows(start: float, end: float, count: int) -> list[tuple[float, float]]:
    """Split ``[start, end]`` into ``count`` equal, contiguous windows."""
    width = (end - start) / count
    return [(start + i * width, start + (i + 1) * width) for i in range(count)]


def balance_by_author(messages: Sequence[dict], max_per_author: int) -> list[dict]:
    """Keep at most ``max_per_author`` messages from each author, preserving order."""
    kept: list[dict] = []
    seen: Counter[str] = Counter()
    for message in messages:
        author = message["user"]
        if seen[author] < max_per_author:
            seen[author] += 1
            kept.append(message)
    return kept


def export(
    client: SlackClient,
    out_dir: Path,
    *,
    start: float,
    end: float,
    n_windows: int,
    max_per_author: int,
    max_messages_per_window: int,
    throttle: Throttle,
    channel_names: Sequence[str] | None = None,
    max_channels: int | None = None,
    include_threads: bool = True,
) -> ExportSummary:
    """Export from an explicit (or capped) set of channels, balancing per author within each."""
    windows = time_windows(start, end, n_windows)
    channels = _select_channels(client, throttle, channel_names, max_channels)

    by_channel: dict[str, list[dict]] = {}
    skipped = 0
    for channel in channels:
        try:
            raw = _channel_history(
                client, channel, windows, throttle, max_messages_per_window, include_threads
            )
        except Exception:  # an unreadable channel (archived, restricted) shouldn't end the run
            skipped += 1
            continue
        balanced = balance_by_author([m for m in raw if is_human_message(m)], max_per_author)
        if balanced:
            by_channel[channel.name] = balanced

    _write(out_dir, by_channel)
    return _summarise(by_channel, skipped)


def export_balanced(
    client: SlackClient,
    out_dir: Path,
    *,
    start: float,
    end: float,
    n_windows: int,
    target_per_author: int,
    floor_per_author: int,
    max_messages_per_window: int,
    throttle: Throttle,
    include_threads: bool = True,
    max_channels: int | None = None,
) -> ExportSummary:
    """Scan joined channels (busiest first) until every retained author hits ``target_per_author``.

    Messages accumulate per author *across* channels (capped at the target), so prolific posters
    don't crowd the file. Scanning stops once no author sits between the floor and the target — i.e.
    everyone worth keeping is saturated — or the channels (or ``max_channels``) run out. Authors who
    never reach ``floor_per_author`` are dropped as too thin to profile.
    """
    windows = time_windows(start, end, n_windows)
    channels = _ranked_member_channels(client, throttle)

    by_channel: dict[str, list[dict]] = {}
    counts: Counter[str] = Counter()
    skipped = 0
    for channel in channels[:max_channels]:
        try:
            raw = _channel_history(
                client, channel, windows, throttle, max_messages_per_window, include_threads
            )
        except Exception:
            skipped += 1
            continue
        for message in raw:
            author = message.get("user", "")
            if is_human_message(message) and counts[author] < target_per_author:
                by_channel.setdefault(channel.name, []).append(message)
                counts[author] += 1
        if _everyone_saturated(counts, floor_per_author, target_per_author):
            break

    retained = _drop_thin_authors(by_channel, counts, floor_per_author)
    dropped = tuple(
        sorted(author for author, count in counts.items() if 0 < count < floor_per_author)
    )
    _write(out_dir, retained)
    return _summarise(retained, skipped, dropped)


def _channel_history(
    client: SlackClient,
    channel: Channel,
    windows: Sequence[tuple[float, float]],
    throttle: Throttle,
    max_messages_per_window: int,
    include_threads: bool,
) -> list[dict]:
    messages: list[dict] = []
    for oldest, latest in windows:
        window_messages = _paginate(
            lambda cursor, o=oldest, latest_=latest: client.history(channel.id, o, latest_, cursor),
            throttle,
            max_items=max_messages_per_window,
        )
        messages.extend(window_messages)
        if include_threads:
            for parent in window_messages:
                messages.extend(
                    _thread_replies(client, channel.id, parent, throttle, max_messages_per_window)
                )
    return messages


def _thread_replies(
    client: SlackClient, channel_id: str, parent: dict, throttle: Throttle, max_items: int
) -> list[dict]:
    if not parent.get("reply_count"):
        return []
    thread_ts = str(parent.get("thread_ts") or parent["ts"])
    replies = _paginate(
        lambda cursor: client.replies(channel_id, thread_ts, cursor), throttle, max_items
    )
    return [reply for reply in replies if reply.get("ts") != parent.get("ts")]


def _ranked_member_channels(client: SlackClient, throttle: Throttle) -> list[Channel]:
    channels = _paginate(lambda cursor: client.list_channels(cursor), throttle, max_items=None)
    members = [c for c in channels if c.is_member]
    return sorted(members, key=lambda c: c.num_members, reverse=True)


def _everyone_saturated(counts: Counter[str], floor: int, target: int) -> bool:
    in_between = any(floor <= count < target for count in counts.values())
    saturated = any(count >= target for count in counts.values())
    return saturated and not in_between


def _drop_thin_authors(
    by_channel: dict[str, list[dict]], counts: Counter[str], floor: int
) -> dict[str, list[dict]]:
    kept = {
        name: [m for m in messages if counts[m.get("user", "")] >= floor]
        for name, messages in by_channel.items()
    }
    return {name: messages for name, messages in kept.items() if messages}


def _select_channels(
    client: SlackClient,
    throttle: Throttle,
    channel_names: Sequence[str] | None,
    max_channels: int | None,
) -> list[Channel]:
    channels = _paginate(lambda cursor: client.list_channels(cursor), throttle, max_items=None)
    if channel_names is not None:
        wanted = set(channel_names)
        channels = [c for c in channels if c.name in wanted]
    readable = [c for c in channels if c.is_member]  # history needs membership
    return readable if max_channels is None else readable[:max_channels]


def _paginate(
    fetch: Callable[[str | None], Page], throttle: Throttle, max_items: int | None
) -> list:
    items: list = []
    cursor: str | None = None
    while True:
        page = fetch(cursor)
        items.extend(page.items)
        throttle()
        reached_cap = max_items is not None and len(items) >= max_items
        if not page.next_cursor or reached_cap:
            break
        cursor = page.next_cursor
    return items if max_items is None else items[:max_items]


def _write(out_dir: Path, by_channel: dict[str, list[dict]]) -> None:
    for channel_name, messages in by_channel.items():
        channel_dir = out_dir / channel_name
        channel_dir.mkdir(parents=True, exist_ok=True)
        (channel_dir / "export.json").write_text(json.dumps(messages, indent=2))


def _summarise(
    by_channel: dict[str, list[dict]], skipped: int, dropped_authors: tuple[str, ...] = ()
) -> ExportSummary:
    messages = [m for channel in by_channel.values() for m in channel]
    return ExportSummary(
        n_channels=len(by_channel),
        n_messages=len(messages),
        n_authors=len({m["user"] for m in messages}),
        n_skipped=skipped,
        dropped_authors=dropped_authors,
    )


def build_slack_client(token: str, *, retries: int = 5) -> SlackClient:
    """A real Slack client whose retry handler backs off on rate limits (HTTP 429)."""
    from slack_sdk import WebClient  # ty: ignore[unresolved-import]
    from slack_sdk.http_retry.builtin_handlers import (  # ty: ignore[unresolved-import]
        RateLimitErrorRetryHandler,
    )

    web = WebClient(token=token)
    web.retry_handlers.append(RateLimitErrorRetryHandler(max_retry_count=retries))
    return _WebApiClient(web)


class _WebApiClient:
    def __init__(self, web: Any) -> None:
        self._web = web

    def list_channels(self, cursor: str | None) -> Page:
        response = self._web.conversations_list(
            types="public_channel,private_channel", limit=200, cursor=cursor or None
        )
        channels = [
            Channel(
                c["id"],
                c["name"],
                is_member=c.get("is_member", False),
                num_members=c.get("num_members", 0),
            )
            for c in response["channels"]
        ]
        return Page(channels, response["response_metadata"].get("next_cursor") or None)

    def history(self, channel_id: str, oldest: float, latest: float, cursor: str | None) -> Page:
        response = self._web.conversations_history(
            channel=channel_id, oldest=oldest, latest=latest, limit=200, cursor=cursor or None
        )
        next_cursor = response.get("response_metadata", {}).get("next_cursor") or None
        return Page(list(response["messages"]), next_cursor)

    def replies(self, channel_id: str, thread_ts: str, cursor: str | None) -> Page:
        response = self._web.conversations_replies(
            channel=channel_id, ts=thread_ts, limit=200, cursor=cursor or None
        )
        next_cursor = response.get("response_metadata", {}).get("next_cursor") or None
        return Page(list(response["messages"]), next_cursor)
