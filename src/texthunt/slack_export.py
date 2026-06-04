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


@dataclass(frozen=True)
class Page:
    items: list
    next_cursor: str | None


@dataclass(frozen=True)
class ExportSummary:
    n_channels: int
    n_messages: int
    n_authors: int


class SlackClient(Protocol):
    def list_channels(self, cursor: str | None) -> Page: ...

    def history(
        self, channel_id: str, oldest: float, latest: float, cursor: str | None
    ) -> Page: ...


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
) -> ExportSummary:
    windows = time_windows(start, end, n_windows)
    channels = _paginate(lambda cursor: client.list_channels(cursor), throttle, max_items=None)

    by_channel: dict[str, list[dict]] = {}
    for channel in channels:
        raw = [
            message
            for oldest, latest in windows
            for message in _paginate(
                lambda cursor, o=oldest, latest_=latest, c=channel: client.history(
                    c.id, o, latest_, cursor
                ),
                throttle,
                max_items=max_messages_per_window,
            )
        ]
        human = [m for m in raw if is_human_message(m)]
        balanced = balance_by_author(human, max_per_author)
        if balanced:
            by_channel[channel.name] = balanced

    _write(out_dir, by_channel)
    return _summarise(by_channel)


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


def _summarise(by_channel: dict[str, list[dict]]) -> ExportSummary:
    messages = [m for channel in by_channel.values() for m in channel]
    return ExportSummary(
        n_channels=len(by_channel),
        n_messages=len(messages),
        n_authors=len({m["user"] for m in messages}),
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
        channels = [Channel(c["id"], c["name"]) for c in response["channels"]]
        return Page(channels, response["response_metadata"].get("next_cursor") or None)

    def history(self, channel_id: str, oldest: float, latest: float, cursor: str | None) -> Page:
        response = self._web.conversations_history(
            channel=channel_id, oldest=oldest, latest=latest, limit=200, cursor=cursor or None
        )
        next_cursor = response.get("response_metadata", {}).get("next_cursor") or None
        return Page(list(response["messages"]), next_cursor)
