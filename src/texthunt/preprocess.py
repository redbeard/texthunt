"""Normalise message text and group messages into analysis blocks.

Two jobs that prepare raw messages for the engines:

- :func:`normalise` strips Slack markup that leaks *who* talks to *whom* (mentions, channel refs,
  links) so the engines learn writing style rather than social graph.
- :func:`block_messages` accumulates several short messages from one author into a block long enough
  to carry a stylistic signal.
"""

import re
from itertools import groupby

from texthunt.models import Block, Message

_MENTION = re.compile(r"<@[A-Z0-9]+(?:\|[^>]+)?>")
_CHANNEL_REF = re.compile(r"<#[A-Z0-9]+(?:\|[^>]+)?>")
_URL = re.compile(r"<https?://[^>]+>")
_WHITESPACE = re.compile(r"\s+")


def normalise(text: str) -> str:
    """Replace identity-leaking Slack markup with placeholders and tidy whitespace."""
    text = _MENTION.sub("@user", text)
    text = _CHANNEL_REF.sub("#channel", text)
    text = _URL.sub("@url", text)
    return _WHITESPACE.sub(" ", text).strip()


def block_messages(messages: list[Message], min_chars: int) -> list[Block]:
    """Join consecutive same-author, same-channel messages into blocks of at least ``min_chars``.

    Messages are normalised then accumulated in arrival order; a block is emitted as soon as it
    reaches ``min_chars``. A trailing remainder below the threshold is dropped, since an
    undersized block would be too weak to profile or score reliably.
    """
    blocks: list[Block] = []
    for (author_id, channel), group in groupby(messages, key=lambda m: (m.author_id, m.channel)):
        blocks.extend(_pack(list(group), author_id, channel, min_chars))
    return blocks


def _pack(messages: list[Message], author_id: str, channel: str, min_chars: int) -> list[Block]:
    blocks: list[Block] = []
    pending: list[str] = []
    for message in messages:
        cleaned = normalise(message.text)
        if cleaned:
            pending.append(cleaned)
        if sum(len(p) for p in pending) + len(pending) - 1 >= min_chars:
            blocks.append(_join(pending, author_id, channel))
            pending = []
    return blocks


def _join(parts: list[str], author_id: str, channel: str) -> Block:
    return Block(
        author_id=author_id,
        channel=channel,
        text=" ".join(parts),
        message_count=len(parts),
    )
