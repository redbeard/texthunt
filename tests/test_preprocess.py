from texthunt.models import Message
from texthunt.preprocess import block_messages, normalise


def test_normalise_replaces_mentions_channels_and_urls():
    raw = "hey <@U123> see <#C9|general> at <https://ex.com|the doc> please"
    assert normalise(raw) == "hey @user see #channel at @url please"


def test_normalise_collapses_whitespace_and_strips():
    assert normalise("  too   much\n\tspace  ") == "too much space"


def _messages(author: str, channel: str, texts: list[str]) -> list[Message]:
    return [
        Message(author_id=author, channel=channel, timestamp=float(i), text=t)
        for i, t in enumerate(texts)
    ]


def test_blocking_accumulates_until_min_chars():
    messages = _messages("U1", "general", ["ab", "cd", "ef", "gh"])

    blocks = block_messages(messages, min_chars=10)

    assert len(blocks) == 1
    assert blocks[0].text == "ab cd ef gh"
    assert blocks[0].message_count == 4
    assert blocks[0].author_id == "U1"


def test_blocking_flushes_each_time_min_chars_reached():
    messages = _messages("U1", "general", ["aaaa", "bbbb", "cccc", "dddd"])

    blocks = block_messages(messages, min_chars=4)

    assert [b.text for b in blocks] == ["aaaa", "bbbb", "cccc", "dddd"]


def test_blocking_does_not_mix_authors_or_channels():
    messages = _messages("U1", "general", ["hello there"]) + _messages(
        "U2", "general", ["hello there"]
    )

    blocks = block_messages(messages, min_chars=1)

    assert {b.author_id for b in blocks} == {"U1", "U2"}
    assert all(b.message_count == 1 for b in blocks)


def test_blocking_drops_trailing_remainder_below_min_chars():
    messages = _messages("U1", "general", ["aaaa", "bb"])

    blocks = block_messages(messages, min_chars=4)

    assert [b.text for b in blocks] == ["aaaa"]
