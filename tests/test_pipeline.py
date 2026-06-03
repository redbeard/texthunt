from pathlib import Path

from texthunt.features import build_stylometric_engine
from texthunt.models import Block
from texthunt.pipeline import load_blocks, train_identifier

STYLES = {
    "shouter": "WOW THIS IS GREAT NEWS EVERYONE LETS CELEBRATE",
    "whisperer": "hmm, i'm really not so sure about this, maybe, idk...",
    "emoji": "love it 😍 so good 🔥 lets gooo 🚀 amazing 🎉 yes",
    "formal": "I would respectfully suggest we reconsider the proposal.",
}


def _corpus() -> list[Block]:
    return [
        Block(author_id=author, channel="general", text=f"{text} (note {i})", message_count=3)
        for author, text in STYLES.items()
        for i in range(6)
    ]


def test_load_blocks_reads_and_blocks_a_slack_export(slack_export: Path):
    blocks = load_blocks(slack_export, min_chars=1)

    assert blocks
    assert {b.author_id for b in blocks} == {"U1", "U2"}


def test_trained_identifier_attributes_text_in_a_known_style():
    identifier = train_identifier(_corpus(), build_stylometric_engine, seed=1)

    verdict = identifier.identify("WOW AMAZING LETS GO TEAM")

    assert verdict.ranked[0].author_id == "shouter"


def test_trained_identifier_rejects_out_of_distribution_text():
    identifier = train_identifier(_corpus(), build_stylometric_engine, seed=1)

    verdict = identifier.identify("¿qué tal? 漢字 123 ... ")

    assert verdict.is_unknown or verdict.probability_unknown > 0.3
