from texthunt.evaluate import author_disjoint_split, evaluate_engine, topic_aware_split
from texthunt.features import build_stylometric_engine
from texthunt.models import Block

STYLES = {
    "shouter": "WOW THIS IS GREAT NEWS EVERYONE LETS CELEBRATE TODAY",
    "whisperer": "hmm, i'm really not so sure about this, maybe later, idk...",
    "emoji": "love it 😍😍 so good 🔥 lets gooo 🚀 amazing stuff 🎉 yes",
    "formal": "I would respectfully suggest that we reconsider the proposal.",
    "hashtagger": "shipping #today #blessed #grind no days off #startup #hustle",
    "rambler": "so anyway like i was saying and then also another thing right",
}


def _corpus(channels: tuple[str, ...] = ("general", "random")) -> list[Block]:
    return [
        Block(
            author_id=author,
            channel=channel,
            text=f"{text} (variation {i})",
            message_count=3,
        )
        for author, text in STYLES.items()
        for channel in channels
        for i in range(4)
    ]


def test_author_disjoint_split_keeps_unknown_authors_out_of_gallery():
    split = author_disjoint_split(_corpus(), unknown_fraction=0.34, query_fraction=0.5, seed=7)

    gallery_authors = {b.author_id for b in split.gallery}
    unknown_authors = {b.author_id for b in split.unknown_queries}

    assert gallery_authors.isdisjoint(unknown_authors)
    assert unknown_authors
    assert {b.author_id for b in split.known_queries} <= gallery_authors


def test_topic_aware_split_puts_query_channels_outside_the_gallery():
    split = topic_aware_split(_corpus(), unknown_fraction=0.34, seed=7)

    for author in {b.author_id for b in split.known_queries}:
        gallery_channels = {b.channel for b in split.gallery if b.author_id == author}
        query_channels = {b.channel for b in split.known_queries if b.author_id == author}
        assert gallery_channels.isdisjoint(query_channels)


def test_evaluation_beats_random_and_separates_known_from_unknown():
    split = author_disjoint_split(_corpus(), unknown_fraction=0.34, query_fraction=0.5, seed=7)

    report = evaluate_engine(split, build_stylometric_engine)

    assert report.top1_accuracy > report.random_baseline
    assert report.verification_auc > 0.7
    assert 0.0 <= report.reject_threshold <= 1.0
