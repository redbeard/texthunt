import numpy as np

from texthunt.features import build_stylometric_engine
from texthunt.models import Block
from texthunt.profiles import build_profiles
from texthunt.verify import rank_authors


def _block(author: str, text: str) -> Block:
    return Block(author_id=author, channel="general", text=text, message_count=1)


SHOUTER = ["TOTALLY AGREE WITH THIS", "YES ABSOLUTELY LETS GO", "BIG IF TRUE HONESTLY"]
WHISPERER = ["hmm, i'm not so sure...", "maybe? could be, idk...", "well, perhaps, we'll see..."]


def _corpus() -> list[Block]:
    return [_block("shouter", t) for t in SHOUTER] + [_block("whisperer", t) for t in WHISPERER]


def test_profiles_are_one_unit_vector_per_author():
    engine = build_stylometric_engine().fit([b.text for b in _corpus()])

    profiles = build_profiles(_corpus(), engine)

    assert profiles.author_ids == ["shouter", "whisperer"]
    assert np.allclose(np.linalg.norm(profiles.matrix, axis=1), 1.0)


def test_query_ranks_its_true_author_first():
    corpus = _corpus()
    engine = build_stylometric_engine().fit([b.text for b in corpus])
    profiles = build_profiles(corpus, engine)

    query = engine.encode(["SOUNDS GREAT LETS DO IT"])[0]
    ranking = rank_authors(query, profiles)

    assert ranking[0].author_id == "shouter"
    assert ranking[0].score >= ranking[1].score
