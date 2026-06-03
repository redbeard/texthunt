"""End-to-end wiring: a Slack export in, a trained identifier out.

Training has two stages because the calibrator and the gallery want different data. The calibrator
is fit on an author-disjoint split so it sees genuine *unknown* authors and learns a realistic
rejection threshold; the final gallery is then built over *every* author so identification can name
anyone in the corpus.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from texthunt.engine import Engine
from texthunt.evaluate import author_disjoint_split
from texthunt.ingest import load_messages
from texthunt.models import Block
from texthunt.preprocess import block_messages, normalise
from texthunt.profiles import AuthorProfiles, build_profiles
from texthunt.verify import Calibrator, Verdict, identify, rank_authors

DEFAULT_MIN_CHARS = 200


def load_blocks(data_dir: Path, min_chars: int = DEFAULT_MIN_CHARS) -> list[Block]:
    return block_messages(load_messages(data_dir), min_chars)


@dataclass(frozen=True)
class Identifier:
    engine: Engine
    profiles: AuthorProfiles
    calibrator: Calibrator

    def identify(self, text: str) -> Verdict:
        query_vector = self.engine.encode([normalise(text)])[0]
        return identify(query_vector, self.profiles, self.calibrator)


def train_identifier(
    blocks: Sequence[Block],
    engine_factory: Callable[[], Engine],
    seed: int = 0,
) -> Identifier:
    calibrator = _fit_calibrator(blocks, engine_factory, seed)
    engine = engine_factory().fit([b.text for b in blocks])
    return Identifier(engine, build_profiles(blocks, engine), calibrator)


def _fit_calibrator(
    blocks: Sequence[Block], engine_factory: Callable[[], Engine], seed: int
) -> Calibrator:
    split = author_disjoint_split(blocks, unknown_fraction=0.3, query_fraction=0.5, seed=seed)
    engine = engine_factory().fit([b.text for b in split.gallery])
    profiles = build_profiles(split.gallery, engine)

    def top_score(block: Block) -> float:
        return rank_authors(engine.encode([block.text])[0], profiles)[0].score

    scores = [top_score(b) for b in split.known_queries + split.unknown_queries]
    is_known = [True] * len(split.known_queries) + [False] * len(split.unknown_queries)
    return Calibrator.fit(scores, is_known)
