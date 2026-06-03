"""Honest open-set evaluation: split, score, and pick the rejection threshold.

The splits are what make the numbers trustworthy:

- **author-disjoint** holds out whole authors as "unknowns", so we can measure whether the system
  correctly refuses to attribute text from someone it has never seen.
- **topic-aware** ensures a known author's query blocks come from channels absent from their
  gallery, so we measure *style* transfer not topic memorisation — usually the more sobering number.
"""

import random
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from sklearn.metrics import roc_auc_score

from texthunt.engine import Engine
from texthunt.metrics import equal_error_rate, top_k_accuracy
from texthunt.models import Block
from texthunt.profiles import build_profiles
from texthunt.verify import rank_authors


@dataclass(frozen=True)
class GalleryQuerySplit:
    gallery: list[Block]
    known_queries: list[Block]
    unknown_queries: list[Block]


@dataclass(frozen=True)
class EvaluationReport:
    n_known_authors: int
    n_queries: int
    top1_accuracy: float
    top5_accuracy: float
    verification_auc: float
    eer: float
    reject_threshold: float
    random_baseline: float


def _by_author(blocks: Sequence[Block]) -> dict[str, list[Block]]:
    grouped: dict[str, list[Block]] = defaultdict(list)
    for block in blocks:
        grouped[block.author_id].append(block)
    return grouped


def _partition_authors(
    blocks: Sequence[Block], unknown_fraction: float, seed: int
) -> tuple[dict[str, list[Block]], list[str], list[str]]:
    by_author = _by_author(blocks)
    authors = sorted(by_author)
    random.Random(seed).shuffle(authors)
    n_unknown = max(1, round(len(authors) * unknown_fraction))
    return by_author, authors[n_unknown:], authors[:n_unknown]


def author_disjoint_split(
    blocks: Sequence[Block], unknown_fraction: float, query_fraction: float, seed: int
) -> GalleryQuerySplit:
    by_author, known, unknown = _partition_authors(blocks, unknown_fraction, seed)
    rng = random.Random(seed)

    gallery: list[Block] = []
    known_queries: list[Block] = []
    for author in known:
        owned = list(by_author[author])
        rng.shuffle(owned)
        cut = max(1, round(len(owned) * query_fraction))
        known_queries.extend(owned[:cut])
        gallery.extend(owned[cut:])

    unknown_queries = [b for author in unknown for b in by_author[author]]
    return GalleryQuerySplit(gallery, known_queries, unknown_queries)


def topic_aware_split(
    blocks: Sequence[Block], unknown_fraction: float, seed: int
) -> GalleryQuerySplit:
    by_author, known, unknown = _partition_authors(blocks, unknown_fraction, seed)
    rng = random.Random(seed)

    gallery: list[Block] = []
    known_queries: list[Block] = []
    for author in known:
        by_channel = defaultdict(list)
        for block in by_author[author]:
            by_channel[block.channel].append(block)
        channels = sorted(by_channel)
        rng.shuffle(channels)
        held_out = set(channels[: len(channels) // 2])  # empty when the author has one channel
        for channel, owned in by_channel.items():
            (known_queries if channel in held_out else gallery).extend(owned)

    unknown_queries = [b for author in unknown for b in by_author[author]]
    return GalleryQuerySplit(gallery, known_queries, unknown_queries)


def evaluate_engine(
    split: GalleryQuerySplit, engine_factory: Callable[[], Engine]
) -> EvaluationReport:
    engine = engine_factory().fit([b.text for b in split.gallery])
    profiles = build_profiles(split.gallery, engine)

    def top_score_and_ranking(block: Block) -> tuple[float, list[str]]:
        ranking = rank_authors(engine.encode([block.text])[0], profiles)
        return ranking[0].score, [c.author_id for c in ranking]

    rankings: list[list[str]] = []
    true_authors: list[str] = []
    scores: list[float] = []
    is_known: list[bool] = []

    for block in split.known_queries:
        score, ranking = top_score_and_ranking(block)
        rankings.append(ranking)
        true_authors.append(block.author_id)
        scores.append(score)
        is_known.append(True)

    for block in split.unknown_queries:
        score, _ = top_score_and_ranking(block)
        scores.append(score)
        is_known.append(False)

    eer, threshold = equal_error_rate(scores, is_known)
    return EvaluationReport(
        n_known_authors=len(profiles.author_ids),
        n_queries=len(scores),
        top1_accuracy=top_k_accuracy(rankings, true_authors, k=1),
        top5_accuracy=top_k_accuracy(rankings, true_authors, k=5),
        verification_auc=float(roc_auc_score(is_known, scores)),
        eer=eer,
        reject_threshold=threshold,
        random_baseline=1 / len(profiles.author_ids),
    )
