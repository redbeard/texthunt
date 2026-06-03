"""Build one profile vector per author from their blocks.

An author's profile is the mean of their block vectors, re-normalised to unit length so it can be
compared to a query by cosine similarity (a plain dot product, since everything is L2-normalised).
"""

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import normalize

from texthunt.engine import Engine
from texthunt.models import Block


@dataclass(frozen=True)
class AuthorProfiles:
    """Aligned author identifiers and their unit-length profile vectors."""

    author_ids: list[str]
    matrix: np.ndarray  # shape (n_authors, dim)


def build_profiles(blocks: Sequence[Block], engine: Engine) -> AuthorProfiles:
    texts_by_author: dict[str, list[str]] = defaultdict(list)
    for block in blocks:
        texts_by_author[block.author_id].append(block.text)

    author_ids = sorted(texts_by_author)
    centroids = [engine.encode(texts_by_author[author]).mean(axis=0) for author in author_ids]
    return AuthorProfiles(author_ids=author_ids, matrix=normalize(np.array(centroids)))
