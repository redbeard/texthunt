"""Score a query against author profiles and rank the candidates.

Open-set rejection and probability calibration build on top of this ranking in a later step; here we
provide the raw, sorted cosine similarities.
"""

from dataclasses import dataclass

import numpy as np

from texthunt.profiles import AuthorProfiles


@dataclass(frozen=True)
class Candidate:
    author_id: str
    score: float


def rank_authors(query_vector: np.ndarray, profiles: AuthorProfiles) -> list[Candidate]:
    """Return every author ranked by cosine similarity to ``query_vector``, most likely first."""
    similarities = profiles.matrix @ query_vector
    order = np.argsort(similarities)[::-1]
    return [Candidate(profiles.author_ids[i], float(similarities[i])) for i in order]
