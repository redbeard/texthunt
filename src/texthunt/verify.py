"""Rank authors, reject unknowns, and turn similarities into calibrated probabilities.

The decision layer answers two questions a raw cosine score cannot:

- *Is the author even in our gallery?* — settled by comparing the top score to ``reject_threshold``
  (the Equal Error Rate operating point) and reported as a calibrated ``probability_unknown``.
- *How confident are we in each candidate?* — a logistic fit (Platt scaling) maps the top score to
  P(known), which is then split across candidates by a softmax over their similarities.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression

from texthunt.metrics import equal_error_rate
from texthunt.profiles import AuthorProfiles


@dataclass(frozen=True)
class Candidate:
    author_id: str
    score: float


@dataclass(frozen=True)
class ScoredCandidate:
    author_id: str
    probability: float


@dataclass(frozen=True)
class Verdict:
    prediction: str | None
    probability_unknown: float
    is_unknown: bool
    ranked: list[ScoredCandidate]


def rank_authors(query_vector: np.ndarray, profiles: AuthorProfiles) -> list[Candidate]:
    """Return every author ranked by cosine similarity to ``query_vector``, most likely first."""
    similarities = profiles.matrix @ query_vector
    order = np.argsort(similarities)[::-1]
    return [Candidate(profiles.author_ids[i], float(similarities[i])) for i in order]


@dataclass(frozen=True)
class Calibrator:
    """Maps a top-1 similarity to P(known) and carries the rejection threshold."""

    model: LogisticRegression
    reject_threshold: float
    temperature: float

    @classmethod
    def fit(
        cls,
        scores: Sequence[float],
        is_known: Sequence[bool],
        temperature: float = 0.1,
    ) -> "Calibrator":
        model = LogisticRegression().fit(np.array(scores).reshape(-1, 1), list(is_known))
        _, threshold = equal_error_rate(scores, is_known)
        return cls(model=model, reject_threshold=threshold, temperature=temperature)

    def probability_known(self, score: float) -> float:
        known_column = list(self.model.classes_).index(True)
        return float(self.model.predict_proba([[score]])[0, known_column])


def identify(query_vector: np.ndarray, profiles: AuthorProfiles, calibrator: Calibrator) -> Verdict:
    ranking = rank_authors(query_vector, profiles)
    top = ranking[0]

    probability_known = calibrator.probability_known(top.score)
    conditional = _softmax([c.score for c in ranking], calibrator.temperature)
    ranked = [
        ScoredCandidate(c.author_id, probability_known * share)
        for c, share in zip(ranking, conditional, strict=True)
    ]

    is_unknown = top.score < calibrator.reject_threshold
    return Verdict(
        prediction=None if is_unknown else top.author_id,
        probability_unknown=1.0 - probability_known,
        is_unknown=is_unknown,
        ranked=ranked,
    )


def _softmax(values: Sequence[float], temperature: float) -> np.ndarray:
    scaled = np.array(values) / temperature
    exponentiated = np.exp(scaled - scaled.max())
    return exponentiated / exponentiated.sum()
