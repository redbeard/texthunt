"""Pure evaluation metrics for open-set authorship verification."""

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import roc_curve


def equal_error_rate(scores: Sequence[float], is_known: Sequence[bool]) -> tuple[float, float]:
    """Return the Equal Error Rate and the score threshold at which it occurs.

    The threshold is where the false-accept rate (unknown authors scoring above it) equals the
    false-reject rate (known authors scoring below it) — the natural operating point for deciding
    "is this author in our gallery at all?".
    """
    false_accept, true_accept, thresholds = roc_curve(list(is_known), list(scores))
    false_reject = 1 - true_accept
    crossover = np.argmin(np.abs(false_accept - false_reject))
    eer = float((false_accept[crossover] + false_reject[crossover]) / 2)
    return eer, float(thresholds[crossover])


def top_k_accuracy(rankings: Sequence[Sequence[str]], true_authors: Sequence[str], k: int) -> float:
    """Fraction of queries whose true author appears in the top ``k`` ranked candidates."""
    hits = sum(true in ranking[:k] for ranking, true in zip(rankings, true_authors, strict=True))
    return hits / len(true_authors)
