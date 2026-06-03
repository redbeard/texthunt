"""Engine A — classical stylometry.

Combines character and word n-gram TF-IDF (the workhorses for short-text attribution) with a handful
of explicit, interpretable style rates. Character n-grams are deliberately weighted heavily because
they capture *how* someone writes (spelling, punctuation, casing) rather than *what* they write
about.
"""

import re
from collections.abc import Sequence

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import Normalizer, normalize

from texthunt.engine import Engine

_LETTERS = re.compile(r"[^\W\d_]", re.UNICODE)
_WORD = re.compile(r"\w+", re.UNICODE)
_EMOJI = re.compile(
    "[\U0001f300-\U0001faff\U00002600-\U000027bf\U0001f1e6-\U0001f1ff]",
    re.UNICODE,
)


class StyleFeatures(BaseEstimator, TransformerMixin):
    """Per-text stylistic rates, each in roughly the same scale so none dominates by magnitude."""

    feature_names = (
        "uppercase_ratio",
        "digit_ratio",
        "punctuation_ratio",
        "exclamation_ratio",
        "question_ratio",
        "ellipsis_ratio",
        "emoji_ratio",
        "avg_word_length",
    )
    n_features = len(feature_names)

    def fit(self, texts: Sequence[str], y: object = None) -> "StyleFeatures":
        return self

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        return np.array([self._rates(text) for text in texts], dtype=np.float64)

    def _rates(self, text: str) -> list[float]:
        length = max(len(text), 1)
        letters = _LETTERS.findall(text)
        words = _WORD.findall(text)
        uppercase = sum(c.isupper() for c in letters)
        return [
            uppercase / max(len(letters), 1),
            sum(c.isdigit() for c in text) / length,
            sum(not c.isalnum() and not c.isspace() for c in text) / length,
            text.count("!") / length,
            text.count("?") / length,
            text.count("...") / length,
            len(_EMOJI.findall(text)) / length,
            np.mean([len(w) for w in words]) / 10 if words else 0.0,
        ]


class StylometricEngine:
    """An :class:`Engine` over char/word n-gram TF-IDF and explicit style features."""

    def __init__(self) -> None:
        self._features = FeatureUnion(
            [
                ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=1)),
                ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1)),
                ("style", Pipeline([("rates", StyleFeatures()), ("scale", Normalizer())])),
            ]
        )

    def fit(self, texts: Sequence[str]) -> Engine:
        self._features.fit(texts)
        return self

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self._features.transform(texts)
        return normalize(vectors).toarray()


def build_stylometric_engine() -> StylometricEngine:
    return StylometricEngine()
