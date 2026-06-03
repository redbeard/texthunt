"""The contract every scoring engine implements.

An engine's sole job is to turn text into an L2-normalised vector. Everything downstream — profile
building, cosine ranking, rejection — is engine-agnostic, so the classical stylometry engine and the
neural style-embedding engine are interchangeable and directly comparable.
"""

from collections.abc import Sequence
from typing import Protocol

import numpy as np


class Engine(Protocol):
    def fit(self, texts: Sequence[str]) -> "Engine":
        """Learn any vocabulary or statistics the engine needs (a no-op for pretrained models)."""
        ...

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return an ``(len(texts), dim)`` array of L2-normalised vectors."""
        ...
