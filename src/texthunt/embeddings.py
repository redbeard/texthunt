"""Engine B — neural style embeddings via LUAR.

LUAR (Learning Universal Authorship Representations) is contrastively trained so that texts by the
same author land close together, and is deliberately content-insensitive — it captures *how*
someone writes rather than *what* about. That makes it a strong fit for short, topic-mixed Slack
messages.

``torch``/``transformers`` are heavy optional dependencies, so all of it is imported lazily inside
:func:`build_luar_engine`. The engine itself takes an injected ``embed`` function, which keeps it
trivially testable without downloading a model.
"""

from collections.abc import Callable, Sequence
from functools import lru_cache

import numpy as np
from sklearn.preprocessing import normalize

from texthunt.engine import Engine

DEFAULT_MODEL = "rrivera1849/LUAR-MUD"

Embedder = Callable[[Sequence[str]], np.ndarray]


class LuarEngine:
    """An :class:`Engine` backed by a style-embedding model (a no-op ``fit``: it is pretrained)."""

    def __init__(self, embed: Embedder) -> None:
        self._embed = embed

    def fit(self, texts: Sequence[str]) -> Engine:
        return self

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        raw = np.asarray(self._embed(list(texts)), dtype=np.float64)
        return normalize(raw)


def build_luar_engine(model_name: str = DEFAULT_MODEL, device: str = "cpu") -> LuarEngine:
    """Construct a LUAR engine, loading the model lazily and sharing it across calls."""
    return LuarEngine(embed=_luar_embedder(model_name, device))


@lru_cache(maxsize=2)
def _luar_embedder(model_name: str, device: str) -> Embedder:
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device).eval()

    def embed(texts: Sequence[str]) -> np.ndarray:
        # LUAR consumes (authors, posts_per_author, tokens); each text is a one-post author here.
        tokens = tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)
        episode = {key: value.unsqueeze(1) for key, value in tokens.items()}
        with torch.no_grad():
            embeddings = model(**episode)
        return embeddings.cpu().numpy()

    return embed
