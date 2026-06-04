import numpy as np

from texthunt.embeddings import LuarEngine


def _stub_embed(texts):
    # Distinct, non-unit vectors so normalisation is observable.
    return np.array([[float(len(t)), 1.0, 0.0] for t in texts])


def test_engine_normalises_embeddings_to_unit_length():
    engine = LuarEngine(embed=_stub_embed)

    vectors = engine.encode(["a", "bb", "ccc"])

    assert vectors.shape == (3, 3)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)


def test_fit_is_a_noop_returning_self():
    engine = LuarEngine(embed=_stub_embed)

    assert engine.fit(["anything"]) is engine


def test_encoding_preserves_input_order():
    engine = LuarEngine(embed=_stub_embed)

    first, second = engine.encode(["short", "much longer text"])

    assert first[0] < second[0]  # length feature grows with text length, order kept
