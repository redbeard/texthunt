import numpy as np

from texthunt.features import StyleFeatures, build_stylometric_engine


def test_style_features_are_finite_and_per_text():
    features = StyleFeatures().transform(["Hello, world!", "lol no"])

    assert features.shape == (2, StyleFeatures.n_features)
    assert np.isfinite(features).all()


def test_uppercase_ratio_distinguishes_shouting_from_whispering():
    shout, whisper = StyleFeatures().transform(["LOUD NOISES", "quiet murmurs"])
    index = StyleFeatures.feature_names.index("uppercase_ratio")

    assert shout[index] > whisper[index]


def test_engine_encodes_to_unit_vectors():
    engine = build_stylometric_engine().fit(["hello there", "general kenobi"])

    vectors = engine.encode(["hello there", "general kenobi"])

    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0)
