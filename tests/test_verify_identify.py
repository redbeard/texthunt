import numpy as np
import pytest

from texthunt.profiles import AuthorProfiles
from texthunt.verify import Calibrator, identify


def _calibrator() -> Calibrator:
    scores = [0.85, 0.78, 0.72, 0.20, 0.15, 0.05]
    is_known = [True, True, True, False, False, False]
    return Calibrator.fit(scores, is_known)


def test_calibrator_probability_increases_with_score():
    calibrator = _calibrator()

    assert calibrator.probability_known(0.9) > calibrator.probability_known(0.1)


def test_calibrator_threshold_sits_between_known_and_unknown_scores():
    assert 0.2 < _calibrator().reject_threshold <= 0.72


def _profiles() -> AuthorProfiles:
    matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    return AuthorProfiles(author_ids=["alice", "bob"], matrix=matrix)


def test_identify_names_the_closest_author_when_confident():
    verdict = identify(np.array([1.0, 0.0, 0.0]), _profiles(), _calibrator())

    assert verdict.prediction == "alice"
    assert not verdict.is_unknown
    assert verdict.ranked[0].author_id == "alice"


def test_identify_reports_unknown_when_no_author_is_close():
    orthogonal_to_everyone = np.array([0.0, 0.0, 1.0])

    verdict = identify(orthogonal_to_everyone, _profiles(), _calibrator())

    assert verdict.is_unknown
    assert verdict.prediction is None
    assert verdict.probability_unknown > 0.5


def test_identify_probabilities_form_a_distribution():
    verdict = identify(np.array([1.0, 0.0, 0.0]), _profiles(), _calibrator())

    total = verdict.probability_unknown + sum(c.probability for c in verdict.ranked)
    assert total == pytest.approx(1.0)
