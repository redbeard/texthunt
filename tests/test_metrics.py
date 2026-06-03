from texthunt.metrics import equal_error_rate, top_k_accuracy


def test_equal_error_rate_is_zero_for_separable_scores():
    scores = [0.9, 0.8, 0.2, 0.1]
    is_known = [True, True, False, False]

    eer, threshold = equal_error_rate(scores, is_known)

    assert eer == 0.0
    assert 0.2 < threshold <= 0.8


def test_equal_error_rate_is_high_for_inverted_scores():
    scores = [0.1, 0.2, 0.8, 0.9]
    is_known = [True, True, False, False]

    eer, _ = equal_error_rate(scores, is_known)

    assert eer == 1.0


def test_top_k_accuracy_counts_true_author_within_k():
    rankings = [["a", "b", "c"], ["b", "a", "c"], ["c", "b", "a"]]
    true_authors = ["a", "a", "a"]

    assert top_k_accuracy(rankings, true_authors, k=1) == 1 / 3
    assert top_k_accuracy(rankings, true_authors, k=2) == 2 / 3
    assert top_k_accuracy(rankings, true_authors, k=3) == 1.0
