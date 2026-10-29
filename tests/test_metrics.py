"""Known-value checks for hawkeye.metrics (values computed with the scorer used for the paper)."""
import math

from hawkeye.metrics import aggregate, degenerate_reason, fbeta_bow, normalize_text, score


def close(a, b, tol=1e-9):
    return math.isclose(a, b, abs_tol=tol)


def test_normalization_ignores_case_and_punctuation():
    assert normalize_text("Hello, World!") == "hello world"
    s = score("Hello, World!", "hello world")
    assert (s.wir_rate, s.wer, s.bow_f1, s.align_f1) == (1.0, 0.0, 1.0, 1.0)


def test_typographic_symbols():
    # the registered sign and em dash become spaces; curly quotes become straight quotes (they are
    # mapped after ASCII punctuation is stripped, exactly as in the paper's scorer)
    assert normalize_text("Coke® — Zero") == "coke zero"
    assert normalize_text("“Zero”") == '"zero"'


def test_partial_overlap():
    s = score("A B C D", "A C D E E")
    assert close(s.wir_rate, 0.75)
    assert close(s.wer, 0.75)               # 1 deletion + 2 insertions over 4 reference words
    assert s.bow_overlap == 3
    assert close(s.bow_precision, 0.6) and close(s.bow_recall, 0.75)
    assert close(s.bow_f1, 2 / 3)
    assert close(s.align_f1, 2 / 3)


def test_bag_of_words_is_order_insensitive_but_alignment_is_not():
    s = score("one two three", "three two one")
    assert close(s.bow_f1, 1.0)
    assert s.align_f1 < 1.0


def test_multiset_counts():
    s = score("go go go", "go")
    assert close(s.wir_rate, 1 / 3)
    assert close(s.bow_precision, 1.0) and close(s.bow_recall, 1 / 3)


def test_empty_cases():
    assert score("", "x").bow_f1 == 0.0
    s = score("x", "")
    assert (s.bow_f1, s.wer) == (0.0, 1.0)


def test_cer():
    assert close(score("abcd", "abed", with_cer=True).cer, 0.25)
    assert close(score("A B C D", "A C D E E", with_cer=True).cer, 5 / 7)


def test_fbeta_reward():
    s = score("a b c d", "a b")                 # precision 1, recall .5
    assert close(fbeta_bow(s, 1.0), 2 / 3)
    assert fbeta_bow(s, 0.5) > fbeta_bow(s, 1.0)  # F0.5 weights precision more


def test_aggregate_macro_and_micro():
    agg = aggregate([score("a b", "a b"), score("a b c d", "a")])
    assert agg["n"] == 2
    assert close(agg["macro"]["bow_recall"], (1.0 + 0.25) / 2)
    assert close(agg["micro"]["bow_recall"], 3 / 6)


def test_degenerate_outputs():
    assert degenerate_reason("") == "no_text"
    assert degenerate_reason("There is no text in this image.") == "no_text"
    assert degenerate_reason(" ".join(["bendon"] * 40)) == "degenerate_repetition"
    assert degenerate_reason(" ".join(str(i) for i in range(1, 40))) == "degenerate_repetition"
    assert degenerate_reason("Nutrition Facts Serving size 1 cup Calories 120") == ""
