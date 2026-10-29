"""
Transcription metrics used in the paper.

Every metric compares a model's transcription (hypothesis) with the reference after the same
normalization: lowercase, ASCII punctuation and a few typographic symbols replaced by spaces,
whitespace collapsed, then split on whitespace into words.

  word inclusion rate (WIR)  share of reference words that appear in the hypothesis (multiset)
  WER                        word-level Levenshtein (S + I + D) / reference words
  BoW precision/recall/F1    multiset (bag-of-words) overlap, order-insensitive  <- headline
  aligned precision/recall/F1  correct words from the Levenshtein alignment, order-aware
  CER                        character-level Levenshtein / reference characters (reported
                             for convenience; not used for the paper's tables)

Per-item scores are averaged over items ("macro"); that is how every table in the paper is
computed. Micro (pooled-count) averages are also returned.

Degenerate outputs (`is_no_text_prediction`, `is_repetitive_prediction`) follow the filter
used for the out-of-distribution smart-glasses evaluation.
"""
from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Iterable, List, Optional, Tuple

_PUNCT_TABLE = str.maketrans({c: " " for c in string.punctuation})

SMART_PUNCT = {
    "®": " ",   # registered sign
    "™": " ",   # trade mark sign
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
    "–": " ",   # en dash
    "—": " ",   # em dash
    "…": " ",   # ellipsis
    "·": " ",   # middle dot
    "★": " ",
    "☆": " ",
}


def normalize_text(text: str, *, lowercase: bool = True, strip_punct: bool = True,
                   collapse_ws: bool = True) -> str:
    if lowercase:
        text = text.lower()
    if strip_punct:
        text = text.translate(_PUNCT_TABLE)
        for ch, repl in SMART_PUNCT.items():
            text = text.replace(ch, repl)
    if collapse_ws:
        text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    text = text.strip()
    return text.split() if text else []


def safe_div(num: float, den: float) -> float:
    return num / den if den != 0 else 0.0


# ----------------------------------------------------------------------------- word level

def word_inclusion(gt_tokens: List[str], hyp_tokens: List[str]) -> int:
    """Reference words found in the hypothesis, as a multiset (a word the reference has
    twice must appear twice)."""
    hyp = Counter(hyp_tokens)
    return sum(min(c, hyp.get(w, 0)) for w, c in Counter(gt_tokens).items())


@dataclass
class Alignment:
    correct: int
    substitutions: int
    insertions: int
    deletions: int


def align_words(gt: List[str], hyp: List[str]) -> Alignment:
    """Levenshtein alignment of two token lists. Tie-breaking (match/substitute, then insert,
    then delete) matches the scorer used for the paper."""
    n, m = len(gt), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0], back[i][0] = i, "D"
    for j in range(1, m + 1):
        dp[0][j], back[0][j] = j, "I"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if gt[i - 1] == hyp[j - 1]:
                best, move = dp[i - 1][j - 1], "M"
            else:
                best, move = dp[i - 1][j - 1] + 1, "S"
            if dp[i][j - 1] + 1 < best:
                best, move = dp[i][j - 1] + 1, "I"
            if dp[i - 1][j] + 1 < best:
                best, move = dp[i - 1][j] + 1, "D"
            dp[i][j], back[i][j] = best, move
    i, j = n, m
    c = s = ins = d = 0
    while i > 0 or j > 0:
        mv = back[i][j]
        if mv == "M":
            c += 1; i -= 1; j -= 1
        elif mv == "S":
            s += 1; i -= 1; j -= 1
        elif mv == "I":
            ins += 1; j -= 1
        elif mv == "D":
            d += 1; i -= 1
        else:
            break
    return Alignment(c, s, ins, d)


def levenshtein(a: str, b: str) -> int:
    """Character edit distance (two-row DP)."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


@dataclass
class Score:
    gt_words: int = 0
    hyp_words: int = 0
    wir_included: int = 0
    wir_rate: float = 0.0
    wer_correct: int = 0
    wer_sub: int = 0
    wer_ins: int = 0
    wer_del: int = 0
    wer: float = 0.0
    bow_overlap: int = 0
    bow_precision: float = 0.0
    bow_recall: float = 0.0
    bow_f1: float = 0.0
    align_precision: float = 0.0
    align_recall: float = 0.0
    align_f1: float = 0.0
    cer: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


def score(reference: str, hypothesis: str, *, lowercase: bool = True, strip_punct: bool = True,
          with_cer: bool = False) -> Score:
    """Score one transcription against its reference."""
    gt_norm = normalize_text(reference, lowercase=lowercase, strip_punct=strip_punct)
    hyp_norm = normalize_text(hypothesis, lowercase=lowercase, strip_punct=strip_punct)
    gt, hyp = tokenize(gt_norm), tokenize(hyp_norm)
    s = Score(gt_words=len(gt), hyp_words=len(hyp))

    s.wir_included = word_inclusion(gt, hyp)
    s.wir_rate = safe_div(s.wir_included, s.gt_words)

    al = align_words(gt, hyp)
    s.wer_correct, s.wer_sub, s.wer_ins, s.wer_del = al.correct, al.substitutions, al.insertions, al.deletions
    s.wer = safe_div(al.substitutions + al.insertions + al.deletions, s.gt_words)

    s.bow_overlap = sum((Counter(gt) & Counter(hyp)).values())
    s.bow_precision = safe_div(s.bow_overlap, s.hyp_words)
    s.bow_recall = safe_div(s.bow_overlap, s.gt_words)
    s.bow_f1 = safe_div(2 * s.bow_precision * s.bow_recall, s.bow_precision + s.bow_recall)

    s.align_precision = safe_div(al.correct, s.hyp_words)
    s.align_recall = safe_div(al.correct, s.gt_words)
    s.align_f1 = safe_div(2 * s.align_precision * s.align_recall, s.align_precision + s.align_recall)

    if with_cer:
        s.cer = safe_div(levenshtein(gt_norm, hyp_norm), len(gt_norm))
    return s


# Name kept for readers of the paper's code: score_sample(gt, hyp) == score(gt, hyp).
score_sample = score


def fbeta_bow(s: Score, beta: float = 1.0) -> float:
    """BoW F-beta (the GRPO reward; beta=1 is BoW F1)."""
    p, r = s.bow_precision, s.bow_recall
    return 0.0 if p + r == 0 else (1 + beta * beta) * p * r / (beta * beta * p + r)


def aggregate(scores: Iterable[Score]) -> dict:
    """Macro (mean of per-item values) and micro (pooled counts) averages."""
    scores = list(scores)
    n = len(scores)
    if not n:
        return {"n": 0, "macro": {}, "micro": {}}
    keys = ("wir_rate", "wer", "bow_precision", "bow_recall", "bow_f1",
            "align_precision", "align_recall", "align_f1")
    macro = {k: sum(getattr(s, k) for s in scores) / n for k in keys}
    if all(s.cer is not None for s in scores):
        macro["cer"] = sum(s.cer for s in scores) / n
    gt = sum(s.gt_words for s in scores)
    hyp = sum(s.hyp_words for s in scores)
    ov = sum(s.bow_overlap for s in scores)
    cor = sum(s.wer_correct for s in scores)
    bp, br = safe_div(ov, hyp), safe_div(ov, gt)
    ap, ar = safe_div(cor, hyp), safe_div(cor, gt)
    micro = {
        "wir_rate": safe_div(sum(s.wir_included for s in scores), gt),
        "wer": safe_div(sum(s.wer_sub + s.wer_ins + s.wer_del for s in scores), gt),
        "bow_precision": bp, "bow_recall": br, "bow_f1": safe_div(2 * bp * br, bp + br),
        "align_precision": ap, "align_recall": ar, "align_f1": safe_div(2 * ap * ar, ap + ar),
    }
    return {"n": n, "macro": macro, "micro": micro}


# ----------------------------------------------------------------------------- degenerate outputs

def _prediction_tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+(?:/[a-z0-9]+)?", text.lower())


_NO_TEXT = ("no text found", "no readable text", "no visible text", "no text visible",
            "no text detected", "no text is visible", "no text is found", "there is no text",
            "i can not find any text", "i cannot find any text", "i do not see any text",
            "i don't see any text", "unable to read any text")


def is_no_text_prediction(prediction: str) -> bool:
    """Empty output, or a short statement that there is no text."""
    text = re.sub(r"\s+", " ", prediction.strip().lower()).strip(" .,!?:;\"'")
    if not text:
        return True
    return len(_prediction_tokens(text)) <= 12 and any(p in text for p in _NO_TEXT)


def _long_numeric_countup(tokens: List[str]) -> bool:
    nums = []
    for tok in tokens[:220]:
        if tok.isdigit():
            nums.append(int(tok))
        elif nums:
            break
    if len(nums) < 20:
        return False
    run = best = 1
    for a, b in zip(nums, nums[1:]):
        run = run + 1 if b == a + 1 else 1
        best = max(best, run)
    return best >= 20


def is_repetitive_prediction(prediction: str) -> bool:
    """A repetition loop in the first 80 words (or a long 1, 2, 3, ... count-up)."""
    tokens = _prediction_tokens(prediction)
    if _long_numeric_countup(tokens):
        return True
    prefix = tokens[:80]
    if len(prefix) < 24:
        return False
    counts = Counter(prefix)
    unique_ratio = len(counts) / len(prefix)
    if counts.most_common(1)[0][1] >= 12 and unique_ratio <= 0.35:
        return True
    bigrams = list(zip(prefix, prefix[1:]))
    if bigrams and Counter(bigrams).most_common(1)[0][1] >= 8 and unique_ratio <= 0.45:
        return True
    for k in (2, 3, 4):
        chunks = [tuple(prefix[i:i + k]) for i in range(0, len(prefix) - k + 1, k)]
        if len(chunks) >= 8 and Counter(chunks).most_common(1)[0][1] >= 6:
            return True
    return False


def degenerate_reason(prediction: str) -> str:
    """'no_text', 'degenerate_repetition', or '' for a normal output."""
    if is_no_text_prediction(prediction):
        return "no_text"
    if is_repetitive_prediction(prediction):
        return "degenerate_repetition"
    return ""


__all__ = ["normalize_text", "tokenize", "score", "score_sample", "Score", "aggregate", "fbeta_bow",
           "levenshtein", "align_words", "word_inclusion", "is_no_text_prediction",
           "is_repetitive_prediction", "degenerate_reason"]
