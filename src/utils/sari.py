from __future__ import annotations

import re
from collections import Counter


def tokenize(text: str) -> list[str]:
    """Lowercase and tokenize by whitespace and punctuation."""
    text = text.lower()
    tokens = re.findall(r"\b\w+\b", text)
    return tokens


def get_ngrams(tokens: list[str], n: int) -> Counter:
    """Return Counter of all n-grams of length n."""
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _precision_recall_f1(
    hyp_ngrams: Counter, ref_ngrams: Counter
) -> tuple[float, float, float]:
    """Compute precision, recall, and F1 between hypothesis and reference n-gram counters."""
    if not hyp_ngrams:
        return 0.0, 0.0, 0.0
    if not ref_ngrams:
        return 0.0, 0.0, 0.0

    overlap = hyp_ngrams & ref_ngrams
    overlap_count = sum(overlap.values())
    hyp_count = sum(hyp_ngrams.values())
    ref_count = sum(ref_ngrams.values())

    precision = overlap_count / hyp_count if hyp_count > 0 else 0.0
    recall = overlap_count / ref_count if ref_count > 0 else 0.0

    if precision + recall == 0:
        return precision, recall, 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def _sari_for_n(
    src_tokens: list[str],
    hyp_tokens: list[str],
    refs_tokens: list[list[str]],
    n: int,
) -> float:
    """Compute SARI score for a single n-gram order.

    Returns average of F_add, F_keep, F_del (each 0–1).
    """
    src_ngrams = get_ngrams(src_tokens, n)
    hyp_ngrams = get_ngrams(hyp_tokens, n)
    ref_ngrams_list = [get_ngrams(ref, n) for ref in refs_tokens]

    # Union and intersection of references
    ref_union: Counter = Counter()
    ref_inter: Counter = Counter()
    for i, rng in enumerate(ref_ngrams_list):
        if i == 0:
            ref_union = Counter(rng)
            ref_inter = Counter(rng)
        else:
            ref_union |= rng
            ref_inter &= rng

    # ---------- ADD ----------
    # n-grams in hyp but NOT in src
    added_hyp = hyp_ngrams - src_ngrams
    # Reward if they appear in references (union)
    added_ref = ref_union - src_ngrams

    _, _, f_add = _precision_recall_f1(added_hyp, added_ref)

    # ---------- KEEP ----------
    # n-grams in BOTH src and hyp
    kept_hyp = src_ngrams & hyp_ngrams
    # Reference for keep: n-grams in src ∩ ref_union
    kept_ref = src_ngrams & ref_union

    _, _, f_keep = _precision_recall_f1(kept_hyp, kept_ref)

    # ---------- DELETE ----------
    # n-grams in src but NOT in hyp
    deleted_hyp = src_ngrams - hyp_ngrams
    # Reward deletion if n-gram is also absent from references
    deleted_ref = src_ngrams - ref_union

    # F_del: precision only (reward deleting what references also deleted)
    del_hyp_count = sum(deleted_hyp.values())
    del_overlap = deleted_hyp & deleted_ref
    del_overlap_count = sum(del_overlap.values())
    f_del = del_overlap_count / del_hyp_count if del_hyp_count > 0 else 1.0

    return (f_add + f_keep + f_del) / 3.0


def compute_sari(
    original: str,
    simplified: str,
    references: list[str],
    n: int = 4,
) -> float:
    """Compute SARI score.

    Args:
        original: The source complex sentence.
        simplified: The model's simplified output.
        references: List of reference simplifications.
        n: Maximum n-gram order (default 4).

    Returns:
        SARI score between 0 and 100.
    """
    src_tokens = tokenize(original)
    hyp_tokens = tokenize(simplified)
    refs_tokens = [tokenize(r) for r in references]

    if not refs_tokens:
        raise ValueError("At least one reference is required.")

    scores = []
    for order in range(1, n + 1):
        s = _sari_for_n(src_tokens, hyp_tokens, refs_tokens, order)
        scores.append(s)

    return float(sum(scores) / len(scores)) * 100.0


def batch_sari(
    originals: list[str],
    simplified_list: list[str],
    references_list: list[list[str]],
) -> float:
    """Compute average SARI over a batch."""
    if not originals:
        return 0.0
    scores = [
        compute_sari(orig, simp, refs)
        for orig, simp, refs in zip(originals, simplified_list, references_list)
    ]
    return float(sum(scores) / len(scores))
