from __future__ import annotations

import math
from dataclasses import dataclass

stop_words = {
    "the",
    "and",
    "to",
    "of",
    "a",
    "in",
    "is",
    "it",
    "you",
    "that",
    "he",
    "was",
    "for",
    "on",
    "are",
    "with",
    "as",
    "i",
    "his",
    "they",
    "be",
    "at",
    "one",
    "have",
    "this",
    "from",
    "or",
    "had",
    "by",
    "but",
    "not",
    "what",
    "all",
    "were",
    "we",
    "when",
    "your",
    "can",
    "said",
    "there",
    "use",
    "an",
    "each",
    "which",
    "do",
    "how",
    "their",
    "if",
    "will",
    "up",
    "other",
    "about",
    "out",
    "many",
    "then",
    "them",
    "these",
    "so",
    "some",
    "her",
    "would",
    "make",
    "like",
    "him",
    "into",
    "time",
    "has",
    "look",
    "two",
    "more",
    "write",
    "go",
    "see",
}


@dataclass(frozen=True)
class RetrievalConfig:
    max_terms_for_candidates: int = 3
    max_candidates_total: int = 50_000
    max_candidates_per_term: int = 30_000

    idf_threshold: float = 0.0
    min_terms_after_threshold: int = 1

    allow_fallback_full_retrieval: bool = True


@dataclass(frozen=True)
class BM25Config:
    # global saturation parameter
    k1: float = 1.2

    # Fielded BM25 boosts (title emphasis)
    boost_title: float = 2.5
    boost_body: float = 1.0

    # per-field length normalization strengths
    b_title: float = 0.75
    b_body: float = 0.75

    # ignore terms with idf < idf_threshold
    idf_threshold: float = 0.5
    # clamp negative idf to 0
    clamp_negative_idf: bool = True
    # keep at least n terms after thresholding
    min_terms_after_threshold: int = 1


def bm25_idf(num_docs: int, df: int, *, clamp_negative: bool = True) -> float:
    """
    idf: ln((N - df + 0.5) / (df + 0.5))
    """
    if num_docs <= 0 or df <= 0:
        return 0.0
    val = math.log((num_docs - df + 0.5) / (df + 0.5))
    if clamp_negative and val < 0.0:
        return 0.0
    return val
