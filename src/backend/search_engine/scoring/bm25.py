# backend/search_engine/scoring/bm25.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence
from collections.abc import Callable

from cpp_utils import PostingList  # type: ignore [import-untyped]


@dataclass(frozen=True)
class BM25Config:
    k1: float = 1.2  # how strong tf's influence is
    b: float = 0.75  # level of document normalization
    # idf thresholding -> ignore terms with idf < idf_threshold
    idf_threshold: float = 0.0
    # clamp negative idf to 0
    clamp_negative_idf: bool = True


def bm25_idf(num_docs: int, df: int, *, clamp_negative: bool = True) -> float:
    """
    idf: ln((N - df + 0.5) / (df + 0.5))
    """
    if num_docs <= 0:
        return 0.0
    # df can be 0, if term unknown
    if df <= 0:
        return 0.0

    val = math.log((num_docs - df + 0.5) / (df + 0.5))
    if clamp_negative and val < 0.0:
        return 0.0
    return val


def bm25_term_contribution(
    tf: int,
    doc_len: int,
    avgdl: float,
    *,
    idf: float,
    k1: float,
    b: float,
) -> float:
    if tf <= 0:
        return 0.0
    if avgdl <= 0:
        avgdl = 1.0

    denom = tf + k1 * (1.0 - b + b * (doc_len / avgdl))
    return idf * (tf * (k1 + 1.0) / denom)


def bm25_score_docs(
    query_terms: Sequence[str],
    *,
    postings_by_term: Mapping[str, PostingList],
    candidate_doc_ids: Iterable[int],
    num_docs: int,
    avgdl: float,
    get_doc_length: Callable[[int], int],
    cfg: BM25Config = BM25Config(),
) -> dict[int, float]:
    """
    scores only the provided candidate_doc_ids (typically: boolean result set)
    """
    # precompute term idf + apply thresholding
    term_idf: dict[str, float] = {}
    for t in query_terms:
        pl = postings_by_term.get(t)
        if pl is None:
            continue
        df = getattr(pl, "doc_frequency", None)
        if df is None:
            df = len(pl.postings)

        idf = bm25_idf(num_docs, int(df), clamp_negative=cfg.clamp_negative_idf)
        if idf >= cfg.idf_threshold:
            term_idf[t] = idf

    # fallback: if thresholding removes everything, take all terms w/o threshold
    if not term_idf:
        for t in query_terms:
            pl = postings_by_term.get(t)
            if pl is None:
                continue
            df = getattr(pl, "doc_frequency", None)
            if df is None:
                df = len(pl.postings)
            term_idf[t] = bm25_idf(
                num_docs, int(df), clamp_negative=cfg.clamp_negative_idf
            )

    scores: dict[int, float] = {int(d): 0.0 for d in candidate_doc_ids}

    for doc_id in list(scores.keys()):
        dl = int(get_doc_length(doc_id))
        s = 0.0
        for t, idf in term_idf.items():
            pl = postings_by_term.get(t)
            if pl is None:
                continue
            tf = int(pl.term_frequencies.get(doc_id, 0))
            s += bm25_term_contribution(
                tf=tf,
                doc_len=dl,
                avgdl=avgdl,
                idf=idf,
                k1=cfg.k1,
                b=cfg.b,
            )
        scores[doc_id] = s

    return scores
