# backend/search_engine/scoring/bm25.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence
from collections.abc import Callable

from cpp_utils import PostingList  # type: ignore [import-untyped]


@dataclass(frozen=True)
class BM25Config:
    k1: float = 1.2  # how strong tf influence is
    b: float = 0.75  # level of document normalization

    # ignore terms with idf < idf_threshold
    idf_threshold: float = 0.0
    # clamp negative idf to 0
    clamp_negative_idf: bool = True

    min_terms_after_threshold: int = 1


def bm25_idf(num_docs: int, df: int, *, clamp_negative: bool = True) -> float:
    """
    idf: ln((N - df + 0.5) / (df + 0.5))
    """
    if num_docs <= 0:
        return 0.0
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
    # materialize candidates once
    cand_list = list(candidate_doc_ids)
    cand_set = set(cand_list)

    # compute idf for al query terms first
    term_idf_all: list[tuple[str, float]] = []  # (term, idf)
    for t in query_terms:
        pl = postings_by_term.get(t)
        if pl is None:
            continue
        df = getattr(pl, "doc_frequency", None)
        if df is None:
            df = len(pl.postings)

        idf = bm25_idf(num_docs, int(df), clamp_negative=cfg.clamp_negative_idf)
        term_idf_all.append((t, idf))

    # order by decreasing idf
    term_idf_all.sort(key=lambda x: x[1], reverse=True)

    # thresholding
    term_idf: dict[str, float] = {
        t: idf for (t, idf) in term_idf_all if idf >= cfg.idf_threshold
    }

    # ensure keep at least n best terms
    min_keep = max(1, int(cfg.min_terms_after_threshold))
    if len(term_idf) < min_keep:
        term_idf = {t: idf for (t, idf) in term_idf_all[:min_keep]}

    # scoring only docs that actually appear in term postings (tf>0)
    scores: dict[int, float] = {int(d): 0.0 for d in cand_list}

    for t, idf in term_idf.items():
        pl = postings_by_term.get(t)
        if pl is None:
            continue

        tf_map = dict(pl.term_frequencies)

        for doc_id, tf in tf_map.items():
            doc_id = int(doc_id)
            if doc_id not in cand_set:
                continue

            dl = int(get_doc_length(doc_id))
            scores[doc_id] += bm25_term_contribution(
                tf=int(tf),
                doc_len=dl,
                avgdl=avgdl,
                idf=idf,
                k1=cfg.k1,
                b=cfg.b,
            )

    return scores
