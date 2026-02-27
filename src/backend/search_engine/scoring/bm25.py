from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence
from collections.abc import Callable

from cpp_utils import PostingList  # type: ignore [import-untyped]


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


def _field_norm_tf(tf: int, *, field_len: int, avg_field_len: float, b_f: float) -> float:
    """
    Field-normalized TF:
      tf / (1 - b_f + b_f * len_f(d) / avglen_f)
    """
    if tf <= 0:
        return 0.0
    if avg_field_len <= 0.0:
        avg_field_len = 1.0
    if field_len <= 0:
        field_len = 1

    denom = (1.0 - b_f) + b_f * (float(field_len) / float(avg_field_len))
    if denom <= 0.0:
        return float(tf)
    return float(tf) / denom


def bm25_score_docs_fielded(
    query_terms: Sequence[str],
    *,
    postings_by_term: Mapping[str, PostingList],
    candidate_doc_ids: Iterable[int],
    num_docs: int,
    avg_title_len: float,
    avg_body_len: float,
    get_title_len: Callable[[int], int],
    get_body_len: Callable[[int], int],
    get_title_tf: Callable[[int, str], int],
    cfg: BM25Config = BM25Config(),
) -> dict[int, float]:
    cand_list = [int(d) for d in candidate_doc_ids]

    # --- idf compute once ---
    term_idf_all: list[tuple[str, float]] = []
    for t in query_terms:
        pl = postings_by_term.get(t)
        if pl is None:
            continue
        df = getattr(pl, "doc_frequency", None)
        if df is None:
            df = len(pl.postings)
        term_idf_all.append((t, float(bm25_idf(num_docs, int(df), clamp_negative=cfg.clamp_negative_idf))))

    term_idf_all.sort(key=lambda x: x[1], reverse=True)

    term_idf: dict[str, float] = {t: idf for (t, idf) in term_idf_all if idf >= cfg.idf_threshold}
    min_keep = max(1, int(cfg.min_terms_after_threshold))
    if len(term_idf) < min_keep:
        term_idf = {t: idf for (t, idf) in term_idf_all[:min_keep]}

    scores: dict[int, float] = {d: 0.0 for d in cand_list}

    # --- length caches ---
    title_len_cache: dict[int, int] = {}
    body_len_cache: dict[int, int] = {}

    def _tlen(d: int) -> int:
        v = title_len_cache.get(d)
        if v is None:
            v = int(get_title_len(d))
            title_len_cache[d] = v
        return v

    def _blen(d: int) -> int:
        v = body_len_cache.get(d)
        if v is None:
            v = int(get_body_len(d))
            body_len_cache[d] = v
        return v

    k1 = float(cfg.k1)
    k1p1 = k1 + 1.0

    # --- iterate candidates (FAST) ---
    for t, idf in term_idf.items():
        pl = postings_by_term.get(t)
        if pl is None:
            continue

        tf_body_map = pl.term_frequencies  # python mapping already
        for d in cand_list:
            tf_body = int(tf_body_map.get(d, 0))
            if tf_body <= 0:
                # candidates are generated from body postings, so in practice tf_body>0
                # for the generating term; for OR-queries it may be 0 -> skip
                continue

            # title tf only computed for docs we actually score
            tf_title = int(get_title_tf(d, t))

            tf_norm = (
                cfg.boost_title
                * _field_norm_tf(tf_title, field_len=_tlen(d), avg_field_len=avg_title_len, b_f=cfg.b_title)
                + cfg.boost_body
                * _field_norm_tf(tf_body, field_len=_blen(d), avg_field_len=avg_body_len, b_f=cfg.b_body)
            )
            if tf_norm <= 0.0:
                continue

            scores[d] += float(idf) * (tf_norm * k1p1 / (tf_norm + k1))

    return scores