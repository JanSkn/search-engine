from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence, Mapping

# your C++ bindings module name might be: cpp_utils or similar.
# In your code snippet: "from cpp_utils import PostingList"
# and normalize_search_query is exported from _core.
# Adjust imports if needed.
from cpp_utils import normalize_search_query, PostingList  # type: ignore

from backend.search_engine.scoring.bm25 import bm25_score_docs_fielded, BM25Config


def parse_query_terms(query: str) -> list[str]:
    # consistent with your engine (stemming + keep operators)
    terms = normalize_search_query(query)
    # for LTR features we do NOT want boolean operators as terms
    # (they otherwise distort match counts)
    drop = {"AND", "OR", "NOT", "&", "|", "-", "(", ")"}
    return [t for t in terms if t not in drop]


def build_postings_by_term(inverted_index, query_terms: Sequence[str]) -> dict[str, PostingList]:
    postings: dict[str, PostingList] = {}
    for t in query_terms:
        pl = inverted_index.index.get(t)
        if pl is None:
            continue
        postings[t] = pl
    return postings


def matched_terms_count(doc_id: int, postings_by_term: Mapping[str, PostingList]) -> int:
    # how many query terms appear in doc (binary per term)
    c = 0
    for t, pl in postings_by_term.items():
        tf = pl.term_frequencies.get(doc_id, 0)
        if tf and tf > 0:
            c += 1
    return c


def phrase_match_indicator(doc_id: int, query_terms: Sequence[str], postings_by_term: Mapping[str, PostingList]) -> int:
    """
    Phrase match using positional postings:
    For terms t1 t2 ... tn, check existence of positions p, p+1, ..., p+n-1.
    We do this via set-intersection shifting positions.
    """
    if not query_terms:
        return 0

    # all terms must exist in postings
    positions_lists: list[list[int]] = []
    for t in query_terms:
        pl = postings_by_term.get(t)
        if pl is None:
            return 0
        pos = pl.positions.get(doc_id)
        if not pos:
            return 0
        positions_lists.append([int(x) for x in pos])

    # fast set-based progressive narrowing
    base = set(positions_lists[0])  # candidate start positions of first term
    for i in range(1, len(positions_lists)):
        shifted = {p - i for p in positions_lists[i]}  # positions where phrase could start
        base &= shifted
        if not base:
            return 0
    return 1


@lru_cache(maxsize=200_000)
def _cached_title_terms(inverted_index, doc_id: int) -> tuple[str, ...]:
    title = inverted_index.doc_store.get_title_only(int(doc_id))
    if not title:
        return tuple()
    terms = normalize_search_query(title)
    drop = {"AND", "OR", "NOT", "&", "|", "-", "(", ")"}
    return tuple(t for t in terms if t not in drop)


def in_title_indicator(inverted_index, doc_id: int, query_terms: Sequence[str]) -> int:
    title_terms = set(_cached_title_terms(inverted_index, int(doc_id)))
    for t in query_terms:
        if t in title_terms:
            return 1
    return 0


def title_tf(inverted_index, doc_id: int, term: str) -> int:
    # used by BM25 scorer; based on cached title terms
    return int(_cached_title_terms(inverted_index, int(doc_id)).count(term))


@dataclass(frozen=True)
class FeatureVector:
    bm25_body: float
    matched_terms: int
    matched_frac: float
    phrase_match: int
    in_title: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "bm25_body": self.bm25_body,
            "matched_terms": self.matched_terms,
            "matched_frac": self.matched_frac,
            "phrase_match": self.phrase_match,
            "in_title": self.in_title,
        }


def compute_features_for_doc(
    inverted_index,
    *,
    doc_id: int,
    query_terms: Sequence[str],
    postings_by_term: Mapping[str, PostingList],
) -> FeatureVector:
    # BM25: compute body-only by setting title boost=0
    cfg = BM25Config(
        boost_title=0.0,
        boost_body=1.0,
        b_title=0.0,            # irrelevant since boost_title=0
        b_body=BM25Config().b_body,
        k1=BM25Config().k1,
        idf_threshold=BM25Config().idf_threshold,
        clamp_negative_idf=BM25Config().clamp_negative_idf,
        min_terms_after_threshold=BM25Config().min_terms_after_threshold,
    )

    scores = bm25_score_docs_fielded(
        list(query_terms),
        postings_by_term=postings_by_term,
        candidate_doc_ids=[int(doc_id)],
        num_docs=int(inverted_index.metadata.num_docs),
        avg_title_len=float(inverted_index.metadata.avg_title_length),
        avg_body_len=float(inverted_index.metadata.avg_body_length),
        get_title_len=lambda d: int(inverted_index.metadata.get_title_length(int(d))),
        get_body_len=lambda d: int(inverted_index.metadata.get_body_length(int(d))),
        get_title_tf=lambda d, t: int(title_tf(inverted_index, int(d), str(t))),
        cfg=cfg,
    )
    bm25_body = float(scores.get(int(doc_id), 0.0))

    mt = matched_terms_count(int(doc_id), postings_by_term)
    qlen = max(1, len(set(query_terms)))
    mf = float(mt) / float(qlen)

    pm = phrase_match_indicator(int(doc_id), list(query_terms), postings_by_term)
    it = in_title_indicator(inverted_index, int(doc_id), list(query_terms))

    return FeatureVector(
        bm25_body=bm25_body,
        matched_terms=mt,
        matched_frac=mf,
        phrase_match=pm,
        in_title=it,
    )