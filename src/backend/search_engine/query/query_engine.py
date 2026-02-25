from __future__ import annotations

import time
import heapq
from dataclasses import dataclass
from typing import Iterable
from stop_words import get_stop_words

from backend.search_engine.index.index_loader import get_index
from backend.search_engine.models.index import SearchResult, SearchResults
from cpp_utils import (  # type: ignore [import-untyped]
    normalize_search_query,
    positional_intersect,
    find_docs,
    PostingList,
)
from backend.search_engine.query.query_preprocessing import (
    Node,
    QueryTree,
    AND,
    OR,
    NOT,
)
from backend.search_engine.error_handling import InvalidOperatorError
from backend.logging_config import get_logger

from backend.search_engine.spell_correction.spell_corrector import get_spell_corrector
from backend.search_engine.spell_correction.spell_correction import repl
from backend.search_engine.scoring.bm25 import (
    bm25_score_docs,
    BM25Config,
    bm25_idf,
)

logger = get_logger(__name__)
stop_words = {'the', 'and', 'to', 'of', 'a', 'in', 'is', 'it', 'you', 'that', 
                    'he', 'was', 'for', 'on', 'are', 'with', 'as', 'i', 'his', 'they', 
                    'be', 'at', 'one', 'have', 'this', 'from', 'or', 'had', 'by', 'but',
                    'not', 'what', 'all', 'were', 'we', 'when', 'your', 'can', 'said',
                    'there', 'use', 'an', 'each', 'which', 'do', 'how', 'their', 'if',
                    'will', 'up', 'other', 'about', 'out', 'many', 'then', 'them',
                    'these', 'so', 'some', 'her', 'would', 'make', 'like', 'him',
                    'into', 'time', 'has', 'look', 'two', 'more', 'write', 'go', 'see'}


@dataclass(frozen=True)
class RetrievalConfig:
    max_terms_for_candidates: int = 3              # take top-idf terms
    max_candidates_total: int = 50_000
    max_candidates_per_term: int = 30_000

    idf_threshold: float = 0.0
    min_terms_after_threshold: int = 1

    allow_fallback_full_retrieval: bool = True


class QueryEngine:
    def __init__(self, q: str) -> None:
        self._query = q
        self.inverted_index = get_index()
        self.corrector = get_spell_corrector()

        self.retr_cfg = RetrievalConfig(
            max_terms_for_candidates=3,
            max_candidates_total=10_000,
            max_candidates_per_term=10_000,
            idf_threshold=0.5,
            min_terms_after_threshold=1,
            allow_fallback_full_retrieval=True,
        )

        self.bm25_cfg = BM25Config(
            k1=1.2,
            b=0.75,
            idf_threshold=0.5,
            clamp_negative_idf=False,
            min_terms_after_threshold=1,
        )

        self._df_cache: dict[str, int] = {}
        self._idf_cache: dict[str, float] = {}

    @staticmethod
    def _empty_pl() -> PostingList:
        return PostingList(postings=[], term_frequencies={}, positions={})

    @staticmethod
    def _filter_posting_list(pl: PostingList | None, allowed: set[int]) -> PostingList:
        if pl is None:
            return QueryEngine._empty_pl()

        # postings: list[int]
        filtered_postings = [int(d) for d in pl.postings if int(d) in allowed]

        # term_frequencies: mapping doc_id -> tf
        tf_src = dict(pl.term_frequencies) if getattr(pl, "term_frequencies", None) is not None else {}
        filtered_tf = {int(d): int(tf_src[d]) for d in tf_src.keys() if int(d) in allowed}

        # positions: mapping doc_id -> list[int]
        pos_src = getattr(pl, "positions", None)
        if pos_src:
            filtered_pos = {int(d): pos_src[d] for d in pos_src.keys() if int(d) in allowed}
        else:
            filtered_pos = {}

        return PostingList(postings=filtered_postings, term_frequencies=filtered_tf, positions=filtered_pos)

    @staticmethod
    def _filter_posting_list_postings_only(pl: PostingList | None, allowed: set[int]) -> PostingList:
        if pl is None:
            return QueryEngine._empty_pl()
        filtered_postings = [int(d) for d in pl.postings if int(d) in allowed]
        return PostingList(postings=filtered_postings, term_frequencies={}, positions={})

    @staticmethod
    def _bool_op_postings(left: PostingList, right: PostingList, op: str) -> PostingList:
        lset = set(int(d) for d in left.postings)
        rset = set(int(d) for d in right.postings)

        if op == "AND":
            out = lset & rset
        elif op == "OR":
            out = lset | rset
        elif op == "NOT":
            out = lset - rset
        else:
            out = set()

        return PostingList(postings=sorted(out), term_frequencies={}, positions={})

    def _df_for_term(self, term: str) -> int:
        cached = self._df_cache.get(term)
        if cached is not None:
            return cached
        pl = self.inverted_index.index.get(term)
        if pl is None:
            self._df_cache[term] = 0
            return 0
        df = getattr(pl, "doc_frequency", 0)
        df_i = int(df)
        self._df_cache[term] = df_i
        return df_i

    def _idf_for_term(self, term: str) -> float:
        cached = self._idf_cache.get(term)
        if cached is not None:
            return cached

        df = self._df_for_term(term)
        if df <= 0:
            self._idf_cache[term] = 0.0
            return 0.0

        md = self.inverted_index.metadata
        idf = bm25_idf(md.num_docs, int(df), clamp_negative=self.bm25_cfg.clamp_negative_idf)
        self._idf_cache[term] = float(idf)
        return float(idf)

    def _select_terms_for_candidates(self, query_terms: list[str]) -> list[str]:
        # compute (term, idf) for existing terms
        term_idf = []
        for t in query_terms:
            if t in AND or t in OR or t in NOT:
                continue
            idf = self._idf_for_term(t)
            if idf > 0.0 or self.retr_cfg.idf_threshold <= 0.0:
                term_idf.append((t, idf))

        term_idf.sort(key=lambda x: x[1], reverse=True)

        # threshold
        kept = [t for (t, idf) in term_idf if idf >= self.retr_cfg.idf_threshold]
        if len(kept) < max(1, int(self.retr_cfg.min_terms_after_threshold)):
            kept = [t for (t, _) in term_idf[: max(1, int(self.retr_cfg.min_terms_after_threshold))]]

        return kept[: max(1, int(self.retr_cfg.max_terms_for_candidates))]

    def _build_candidates_from_terms(self, terms: list[str]) -> list[int]:
        start = time.perf_counter()

        cand: set[int] = set()
        total_cap = int(self.retr_cfg.max_candidates_total)
        per_term_cap = int(self.retr_cfg.max_candidates_per_term)

        for t in terms:
            pl = self.inverted_index.index.get(t)
            if pl is None:
                continue

            for d in pl.postings[:per_term_cap]:
                cand.add(int(d))
                if len(cand) >= total_cap:
                    break
            if len(cand) >= total_cap:
                break

        logger.debug(
            f"Candidate generation using terms={terms} -> {len(cand)} candidates "
            f"in {time.perf_counter() - start:.6f}s"
        )

        return list(cand)


    def _bool_search_restricted(self, node: Node | None, allowed: set[int]) -> PostingList:
       # every term leaf reduced to `allowed` candidates first -> prevents full boolean retrieval
        if node is None:
            return self._empty_pl()

        if node.value not in AND | OR | NOT:
            pl = self.inverted_index.index.get(node.value)
            return self._filter_posting_list_postings_only(pl, allowed)

        if node.value in AND:
            left_is_not = node.left and node.left.value in NOT
            right_is_not = node.right and node.right.value in NOT

            if left_is_not:
                not_docs = self._bool_search_restricted(node.left.right if node.left else None, allowed)
                right = self._bool_search_restricted(node.right, allowed)
                return self._bool_op_postings(right, not_docs, "NOT")

            if right_is_not:
                left = self._bool_search_restricted(node.left, allowed)
                not_docs = self._bool_search_restricted(node.right.right if node.right else None, allowed)
                return self._bool_op_postings(left, not_docs, "NOT")

            left = self._bool_search_restricted(node.left, allowed)
            right = self._bool_search_restricted(node.right, allowed)
            return self._bool_op_postings(left, right, "AND")

        # OR
        left = self._bool_search_restricted(node.left, allowed)
        right = self._bool_search_restricted(node.right, allowed)
        return self._bool_op_postings(left, right, "OR")

    def _positional_phrase_search_restricted(self, terms: list[str], allowed: set[int]) -> PostingList:
        # phrase saerch only across candidate docs
        if not terms:
            return self._empty_pl()

        first = self._filter_posting_list(self.inverted_index.index.get(terms[0]), allowed)
        if len(first.postings) == 0:
            return self._empty_pl()

        result = first
        for i, term in enumerate(terms[1:], start=1):
            next_pl = self._filter_posting_list(self.inverted_index.index.get(term), allowed)
            if len(next_pl.postings) == 0:
                return self._empty_pl()
            result = positional_intersect(result, next_pl, distance=i)
            if len(result.postings) == 0:
                break

        return result

    # full retrieval (fallback only)
    def _positional_phrase_search_full(self, terms: list[str]) -> PostingList:
        if not terms:
            return self._empty_pl()

        result = self.inverted_index.index.get(terms[0])
        if result is None:
            return self._empty_pl()

        for i, term in enumerate(terms[1:], start=1):
            next_pl = self.inverted_index.index.get(term)
            if next_pl is None:
                return self._empty_pl()
            result = positional_intersect(result, next_pl, distance=i)
            if len(result.postings) == 0:
                break
        return result

    def _bool_search_full(self, node: Node | None) -> PostingList:
        if node is None:
            return self._empty_pl()

        if node.value not in AND | OR | NOT:
            pl = self.inverted_index.index.get(node.value)
            return pl or self._empty_pl()

        if node.value in AND:
            left_is_not = node.left and node.left.value in NOT
            right_is_not = node.right and node.right.value in NOT

            if left_is_not:
                not_docs = self._bool_search_full(node.left.right if node.left else None)
                right = self._bool_search_full(node.right)
                return find_docs(right, not_docs, "NOT")

            if right_is_not:
                left = self._bool_search_full(node.left)
                not_docs = self._bool_search_full(node.right.right if node.right else None)
                return find_docs(left, not_docs, "NOT")

            left = self._bool_search_full(node.left)
            right = self._bool_search_full(node.right)
            return find_docs(left, right, "AND")

        left = self._bool_search_full(node.left)
        right = self._bool_search_full(node.right)
        return find_docs(left, right, "OR")


    @staticmethod
    def _to_boolean_normalized_query(tokens: list[str]) -> list[str]:
        if not tokens:
            return []
        query_str = tokens[0]
        for term in tokens[1:]:
            query_str = f"({query_str} AND {term})"
        return normalize_search_query(query_str)


    def search_results(self, limit: int = 10) -> SearchResults:
        start = time.perf_counter()
        logger.debug("Starting query execution (efficient top-k)")

        qt = QueryTree()
        normalized_tokens = normalize_search_query(self._query)
        raw_query = self._query.strip()

        correction = repl(self.corrector, raw_query)

        # determine query terms for scoring / candidates
        base_terms = [t for t in normalized_tokens if t not in (AND | OR | NOT)]

        # snippets need query terms
        self.inverted_index.doc_store.query_terms = list(set(base_terms))

        is_quoted_phrase = (raw_query.startswith('"') and raw_query.endswith('"')) or (
            raw_query.startswith("'") and raw_query.endswith("'")
        )

        has_ops = qt._has_operators(normalized_tokens)

        # f operators are there -> parse to get qt.unique_terms
        if has_ops:
            try:
                qt.parse_query(normalized_tokens)
                self.inverted_index.doc_store.query_terms = qt.unique_terms
            except InvalidOperatorError as e:
                logger.error(f"Invalid query syntax: {e}")
                raise

        # decide query terms for scoring/candidates
        _query_terms: list[str] = (
            list(qt.unique_terms)
            if (has_ops and getattr(qt, "unique_terms", None))
            else list(dict.fromkeys(base_terms))
        )
        query_terms = [term for term in _query_terms if term not in stop_words]

        print("IDFs:")  # DEBUG
        for t in query_terms:  # DEBUG
            print(t, self._idf_for_term(t))  # DEBUG

        
        for t in query_terms:
            pl = self.inverted_index.index.get(t)
            if pl is None:
                print("TERM", t, "-> pl=None")
                continue
            df_attr = getattr(pl, "doc_frequency", None)
            try:
                postings_len = len(pl.postings)
            except Exception as e:
                postings_len = f"len-error:{e}"
            print("TERM", t, "df_attr", df_attr, "len(postings)", postings_len)


        t_cand = time.perf_counter()
        cand_terms = self._select_terms_for_candidates(query_terms)
        candidate_doc_ids = self._build_candidates_from_terms(cand_terms)
        cand_set = set(candidate_doc_ids)
        logger.debug(f"Candidate phase time: {time.perf_counter() - t_cand:.6f}s")

        # if no candidates at all -> exit or fallback
        if not candidate_doc_ids and not self.retr_cfg.allow_fallback_full_retrieval:
            return SearchResults(search_results=[], correction=correction)


        t_filter = time.perf_counter()
        restricted_result: PostingList

        if is_quoted_phrase:
            phrase_terms = normalize_search_query(raw_query[1:-1])
            restricted_result = self._positional_phrase_search_restricted(phrase_terms, cand_set)
        elif has_ops:
            restricted_result = self._bool_search_restricted(qt.root, cand_set)
        else:
            # no operators: dont build full AND boolean over all docs
            and_query = self._to_boolean_normalized_query(query_terms)
            qt2 = QueryTree()
            qt2.parse_query(and_query)
            restricted_result = self._bool_search_restricted(qt2.root, cand_set)

        logger.debug(f"Restricted filter time: {time.perf_counter() - t_filter:.6f}s")

        logger.debug(f"restricted pre-fallback hits={len(restricted_result.postings)}")
        # if too few hits -> fall back to full retrieval
        if (restricted_result is None or len(restricted_result.postings) == 0) and self.retr_cfg.allow_fallback_full_retrieval:
            logger.debug("Restricted phase returned 0 hits; running fallback full retrieval")

            if is_quoted_phrase:
                phrase_terms = normalize_search_query(raw_query[1:-1])
                restricted_result = self._positional_phrase_search_full(phrase_terms)
            elif has_ops:
                restricted_result = self._bool_search_full(qt.root)
            else:
                and_query = self._to_boolean_normalized_query(query_terms)
                qt3 = QueryTree()
                qt3.parse_query(and_query)
                restricted_result = self._bool_search_full(qt3.root)
        
        logger.debug(f"restricted post-fallback hits={len(restricted_result.postings)}")

        if restricted_result is None or len(restricted_result.postings) == 0:
            self.inverted_index.clear_cache()
            return SearchResults(search_results=[], correction=correction)

        t_score = time.perf_counter()
        metadata = self.inverted_index.metadata

        final_candidate_doc_ids = list(restricted_result.postings)

        scores = bm25_score_docs(
            query_terms=query_terms,
            postings_by_term=self.inverted_index.index,
            candidate_doc_ids=final_candidate_doc_ids,
            num_docs=metadata.num_docs,
            avgdl=metadata.avg_doc_length,
            get_doc_length=metadata.get_doc_length,
            cfg=self.bm25_cfg,
        )
        logger.debug(f"BM25 scoring time: {time.perf_counter() - t_score:.6f}s")


        # top k selection
        t_sort = time.perf_counter()
        ranked_top = heapq.nlargest(
            limit,
            ((doc_id, scores.get(int(doc_id), 0.0)) for doc_id in final_candidate_doc_ids),
            key=lambda x: x[1],
        )
        logger.debug(f"Ranking sort time: {time.perf_counter() - t_sort:.6f}s")

        t_top = time.perf_counter()
        search_results: list[SearchResult] = []

        for doc_id, score in ranked_top:
            doc_id = int(doc_id)
            doc_data = self.inverted_index.doc_store.get(doc_id)
            if doc_data is None:
                continue

            url = doc_data.url
            if url is None:
                continue

            title = doc_data.title or "Untitled"
            snippet = doc_data.snippet

            try:
                search_results.append(
                    SearchResult(
                        document_id=doc_id,
                        url=url,  # type: ignore[arg-type]
                        title=title,
                        snippet=snippet,
                        score=float(score),
                    )
                )
            except Exception as e:
                logger.error(f"Error creating SearchResult for doc_id {doc_id}: {e}")

        logger.debug(f"Build top-{limit} results time: {time.perf_counter() - t_top:.6f}s")

        logger.debug(
            f"Returned {len(search_results)} results. Total time: {time.perf_counter() - start:.6f}s"
        )

        self.inverted_index.clear_cache()
        return SearchResults(search_results=search_results, correction=correction)