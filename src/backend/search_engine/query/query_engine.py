from __future__ import annotations

import time
import heapq
from dataclasses import dataclass
from typing import Iterable
from collections import Counter

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
    bm25_score_docs_fielded,
    BM25Config,
    bm25_idf,
)

logger = get_logger(__name__)

# keep your stopword list as-is (fast set membership)
stop_words = {
    "the", "and", "to", "of", "a", "in", "is", "it", "you", "that",
    "he", "was", "for", "on", "are", "with", "as", "i", "his", "they",
    "be", "at", "one", "have", "this", "from", "or", "had", "by", "but",
    "not", "what", "all", "were", "we", "when", "your", "can", "said",
    "there", "use", "an", "each", "which", "do", "how", "their", "if",
    "will", "up", "other", "about", "out", "many", "then", "them",
    "these", "so", "some", "her", "would", "make", "like", "him",
    "into", "time", "has", "look", "two", "more", "write", "go", "see",
}


@dataclass(frozen=True)
class RetrievalConfig:
    max_terms_for_candidates: int = 3
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

        # Fielded BM25 config (title emphasis)
        self.bm25_cfg = BM25Config(
            k1=1.2,
            boost_title=2.5,
            boost_body=1.0,
            b_title=0.75,
            b_body=0.75,
            idf_threshold=0.5,
            clamp_negative_idf=False,
            min_terms_after_threshold=1,
        )

        # tiny per-query caches (avoid repeated df/idf computation)
        self._df_cache: dict[str, int] = {}
        self._idf_cache: dict[str, float] = {}

    # -----------------
    # PostingList helpers
    # -----------------
    @staticmethod
    def _empty_pl() -> PostingList:
        return PostingList(postings=[], term_frequencies={}, positions={})

    @staticmethod
    def _filter_posting_list(pl: PostingList | None, allowed: set[int]) -> PostingList:
        if pl is None:
            return QueryEngine._empty_pl()

        filtered_postings = [int(d) for d in pl.postings if int(d) in allowed]

        tf_src = pl.term_frequencies if getattr(pl, "term_frequencies", None) is not None else {}
        filtered_tf = {int(d): int(tf_src[d]) for d in tf_src.keys() if int(d) in allowed}

        pos_src = getattr(pl, "positions", None)
        if pos_src:
            filtered_pos = {int(d): pos_src[d] for d in pos_src.keys() if int(d) in allowed}
        else:
            filtered_pos = {}

        return PostingList(postings=filtered_postings, term_frequencies=filtered_tf, positions=filtered_pos)


######################delete
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

    # -----------------
    # DF / IDF caching
    # -----------------
    def _df_for_term(self, term: str) -> int:
        cached = self._df_cache.get(term)
        if cached is not None:
            return cached

        df = self.inverted_index.get_docfreq(term)  # -> int | None aus C++
        df_i = int(df) if df is not None else 0

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
        idf = bm25_idf(int(md.num_docs), int(df), clamp_negative=self.bm25_cfg.clamp_negative_idf)
        self._idf_cache[term] = float(idf)
        return float(idf)

    # -----------------
    # Candidate selection
    # -----------------
    def _select_terms_for_candidates(self, query_terms: list[str]) -> list[str]:
        term_idf: list[tuple[str, float]] = []
        for t in query_terms:
            if t in AND or t in OR or t in NOT:
                continue
            idf = self._idf_for_term(t)
            if idf > 0.0 or self.retr_cfg.idf_threshold <= 0.0:
                term_idf.append((t, idf))

        term_idf.sort(key=lambda x: x[1], reverse=True)

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

    # -----------------
    # Restricted boolean / phrase
    # -----------------
    def _bool_search_restricted(self, node: Node | None, allowed: set[int], cand_list: list[int]) -> PostingList:
        if node is None:
            return self._empty_pl()

        if node.value not in (AND | OR | NOT):
            pl = self.inverted_index.index.get(node.value)
            if pl is None:
                return self._empty_pl()

            tf_map = pl.term_frequencies  # dict-like: doc_id -> tf
            # candidate-driven: O(|cand_list|) statt O(df(term))
            out = [d for d in cand_list if d in tf_map]
            return PostingList(postings=out, term_frequencies={}, positions={})

        if node.value in AND:
            left_is_not = node.left and node.left.value in NOT
            right_is_not = node.right and node.right.value in NOT

            if left_is_not:
                not_docs = self._bool_search_restricted(node.left.right if node.left else None, allowed, cand_list)
                right = self._bool_search_restricted(node.right, allowed, cand_list)
                return self._bool_op_postings(right, not_docs, "NOT")

            if right_is_not:
                left = self._bool_search_restricted(node.left, allowed, cand_list)
                not_docs = self._bool_search_restricted(node.right.right if node.right else None, allowed, cand_list)
                return self._bool_op_postings(left, not_docs, "NOT")

            left = self._bool_search_restricted(node.left, allowed, cand_list)
            right = self._bool_search_restricted(node.right, allowed, cand_list)
            return self._bool_op_postings(left, right, "AND")

        left = self._bool_search_restricted(node.left, allowed, cand_list)
        right = self._bool_search_restricted(node.right, allowed, cand_list)
        return self._bool_op_postings(left, right, "OR")

    def _positional_phrase_search_restricted(self, terms: list[str], allowed: set[int]) -> PostingList:
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

    # -----------------
    # Full retrieval (fallback only)
    # -----------------
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

    # -----------------
    # Helpers
    # -----------------
    @staticmethod
    def _to_boolean_normalized_query(tokens: list[str]) -> list[str]:
        if not tokens:
            return []
        query_str = tokens[0]
        for term in tokens[1:]:
            query_str = f"({query_str} AND {term})"
        return normalize_search_query(query_str)

    # -----------------
    # Main
    # -----------------
    def search_results(self, limit: int = 10) -> SearchResults:
        start = time.perf_counter()
        logger.debug("Starting query execution (efficient top-k)")

        qt = QueryTree()

        t_norm = time.perf_counter()
        normalized_tokens = normalize_search_query(self._query)
        logger.debug(f"normalize_search_query time: {time.perf_counter() - t_norm:.6f}s")

        raw_query = self._query.strip()

        t_corr = time.perf_counter()
        correction = repl(self.corrector, raw_query)
        logger.debug(f"spell correction total time: {time.perf_counter() - t_corr:.6f}s")

        # determine base terms
        t_base = time.perf_counter()
        base_terms = [t for t in normalized_tokens if t not in (AND | OR | NOT)]
        logger.debug(f"base_terms build time: {time.perf_counter() - t_base:.6f}s")

        t_set_qt = time.perf_counter()
        self.inverted_index.doc_store.query_terms = list(set(base_terms))
        logger.debug(f"set doc_store.query_terms (base) time: {time.perf_counter() - t_set_qt:.6f}s")

        t_flags = time.perf_counter()
        is_quoted_phrase = (raw_query.startswith('"') and raw_query.endswith('"')) or (
            raw_query.startswith("'") and raw_query.endswith("'")
        )
        logger.debug(f"quoted phrase check time: {time.perf_counter() - t_flags:.6f}s")

        t_has_ops = time.perf_counter()
        has_ops = qt._has_operators(normalized_tokens)
        logger.debug(f"_has_operators time: {time.perf_counter() - t_has_ops:.6f}s")

        if has_ops:
            try:
                t_parse_ops = time.perf_counter()
                qt.parse_query(normalized_tokens)
                logger.debug(f"parse_query (ops) time: {time.perf_counter() - t_parse_ops:.6f}s")

                t_set_qt2 = time.perf_counter()
                self.inverted_index.doc_store.query_terms = qt.unique_terms
                logger.debug(f"set doc_store.query_terms (ops) time: {time.perf_counter() - t_set_qt2:.6f}s")
            except InvalidOperatorError as e:
                logger.error(f"Invalid query syntax: {e}")
                raise

        t_qterms = time.perf_counter()
        _query_terms: list[str] = (
            list(qt.unique_terms)
            if (has_ops and getattr(qt, "unique_terms", None))
            else list(dict.fromkeys(base_terms))
        )
        logger.debug(f"_query_terms build time: {time.perf_counter() - t_qterms:.6f}s")

        t_filter_terms = time.perf_counter()
        query_terms = [term for term in _query_terms if term not in stop_words]
        logger.debug(f"stop_words filter time: {time.perf_counter() - t_filter_terms:.6f}s")

        logger.debug(f"query_terms (filtered) count={len(query_terms)} terms={query_terms}")

        # 1) candidate generation
        t_select = time.perf_counter()
        cand_terms = self._select_terms_for_candidates(query_terms)
        logger.debug(f"select_terms_for_candidates time: {time.perf_counter() - t_select:.6f}s terms={cand_terms}")

        t_cand = time.perf_counter()
        candidate_doc_ids = self._build_candidates_from_terms(cand_terms)

        t_set_cand = time.perf_counter()
        cand_set = set(candidate_doc_ids)
        logger.debug(f"cand_set build time: {time.perf_counter() - t_set_cand:.6f}s")
        logger.debug(f"Candidate phase time: {time.perf_counter() - t_cand:.6f}s")

        logger.debug(f"candidate_doc_ids count={len(candidate_doc_ids)}")

        if not candidate_doc_ids and not self.retr_cfg.allow_fallback_full_retrieval:
            logger.debug("No candidates and fallback disabled -> returning empty results")
            return SearchResults(search_results=[], correction=correction)

        # 2) restricted retrieval
        t_filter = time.perf_counter()
        restricted_result: PostingList

        if is_quoted_phrase:
            t_phrase_norm = time.perf_counter()
            phrase_terms = normalize_search_query(raw_query[1:-1])
            logger.debug(f"normalize_search_query (phrase) time: {time.perf_counter() - t_phrase_norm:.6f}s")
            restricted_result = self._positional_phrase_search_restricted(phrase_terms, cand_set)

        elif has_ops:
            restricted_result = self._bool_search_restricted(qt.root, cand_set, candidate_doc_ids)

        else:
            t_and_build = time.perf_counter()
            and_query = self._to_boolean_normalized_query(query_terms)
            logger.debug(f"_to_boolean_normalized_query time: {time.perf_counter() - t_and_build:.6f}s")

            qt2 = QueryTree()
            t_parse_no_ops = time.perf_counter()
            qt2.parse_query(and_query)
            logger.debug(f"parse_query (no ops) time: {time.perf_counter() - t_parse_no_ops:.6f}s")

            t_bool_restricted = time.perf_counter()
            restricted_result = self._bool_search_restricted(qt2.root, cand_set, candidate_doc_ids)
            logger.debug(f"_bool_search_restricted time: {time.perf_counter() - t_bool_restricted:.6f}s")

        logger.debug(f"Restricted filter time: {time.perf_counter() - t_filter:.6f}s")

        # fallback full retrieval
        if (restricted_result is None or len(restricted_result.postings) == 0) and self.retr_cfg.allow_fallback_full_retrieval:
            logger.debug("Restricted phase returned 0 hits; running fallback full retrieval")

            t_fallback = time.perf_counter()
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

            logger.debug(f"fallback retrieval time: {time.perf_counter() - t_fallback:.6f}s")

        if restricted_result is None or len(restricted_result.postings) == 0:
            self.inverted_index.clear_cache()
            return SearchResults(search_results=[], correction=correction)

        # 3) scoring (Fielded BM25)
        t_score = time.perf_counter()
        metadata = self.inverted_index.metadata

        t_final_ids = time.perf_counter()
        final_candidate_doc_ids = list(restricted_result.postings)
        logger.debug(f"final_candidate_doc_ids build time: {time.perf_counter() - t_final_ids:.6f}s")

        # Title TF cache: doc_id -> Counter(term->tf) for only query terms.
        # IMPORTANT: uses get_title_only (no snippet IO).
        qterm_set = set(query_terms)
        title_tf_cache: dict[int, Counter[str]] = {}

        def get_title_tf(doc_id: int, term: str) -> int:
            c = title_tf_cache.get(doc_id)
            if c is None:
                title_opt = self.inverted_index.doc_store.get_title_only(int(doc_id))
                title = title_opt or ""
                toks = normalize_search_query(title)  # stem+tokenize in C++
                c = Counter(t for t in toks if t in qterm_set)
                title_tf_cache[doc_id] = c
            return int(c.get(term, 0))

        scores = bm25_score_docs_fielded(
            query_terms=query_terms,
            postings_by_term=self.inverted_index.index,  # term -> PostingList
            candidate_doc_ids=final_candidate_doc_ids,
            num_docs=int(metadata.num_docs),
            avg_title_len=float(metadata.avg_title_length),
            avg_body_len=float(metadata.avg_body_length),
            get_title_len=metadata.get_title_length,
            get_body_len=metadata.get_body_length,
            get_title_tf=get_title_tf,
            cfg=self.bm25_cfg,
        )
        logger.debug(f"Fielded BM25 scoring time: {time.perf_counter() - t_score:.6f}s")

        # 4) top-k
        t_sort = time.perf_counter()
        ranked_top = heapq.nlargest(
            limit,
            ((doc_id, scores.get(int(doc_id), 0.0)) for doc_id in final_candidate_doc_ids),
            key=lambda x: x[1],
        )
        logger.debug(f"Ranking sort time: {time.perf_counter() - t_sort:.6f}s")

        # 5) build results (doc_store.get triggers snippet ONLY for top-k, that's fine)
        t_top = time.perf_counter()
        search_results: list[SearchResult] = []

        t_docstore = time.perf_counter()
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

        logger.debug(f"doc_store.get + SearchResult build time: {time.perf_counter() - t_docstore:.6f}s")
        logger.debug(f"Build top-{limit} results time: {time.perf_counter() - t_top:.6f}s")

        logger.debug(f"Returned {len(search_results)} results. Total time: {time.perf_counter() - start:.6f}s")

        self.inverted_index.clear_cache()
        return SearchResults(search_results=search_results, correction=correction)