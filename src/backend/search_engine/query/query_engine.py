from __future__ import annotations

import heapq
import time

from cpp_utils import (  # type: ignore [import-untyped]
    PostingList,
    normalize_search_query,
)

from backend.logging_config import get_logger
from backend.search_engine.error_handling import InvalidOperatorError
from backend.search_engine.index.index_loader import get_index
from backend.search_engine.models.index import SearchResult, SearchResults
from backend.search_engine.query.query_preprocessing import (
    AND,
    NOT,
    OR,
    Node,
    QueryTree,
)
from backend.search_engine.scoring.bm25 import (
    BM25Config,
    RetrievalConfig,
    bm25_idf,
    stop_words,
)
from backend.search_engine.semantic_search.query_embeddings import SemanticSearcher
from backend.search_engine.spell_correction.spell_correction import repl
from backend.search_engine.spell_correction.spell_corrector import get_spell_corrector

try:
    from backend.search_engine.ltr import reranker as _ltr_module

    _LTR_AVAILABLE = True
except ImportError:
    _ltr_module = None  # type: ignore[assignment]
    _LTR_AVAILABLE = False

logger = get_logger(__name__)


# TODO: wild mix of query_terms in index for snippets


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
            allow_fallback_full_retrieval=False,
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

        self.semantic_searcher = SemanticSearcher()

        # tiny per-query caches (avoid repeated df/idf computation)
        self._df_cache: dict[str, int] = {}
        self._idf_cache: dict[str, float] = {}

    @staticmethod
    def _empty_pl() -> PostingList:
        return PostingList(postings=[], term_frequencies={}, positions={})

    @staticmethod
    def _filter_posting_list(pl: PostingList | None, allowed: set[int]) -> PostingList:
        if pl is None:
            return QueryEngine._empty_pl()

        filtered_postings = [int(d) for d in pl.postings if int(d) in allowed]

        tf_src = (
            pl.term_frequencies
            if getattr(pl, "term_frequencies", None) is not None
            else {}
        )
        filtered_tf = {
            int(d): int(tf_src[d]) for d in tf_src.keys() if int(d) in allowed
        }

        pos_src = getattr(pl, "positions", None)
        if pos_src:
            filtered_pos = {
                int(d): pos_src[d] for d in pos_src.keys() if int(d) in allowed
            }
        else:
            filtered_pos = {}

        return PostingList(
            postings=filtered_postings,
            term_frequencies=filtered_tf,
            positions=filtered_pos,
        )

    @staticmethod
    def _filter_posting_list_postings_only(
        pl: PostingList | None, allowed: set[int]
    ) -> PostingList:
        if pl is None:
            return QueryEngine._empty_pl()
        filtered_postings = [int(d) for d in pl.postings if int(d) in allowed]
        return PostingList(
            postings=filtered_postings, term_frequencies={}, positions={}
        )

    @staticmethod
    def _bool_op_postings(
        left: PostingList, right: PostingList, op: str
    ) -> PostingList:
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

    # DF / IDF caching
    def _df_for_term(self, term: str) -> int:
        cached = self._df_cache.get(term)
        if cached is not None:
            return cached

        df = self.inverted_index.get_docfreq(term)
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
        idf = bm25_idf(
            int(md.num_docs), int(df), clamp_negative=self.bm25_cfg.clamp_negative_idf
        )
        self._idf_cache[term] = float(idf)
        return float(idf)

    # candidate selection
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
            kept = [
                t
                for (t, _) in term_idf[
                    : max(1, int(self.retr_cfg.min_terms_after_threshold))
                ]
            ]

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

    def _bool_search(
        self, node: Node | None, allowed: set[int], cand_list: list[int]
    ) -> PostingList:
        return self.inverted_index.bool_search(node, cand_list)

    def _positional_phrase_search(
        self, terms: list[str], allowed: set[int]
    ) -> PostingList:
        start = time.perf_counter()
        logger.debug(f"Performing phrase search for: {terms}")

        result = self.inverted_index.positional_phrase_search(terms, allowed)

        logger.debug(
            f"Result docs: {len(result.postings)}, "
            f"Execution time: {time.perf_counter() - start:.6f} seconds"
        )
        return result

    @staticmethod
    def _to_boolean_normalized_query(tokens: list[str]) -> list[str]:
        if not tokens:
            return []
        query_str = tokens[0]
        for term in tokens[1:]:
            query_str = f"({query_str} AND {term})"
        return normalize_search_query(query_str)

    @staticmethod
    def _reciprocal_rank_fusion(
        lists: list[list[tuple[int, float]]], top_n: int, k: int
    ) -> list[tuple[int, float]]:
        """
        Fuse multiple ranked lists using Reciprocal Rank Fusion (RRF).

        Returns:
            List of top_n (doc_id, combined_score) sorted by RRF score
        """
        rrf_scores: dict[int, float] = {}

        for ranked_list in lists:
            for rank, (doc_id, _) in enumerate(ranked_list):
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1 / (k + rank + 1)

        return heapq.nlargest(top_n, rrf_scores.items(), key=lambda x: x[1])

    def search_results(self, limit: int = 10) -> SearchResults:
        start = time.perf_counter()
        logger.debug("Starting query execution...")

        qt = QueryTree()

        normalized_tokens = normalize_search_query(self._query)
        logger.debug(f"Normalized search query: {normalized_tokens}")

        raw_query = self._query.strip()

        t_corr = time.perf_counter()
        correction = repl(self.corrector, raw_query)
        logger.debug(
            f"Spell correction total time: {time.perf_counter() - t_corr:.6f}s"
        )

        base_terms = [t for t in normalized_tokens if t not in (AND | OR | NOT)]

        is_quoted_phrase = (raw_query.startswith('"') and raw_query.endswith('"')) or (
            raw_query.startswith("'") and raw_query.endswith("'")
        )

        has_ops = qt._has_operators(normalized_tokens)

        if has_ops:
            try:
                qt.parse_query(normalized_tokens)
            except InvalidOperatorError as e:
                logger.error(f"Invalid query syntax: {e}")
                raise

        unique_terms = (
            list(qt.unique_terms) if has_ops else list(dict.fromkeys(base_terms))
        )
        query_terms = [t for t in unique_terms if t not in stop_words]
        self.inverted_index.doc_store.query_terms = query_terms
        cand_terms = self._select_terms_for_candidates(query_terms)
        candidate_doc_ids = self._build_candidates_from_terms(cand_terms)
        cand_set = set(candidate_doc_ids)

        logger.debug(f"candidate_doc_ids count={len(candidate_doc_ids)}")

        restricted_result: PostingList | None = None

        if is_quoted_phrase:
            logger.debug("Executing positional phrase query search...")
            phrase_terms = normalize_search_query(raw_query[1:-1])
            restricted_result = self._positional_phrase_search(phrase_terms, cand_set)

        elif has_ops:
            logger.debug("Executing bool query search...")
            logger.debug(f"Query tree: {qt.root}")
            restricted_result = self._bool_search(qt.root, cand_set, candidate_doc_ids)

        else:
            logger.debug("Executing phrase query search...")
            and_query = self._to_boolean_normalized_query(query_terms)

            logger.debug(f"Converted to AND query: {and_query}")
            if and_query:
                qt2 = QueryTree()
                qt2.parse_query(and_query)
                logger.debug(f"Query tree: {qt2.root}")

                t_bool = time.perf_counter()
                restricted_result = self._bool_search(
                    qt2.root, cand_set, candidate_doc_ids
                )
                logger.debug(f"Bool search time: {time.perf_counter() - t_bool:.6f}s")

        has_boolean_results = (
            restricted_result is not None and len(restricted_result.postings) > 0
        )

        # scoring
        if has_boolean_results:
            logger.debug(
                f"Found {len(restricted_result.postings)} results "
                f"in {time.perf_counter() - start:.6f} seconds"
            )
            t_score = time.perf_counter()

            final_candidate_doc_ids = restricted_result.postings

            cfg = self.bm25_cfg
            scores = self.inverted_index.bm25_score_fielded(
                query_terms=query_terms,
                candidate_doc_ids=final_candidate_doc_ids,
                k1=cfg.k1,
                boost_title=cfg.boost_title,
                boost_body=cfg.boost_body,
                b_title=cfg.b_title,
                b_body=cfg.b_body,
                idf_threshold=cfg.idf_threshold,
                clamp_negative_idf=cfg.clamp_negative_idf,
                min_terms_after_threshold=cfg.min_terms_after_threshold,
            )
            logger.debug(
                f"Fielded BM25 scoring time: {time.perf_counter() - t_score:.6f}s"
            )

            # top-k BM25
            t_sort = time.perf_counter()
            bm25_ranked_top = heapq.nlargest(
                limit,
                (
                    (doc_id, scores.get(int(doc_id), 0.0))
                    for doc_id in final_candidate_doc_ids
                ),
                key=lambda x: x[1],
            )
            logger.debug(f"Ranking sort time: {time.perf_counter() - t_sort:.6f}s")
        else:
            logger.debug(
                "No boolean results found, falling back to semantic search only"
            )
            bm25_ranked_top = []

        # semantic search
        t_semantic = time.perf_counter()
        semantic_scores = self.semantic_searcher.search(raw_query, limit * 2)
        semantic_ranked_top = heapq.nlargest(
            limit,
            semantic_scores,
            key=lambda x: x[1],
        )
        logger.debug(f"Semantic ranking time: {time.perf_counter() - t_semantic:.6f}s")

        # RRF
        ranked_lists = [semantic_ranked_top]
        if bm25_ranked_top:
            ranked_lists.append(bm25_ranked_top)

        final_top = QueryEngine._reciprocal_rank_fusion(
            lists=ranked_lists, top_n=limit, k=60
        )

        # if _LTR_AVAILABLE and _ltr_module is not None:
        #     try:
        #         t_ltr = time.perf_counter()
        #         reranker = _ltr_module.get_reranker()
        #         final_top = reranker.rerank(final_top, raw_query, self.inverted_index)
        #         logger.debug(f"LTR rerank time: {time.perf_counter() - t_ltr:.6f}s")
        #     except Exception as e:
        #         logger.warning(
        #             f"LTR reranking failed, using BM25 + semantic ranking: {e}"
        #         )

        t_top = time.perf_counter()
        search_results: list[SearchResult] = []

        for doc_id, rrf_score in final_top:
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
                        rrf_score=rrf_score,
                    )
                )
            except Exception as e:
                logger.error(f"Error creating SearchResult for doc_id {doc_id}: {e}")

        logger.debug(
            f"Build top-{limit} results time: {time.perf_counter() - t_top:.6f}s"
        )

        end = time.perf_counter()
        logger.debug(
            f"Returned {len(search_results)} results. "
            f"Total execution time: {end - start:.6f} seconds"
        )
        return SearchResults(search_results=search_results, correction=correction)
