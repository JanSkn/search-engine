import time
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
from backend.search_engine.scoring.bm25 import bm25_score_docs, BM25Config

logger = get_logger(__name__)


class QueryEngine:
    def __init__(self, q: str) -> None:
        self._query = q
        self.inverted_index = get_index()
        self.corrector = get_spell_corrector()

    def _positional_phrase_search(self, terms: list[str]) -> PostingList:
        start = time.perf_counter()
        logger.debug(f"Performing phrase search for: {terms}")

        if not terms:
            return PostingList(postings=[], term_frequencies={}, positions={})

        result = self.inverted_index.index.get(terms[0])

        if result is None:
            return PostingList(postings=[], term_frequencies={}, positions={})

        # for each subsequent term, check positions
        for i, term in enumerate(terms[1:], start=1):
            next_pl = self.inverted_index.index.get(term)

            if next_pl is None:
                return PostingList(postings=[], term_frequencies={}, positions={})

            start_positional_intersect = time.perf_counter()
            result = positional_intersect(result, next_pl, distance=i)
            logger.debug(
                f"Positional intersect for term '{term}' "
                f"with distance={i} completed in "
                f"{time.perf_counter() - start_positional_intersect:.6f}s"
            )
            if len(result.postings) == 0:
                break

        end = time.perf_counter()
        logger.debug(
            f"Result docs: {len(result.postings)}, "
            f"Execution time: {end - start:.6f} seconds"
        )

        return result

    def _bool_search(self, node: Node | None) -> PostingList:
        start = time.perf_counter()
        logger.debug(f"Evaluating node: {getattr(node, 'value', None)}")

        if node is None:
            return PostingList(postings=[], term_frequencies={}, positions={})

        if node.value not in AND | OR | NOT:
            pl = self.inverted_index.index.get(node.value)
            result = pl or PostingList(postings=[], term_frequencies={}, positions={})

        elif node.value in AND:
            # check if one of the nodes has NOT child
            left_is_not = node.left and node.left.value in NOT
            right_is_not = node.right and node.right.value in NOT

            if left_is_not:
                # NOT A AND B -> B minus A
                not_docs = self._bool_search(node.left.right if node.left else None)
                right = self._bool_search(node.right)
                result = find_docs(right, not_docs, "NOT")

            elif right_is_not:
                # A AND NOT B -> A minus B
                left = self._bool_search(node.left)
                not_docs = self._bool_search(node.right.right if node.right else None)
                result = find_docs(left, not_docs, "NOT")

            else:
                # regular AND without NOT
                left = self._bool_search(node.left)
                right = self._bool_search(node.right)
                result = find_docs(left, right, "AND")

        else:  # OR
            left = self._bool_search(node.left)
            right = self._bool_search(node.right)
            result = find_docs(left, right, "OR")

        end = time.perf_counter()
        logger.debug(
            f"Node={node.value!r}, Result docs={len(result.postings)}, "
            f"Execution time: {end - start:.6f} seconds"
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

    def search_results(self, limit: int = 10) -> SearchResults:
        start = time.perf_counter()
        logger.debug("Starting query execution")

        qt = QueryTree()
        normalized_tokens = normalize_search_query(self._query)
        logger.debug(f"Normalized search query: {normalized_tokens}")

        raw_query = self._query.strip()

        correction = repl(self.corrector, raw_query)

        if not qt._has_operators(normalized_tokens):
            self.inverted_index.doc_store.query_terms = list(
                set(normalized_tokens)
            )  # needed for snippets
            if (raw_query.startswith('"') and raw_query.endswith('"')) or (
                raw_query.startswith("'") and raw_query.endswith("'")
            ):
                # positional phrase search
                logger.debug("Executing positional phrase query search...")
                normalized_tokens_no_quots = normalize_search_query(raw_query[1:-1])
                result = self._positional_phrase_search(normalized_tokens_no_quots)
            else:
                # any order -> create AND query
                logger.debug("Executing phrase query search...")
                and_query = QueryEngine._to_boolean_normalized_query(normalized_tokens)
                logger.debug(f"Converted to AND query: {and_query}")
                qt.parse_query(and_query)
                logger.debug(f"Query tree: {qt.root}")
                result = self._bool_search(qt.root)
        else:
            logger.debug("Executing bool query search...")
            try:
                qt.parse_query(normalized_tokens)
                self.inverted_index.doc_store.query_terms = (
                    qt.unique_terms
                )  # needed for snippets
                logger.debug(f"Query tree: {qt.root}")
                result = self._bool_search(qt.root)
            except InvalidOperatorError as e:
                logger.error(f"Invalid query syntax: {e}")
                raise

        if result is None or len(result.postings) == 0:
            return SearchResults(search_results=[], correction=correction)

        logger.debug(
            f"Found {len(result.postings)} results in {time.perf_counter() - start:.6f} seconds"
        )

        # candidates: bool/phrase search returns doc_ids in result.postings
        candidate_doc_ids = result.postings

        # score which query terms
        # - bool queries: qt.unique_terms
        # - AND-auto-query w/o operators: normalized_tokens
        query_terms: list[str] = (
            list(qt.unique_terms)
            if getattr(qt, "unique_terms", None)
            else list(dict.fromkeys(normalized_tokens))
        )

        metadata = self.inverted_index.metadata
        scores = bm25_score_docs(
            query_terms=query_terms,
            postings_by_term=self.inverted_index.index,  # term -> PostingList
            candidate_doc_ids=candidate_doc_ids,
            num_docs=metadata.num_docs,
            avgdl=metadata.avg_doc_length,
            get_doc_length=metadata.get_doc_length,
            cfg=BM25Config(k1=1.2, b=0.75, idf_threshold=0.0, clamp_negative_idf=True),
        )

        # sort doc_ids acc to score
        ranked = sorted(
            ((doc_id, scores.get(doc_id, 0.0)) for doc_id in candidate_doc_ids),
            key=lambda x: x[1],
            reverse=True,
        )

        search_results = []
        for doc_id, score in ranked[:limit]:
            doc_data = self.inverted_index.doc_store.get(doc_id)
            if doc_data is None:
                continue

            url = doc_data.url
            if url is None:
                continue
            title = doc_data.title or "Untitled"
            snippet = doc_data.snippet

            try:
                search_result = SearchResult(
                    document_id=doc_id,
                    url=url,  # type: ignore[arg-type]
                    title=title,
                    snippet=snippet,
                    score=score,
                )
                search_results.append(search_result)
            except Exception as e:
                logger.error(f"Error creating SearchResult for doc_id {doc_id}: {e}")
                continue

        end = time.perf_counter()
        logger.debug(
            f"Returned {len(search_results)} results. "
            f"Total execution time: {end - start:.6f} seconds"
        )
        # clear cache to free memory
        self.inverted_index.clear_cache()

        return SearchResults(search_results=search_results, correction=correction)
