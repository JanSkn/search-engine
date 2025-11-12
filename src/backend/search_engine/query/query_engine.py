from typing import Literal
import numpy as np
from backend.search_engine.models.index import PostingList, SearchResult
from backend.search_engine.indexer.index_builder import lemmatize_search_query
from backend.search_engine.index.inverted_index import InvertedIndex
from backend.search_engine.query.query_preprocessing import (
    Node,
    QueryTree,
    AND,
    OR,
    NOT,
)
from backend.search_engine.error_handling import InvalidOperatorError
from backend.logging_config import get_logger

logger = get_logger(__name__)

inverted_index = InvertedIndex()


# TODO STILL NOT FILTERING STOP WORDS IN QUERY AND INDEX, make sure to do for both otherwise errors
# TODO depending on postings size directly sorting list in place faster than conversion to numpy?
class QueryEngine:
    def __init__(self, q: str) -> None:
        self._query = q

    def _normalized_query(self) -> list[str]:
        return lemmatize_search_query(self._query)

    @staticmethod
    def _positional_intersect(
        posting_list_1: PostingList, posting_list_2: PostingList, distance: int = 1
    ) -> PostingList:
        # term2 appears 'distance' positions after term1.
        doc_ids = []
        result_tf = {}
        result_pos = {}

        postings_1 = posting_list_1.postings
        postings_2 = posting_list_2.postings
        pos_1 = posting_list_1.positions
        pos_2 = posting_list_2.positions
        skip_1 = posting_list_1.skip_pointers
        skip_2 = posting_list_2.skip_pointers
        len_1, len_2 = len(postings_1), len(postings_2)

        i = j = 0

        while i < len_1 and j < len_2:
            if postings_1[i] == postings_2[j]:
                doc_id = postings_1[i]

                # Check if positions match (term2 at position+distance from term1)
                if doc_id in pos_1 and doc_id in pos_2:
                    positions_1 = pos_1[doc_id]
                    positions_2 = pos_2[doc_id]

                    valid_positions = []
                    for pos_a in positions_1:
                        # Check if term2 appears at pos_a + distance
                        # TODO positions_2 to set for faster lookup?
                        if (pos_a + distance) in positions_2:
                            valid_positions.append(pos_a)

                    if valid_positions:
                        doc_ids.append(doc_id)
                        result_tf[doc_id] = len(valid_positions)
                        result_pos[doc_id] = np.array(valid_positions)

                i += 1
                j += 1
            elif postings_1[i] < postings_2[j]:
                if i in skip_1 and postings_1[skip_1[i]] <= postings_2[j]:
                    i = skip_1[i]
                else:
                    i += 1
            else:
                if j in skip_2 and postings_2[skip_2[j]] <= postings_1[i]:
                    j = skip_2[j]
                else:
                    j += 1

        res = PostingList(
            postings=np.array(doc_ids), term_frequencies=result_tf, positions=result_pos
        )
        res.build_skip_pointers()

        return res

    def _positional_phrase_search(self, terms: list[str]) -> PostingList:
        if not terms:
            return PostingList(postings=np.array([]), term_frequencies={}, positions={})

        result = inverted_index.index.get(terms[0])

        if result is None:
            return PostingList(postings=np.array([]), term_frequencies={}, positions={})

        # for each subsequent term, check positions
        for i, term in enumerate(terms[1:], start=1):
            next_pl = inverted_index.index.get(term)

            if next_pl is None:
                return PostingList(
                    postings=np.array([]), term_frequencies={}, positions={}
                )

            result = self._positional_intersect(result, next_pl, distance=i)
            if len(result.postings) == 0:
                break

        return result

    @staticmethod
    def _find_docs(
        posting_list_1: PostingList,
        posting_list_2: PostingList,
        mode: Literal["AND", "OR", "NOT"],
    ) -> PostingList:
        l_1, l_2 = len(posting_list_1.postings), len(posting_list_2.postings)
        postings_1 = posting_list_1.postings
        postings_2 = posting_list_2.postings
        skip_1 = posting_list_1.skip_pointers
        skip_2 = posting_list_2.skip_pointers
        tf_1 = posting_list_1.term_frequencies
        tf_2 = posting_list_2.term_frequencies

        i = j = 0

        if mode == "AND":
            and_doc_ids = []
            result_tf = {}

            while i < l_1 and j < l_2:
                if postings_1[i] == postings_2[j]:
                    doc_id = postings_1[i]
                    and_doc_ids.append(doc_id)

                    result_tf[doc_id] = tf_1.get(doc_id, 0) + tf_2.get(doc_id, 0)

                    i += 1
                    j += 1
                elif postings_1[i] < postings_2[j]:
                    if i in skip_1 and postings_1[skip_1[i]] <= postings_2[j]:
                        i = skip_1[i]
                    else:
                        i += 1
                else:
                    if j in skip_2 and postings_2[skip_2[j]] <= postings_1[i]:
                        j = skip_2[j]
                    else:
                        j += 1

            res = PostingList(
                postings=np.array(and_doc_ids),
                term_frequencies=result_tf,
                positions={},
            )
            res.build_skip_pointers()
            return res
        elif mode == "OR":
            or_doc_ids = np.union1d(postings_1, postings_2)
            result_tf = {}

            for doc_id in or_doc_ids:
                result_tf[doc_id] = tf_1.get(doc_id, 0) + tf_2.get(doc_id, 0)

            res = PostingList(
                postings=or_doc_ids, term_frequencies=result_tf, positions={}
            )
            res.build_skip_pointers()
            return res
        else:  # NOT
            if len(posting_list_1.postings) == 0:
                # no base documents
                return PostingList(
                    postings=np.array([]), term_frequencies={}, positions={}
                )

            result_docs = np.setdiff1d(posting_list_1.postings, posting_list_2.postings)
            result_tf = {
                doc_id: posting_list_1.term_frequencies.get(doc_id, 0)
                for doc_id in result_docs
            }

            res = PostingList(
                postings=result_docs, term_frequencies=result_tf, positions={}
            )
            res.build_skip_pointers()
            return res

    @staticmethod
    def _bool_search(node: Node | None) -> PostingList:
        if node is None:
            return PostingList(postings=np.array([]), term_frequencies={}, positions={})

        if node.value not in AND | OR | NOT:
            pl = inverted_index.index.get(node.value)
            if pl is None:
                return PostingList(
                    postings=np.array([]), term_frequencies={}, positions={}
                )
            return pl

        elif node.value in AND:
            # check if one of the nodes has NOT child
            left_is_not = node.left and node.left.value in NOT
            right_is_not = node.right and node.right.value in NOT

            if left_is_not:
                # NOT A AND B -> B minus A
                not_docs = QueryEngine._bool_search(
                    node.left.right if node.left else None
                )
                right = QueryEngine._bool_search(node.right)
                return QueryEngine._find_docs(right, not_docs, "NOT")

            elif right_is_not:
                # A AND NOT B -> A minus B
                left = QueryEngine._bool_search(node.left)
                not_docs = QueryEngine._bool_search(
                    node.right.right if node.right else None
                )
                return QueryEngine._find_docs(left, not_docs, "NOT")

            else:
                # regular AND without NOT
                left = QueryEngine._bool_search(node.left)
                right = QueryEngine._bool_search(node.right)
                return QueryEngine._find_docs(left, right, "AND")

        else:  # OR
            left = QueryEngine._bool_search(node.left)
            right = QueryEngine._bool_search(node.right)
            return QueryEngine._find_docs(left, right, "OR")

    def search_results(self, limit: int = 10) -> list[SearchResult]:
        qt = QueryTree()
        normalized_tokens = self._normalized_query()
        logger.debug(f"Normalized search query: {normalized_tokens}")

        if not qt._has_operators(normalized_tokens):
            logger.debug("Executing positional phrase query search...")
            posting_lists = self._positional_phrase_search(normalized_tokens)
        else:
            logger.debug("Executing bool query search...")
            try:
                qt.parse_query(normalized_tokens)
                logger.debug(f"Query tree: {qt.root}")
                posting_lists = QueryEngine._bool_search(qt.root)
            except InvalidOperatorError as e:
                logger.error(f"Invalid query syntax: {e}")
                raise

        if posting_lists is None or len(posting_lists.postings) == 0:
            return []

        search_results = []
        for doc_id in posting_lists.postings[:limit]:
            doc_data = inverted_index.doc_store.get(doc_id)
            if doc_data is None:
                continue

            url = doc_data.get("url")
            if url is None:
                continue
            title = doc_data.get("title", "Untitled")

            try:
                search_result = SearchResult(
                    document_id=doc_id,
                    url=url,  # type: ignore[arg-type]
                    title=title,
                )
                search_results.append(search_result)
            except Exception as e:
                logger.error(f"Error creating SearchResult for doc_id {doc_id}: {e}")
                continue

        return search_results
