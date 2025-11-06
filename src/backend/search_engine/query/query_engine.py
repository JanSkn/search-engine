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

inverted_index = InvertedIndex()


class QueryEngine:
    def __init__(self, q: str) -> None:
        self._query = q

    def _normalized_query(self) -> list[str]:
        return lemmatize_search_query(self._query)

    @staticmethod
    def _find_docs(
        posting_list_1: PostingList,
        posting_list_2: PostingList,
        mode: Literal["AND", "OR", "NOT"],
    ) -> PostingList:
        # shorter postings first to reduce comparisons
        l_1, l_2 = len(posting_list_1.postings), len(posting_list_2.postings)
        if l_1 < l_2:
            postings_1 = posting_list_1.postings
            postings_2 = posting_list_2.postings
            skip_pointers_1 = posting_list_1.skip_pointers
            skip_pointers_2 = posting_list_2.skip_pointers
        else:
            postings_1 = posting_list_2.postings
            postings_2 = posting_list_1.postings
            skip_pointers_1 = posting_list_2.skip_pointers
            skip_pointers_2 = posting_list_1.skip_pointers

        i = j = 0

        if mode == "AND":
            doc_ids = []

            while i < l_1 and j < l_2:
                if postings_1[i] == postings_2[j]:
                    doc_ids.append(postings_1[i])
                    i += 1
                    j += 1
                elif postings_1[i] < postings_2[j]:
                    if (
                        i in skip_pointers_1
                        and postings_1[skip_pointers_1[i]] <= postings_2[j]
                    ):
                        i = skip_pointers_1[i]
                    else:
                        i += 1
                else:
                    if (
                        j in skip_pointers_2
                        and postings_2[skip_pointers_2[j]] <= postings_1[i]
                    ):
                        j = skip_pointers_2[j]
                    else:
                        j += 1

            res = PostingList(postings=np.array(doc_ids))
            res.build_skip_pointers()
        elif mode == "OR":
            doc_ids = np.union1d(postings_1, postings_2)

            res = PostingList(postings=doc_ids)
            res.build_skip_pointers()
        else:
            # all_doc_ids and postings must be sorted
            all_doc_ids = inverted_index.all_doc_ids
            doc_ids = []
            i = j = 0
            postings = posting_list_2.postings
            len_docs, len_postings = len(all_doc_ids), len(postings)

            while i < len_docs and j < len_postings:
                if all_doc_ids[i] < postings[j]:
                    doc_ids.append(all_doc_ids[i])
                    i += 1
                elif all_doc_ids[i] == postings[j]:
                    i += 1
                    j += 1
                else:
                    j += 1

            # add missing docs
            if i < len_docs:
                doc_ids.extend(all_doc_ids[i:])

            res = PostingList(postings=np.array(doc_ids))
            res.build_skip_pointers()

        return res

    @staticmethod
    def evaluate(node: Node) -> PostingList:
        if node.value not in AND | OR | NOT:
            return inverted_index.index.get(node.value)

        if node.value in AND:
            l = QueryEngine.evaluate(node.left)
            r = QueryEngine.evaluate(node.right)
            return QueryEngine._find_docs(l, r, "AND")

        if node.value in OR:
            l = QueryEngine.evaluate(node.left)
            r = QueryEngine.evaluate(node.right)
            return QueryEngine._find_docs(l, r, "OR")

        if node.value in NOT:
            r = QueryEngine.evaluate(node.right)  # not-child stored right
            return QueryEngine._find_docs(None, r, "NOT")

    def search_results(self, limit: int = 10) -> list[SearchResult]:
        qt = QueryTree()
        qt.parse_query(self._normalized_query())
        posting_lists = QueryEngine.evaluate(qt.root)

        search_results = []
        for doc_id in posting_lists.postings[:limit]:
            search_result = SearchResult(
                document_id=doc_id,
                url=inverted_index.doc_store.get(doc_id).get("url"),
                title=inverted_index.doc_store.get(doc_id).get("title"),
            )
            search_results.append(search_result)

        return search_results
