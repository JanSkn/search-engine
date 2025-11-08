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

# TODO later without loading from JSON
inverted_index = InvertedIndex.from_json("/Users/Jan/VSCode/search-engine/src/index.json")


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
            tf_1 = posting_list_1.term_frequencies
            tf_2 = posting_list_2.term_frequencies
            pos_1 = posting_list_1.positions
            pos_2 = posting_list_2.positions
        else:
            postings_1 = posting_list_2.postings
            postings_2 = posting_list_1.postings
            skip_pointers_1 = posting_list_2.skip_pointers
            skip_pointers_2 = posting_list_1.skip_pointers
            tf_1 = posting_list_2.term_frequencies
            tf_2 = posting_list_1.term_frequencies
            pos_1 = posting_list_2.positions
            pos_2 = posting_list_1.positions

        i = j = 0
        
        if mode == "AND":
            doc_ids = []
            result_tf = {}
            result_pos = {}
            
            while i < l_1 and j < l_2:
                if postings_1[i] == postings_2[j]:
                    doc_id = postings_1[i]
                    doc_ids.append(doc_id)
                    
                    result_tf[doc_id] = tf_1.get(doc_id, 0) + tf_2.get(doc_id, 0)
                    
                    pos_list = []
                    if doc_id in pos_1:
                        pos_list.append(pos_1[doc_id])
                    if doc_id in pos_2:
                        pos_list.append(pos_2[doc_id])
                    if pos_list:
                        result_pos[doc_id] = np.concatenate(pos_list)
                    
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
            
            res = PostingList(
                postings=np.array(doc_ids),
                term_frequencies=result_tf,
                positions=result_pos
            )
            res.build_skip_pointers()
        
        elif mode == "OR":
            doc_ids = np.union1d(postings_1, postings_2)
            result_tf = {}
            result_pos = {}
            
            for doc_id in doc_ids:
                result_tf[doc_id] = tf_1.get(doc_id, 0) + tf_2.get(doc_id, 0)
                
                pos_list = []
                if doc_id in pos_1:
                    pos_list.append(pos_1[doc_id])
                if doc_id in pos_2:
                    pos_list.append(pos_2[doc_id])
                if pos_list:
                    result_pos[doc_id] = np.concatenate(pos_list)
            
            res = PostingList(
                postings=doc_ids,
                term_frequencies=result_tf,
                positions=result_pos
            )
            res.build_skip_pointers()
        
        else:  # NOT
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
            
            # TODO no tf and positions because excluding?
            res = PostingList(
                postings=np.array(doc_ids),
                term_frequencies={},
                positions={}
            )
            res.build_skip_pointers()
        
        return res

    @staticmethod
    def evaluate(node: Node) -> PostingList:
        if node.value not in AND | OR | NOT:
            pl = inverted_index.index.get(node.value)
            if pl is None:
                return PostingList(
                    postings=np.array([]),
                    term_frequencies={},
                    positions={}
                )
            return pl

        if node.value in AND:
            l = QueryEngine.evaluate(node.left)
            r = QueryEngine.evaluate(node.right)
            return QueryEngine._find_docs(l, r, "AND")

        if node.value in OR:
            l = QueryEngine.evaluate(node.left)
            r = QueryEngine.evaluate(node.right)
            return QueryEngine._find_docs(l, r, "OR")

        if node.value in NOT:
            empty = PostingList(
                postings=np.array([]),
                term_frequencies={},
                positions={}
            )
            r = QueryEngine.evaluate(node.right)  # not-child stored right
            return QueryEngine._find_docs(empty, r, "NOT")

    def search_results(self, limit: int = 10) -> list[SearchResult]:
        qt = QueryTree()
        qt.parse_query(self._normalized_query())
        posting_lists = QueryEngine.evaluate(qt.root)
        
        if posting_lists is None or len(posting_lists.postings) == 0:
            return []
        
        search_results = []
        for doc_id in posting_lists.postings[:limit]:
            doc_data = inverted_index.doc_store.get(doc_id)
            
            if doc_data is None:
                print(f"Warning: doc_id {doc_id} not found in doc_store")
                continue
            
            url = doc_data.get("url")
            title = doc_data.get("title", "Untitled")
            
            if url is None:
                print(f"Warning: doc_id {doc_id} has no URL")
                continue
            
            try:
                search_result = SearchResult(
                    document_id=doc_id,
                    url=url,
                    title=title,
                )
                search_results.append(search_result)
            except Exception as e:
                print(f"Error creating SearchResult for doc_id {doc_id}: {e}")
                continue
        
        return search_results
