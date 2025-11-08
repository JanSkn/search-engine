import numpy as np
from collections import defaultdict
from typing import Dict, List, Iterable, Tuple, Set
from pydantic import HttpUrl
from backend.search_engine.models.index import PostingList


class InvertedIndex:
    # term -> posting list
    def __init__(self) -> None:
        self.doc_ids: np.ndarray
        self.index: Dict[str, PostingList] = defaultdict(dict)
        self.doc_store: Dict[int, HttpUrl]


    def finalize(self) -> None:
        # computes document frequencies and caches doc count
        self._num_docs = len(self._doc_store)
        self._df = {t: len(docs) for t, docs in self._index.items()}


index = {
    "Hello": ([doc_ids], skip_list),
    "World": ([doc_ids], skip_list)
}