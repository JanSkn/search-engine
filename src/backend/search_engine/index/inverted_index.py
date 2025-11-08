import numpy as np
from backend.search_engine.models.index import PostingList


class InvertedIndex:
    def __init__(self) -> None:
        self.all_doc_ids: np.ndarray = np.array([], dtype=int)  # TODO needs much memory
        self.index: dict[str, PostingList] = {}
        self.doc_store: dict[int, str] = {}

    def finalize(self) -> None:
        # computes document frequencies and caches doc count
        self._num_docs = len(self.doc_store)
        self._df = {t: pl.doc_freq for t, pl in self.index.items()}

        for pl in self.index.values():
            if len(pl.postings) > 1:
                sorted_idx = np.argsort(pl.postings)
                pl.postings = pl.postings[sorted_idx]

                pl.build_skip_pointers()

    @classmethod
    def from_json(cls, path: str) -> "InvertedIndex":
        import json
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        inv = cls()
        inv.doc_store = {int(k): v for k, v in data.get("doc_store", {}).items()}
        inv._num_docs = data.get("num_docs", len(inv.doc_store))

        index_data = data.get("index", {})
        for term, pl_dict in index_data.items():
            postings = np.array(pl_dict["postings"], dtype=int)
            term_frequencies = {int(k): v for k, v in pl_dict["term_frequencies"].items()}
            positions = {int(k): np.array(v, dtype=int) for k, v in pl_dict["positions"].items()}
            skip_pointers = pl_dict.get("skip_pointers", {})

            pl = PostingList(
                postings=postings,
                term_frequencies=term_frequencies,
                positions=positions,
                skip_pointers=skip_pointers,
            )
            inv.index[term] = pl

        inv.finalize()
        return inv
