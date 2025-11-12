from math import sqrt
import numpy as np
from dataclasses import dataclass, field
from pydantic import BaseModel, HttpUrl


@dataclass(slots=True)
class PostingList:
    postings: np.ndarray  # doc ids
    # dicts with doc ids as key
    # avoids resorting if postings get sorted
    term_frequencies: dict[int, int]
    positions: dict[int, np.ndarray]

    skip_pointers: dict[int, int] = field(default_factory=dict)

    @property
    def doc_freq(self) -> int:
        return len(self.postings)

    def build_skip_pointers(self) -> None:
        n = len(self.postings)
        step = int(sqrt(n)) if n > 0 else 0

        if step > 1:
            for i in range(0, n, step):
                j = i + step
                if j < n:
                    self.skip_pointers[i] = j  # skip from i to j


class SearchResult(BaseModel):
    document_id: int
    url: HttpUrl
    title: str
