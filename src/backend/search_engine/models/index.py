from math import sqrt
import numpy as np
from pydantic import BaseModel, Field, HttpUrl


class PostingList(BaseModel):
    postings: np.ndarray  # doc ids
    skip_pointers: dict[int, int] = Field(default_factory=dict)

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

    # for numpy
    model_config = {
        "arbitrary_types_allowed": True
    }


class SearchResult(BaseModel):
    document_id: int
    url: HttpUrl
    title: str