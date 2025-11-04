from math import sqrt
from pydantic import BaseModel, HttpUrl


class PostingList(BaseModel):
    """
    Posting list for a term.

    Elements at same index belong to each other,
    e.g. postings[i] has urls[i] etc.

    Design reason is faster access for sorting and comparing documents
    """

    urls: list[HttpUrl]
    titles: list[str]
    doc_freq: int  # equals length of postings
    term_frequencies: list[int]
    positions: list[list[int]]
    postings: list[int]  # doc ids
    skip_pointers: dict[int, int]

    def build_skip_pointers(self, index: int) -> None:
        n = len(self.postings)
        step = int(sqrt(n)) if n > 0 else 0

        if step > 1:
            for i in range(0, n, step):
                j = i + step
                if j < n:
                    self.skip_pointers[i] = j  # skip from i to j

    # zip to keep index order
    # TODO check if faster if ordering separately with
    # order = sorted(range(len(self.postings)), key=self.postings.__getitem__)
    # self.postings = [self.postings[i] for i in order]
    # self.urls = [self.urls[i] for i in order]
    # ...
    def sort_postings(self) -> None:
        combined = list(
            zip(
                self.postings,
                self.urls,
                self.titles,
                self.term_frequencies,
                self.positions,
            )
        )

        combined.sort(key=lambda x: x[0])

        self.postings, self.urls, self.titles, self.term_frequencies, self.positions = (
            map(list, zip(*combined))
        )


class SearchResult(BaseModel):
    document_id: int
    url: HttpUrl
    title: str
