from pydantic import BaseModel, HttpUrl


class SearchResult(BaseModel):
    document_id: int
    url: HttpUrl
    title: str
    snippet: str
    score: float


class SearchResults(BaseModel):
    search_results: list[SearchResult]
    correction: str | None = None
