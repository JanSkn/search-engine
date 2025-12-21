from pydantic import BaseModel, HttpUrl


class SearchResult(BaseModel):
    document_id: int
    url: HttpUrl
    title: str
    # snippet: str
