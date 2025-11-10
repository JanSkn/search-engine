from typing import Annotated
from fastapi import FastAPI, Query, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.search_engine.models.index import SearchResult
from backend.search_engine.query.query_engine import QueryEngine

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# TODO change once not in json anymore
from backend.search_engine.query.query_engine import inverted_index
from backend.search_engine.index.inverted_index import InvertedIndex

inverted_index_loaded = InvertedIndex.from_json(
    "PATH"
)
inverted_index.index = inverted_index_loaded.index
inverted_index.doc_store = inverted_index_loaded.doc_store
inverted_index.all_doc_ids = inverted_index_loaded.all_doc_ids
# ---------------------


@app.get("/search", response_model=list[SearchResult])
async def search(
    q: Annotated[
        str, Query(min_length=1, max_length=50, description="Search query")
    ] = ...,
    limit: Annotated[
        int, Query(ge=1, le=100, description="Maximum number of results")
    ] = 10,
) -> list[SearchResult]:
    if inverted_index is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Search index not loaded",
        )

    try:
        qe = QueryEngine(q)

        return qe.search_results(limit)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid query syntax: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Search operation failed",
        )
