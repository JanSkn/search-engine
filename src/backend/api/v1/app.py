import os
from contextlib import asynccontextmanager
from typing import Annotated

from backend.logging_config import get_logger, setup_logging
from backend.search_engine.error_handling import InvalidOperatorError
from backend.search_engine.index.index_loader import get_index
from backend.search_engine.models.index import SearchResults
from backend.search_engine.query.query_engine import QueryEngine
from backend.search_engine.semantic_search.embedding_model import MAX_QUERY_LENGTH
from backend.search_engine.spell_correction.spell_corrector import get_spell_corrector
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware

setup_logging(level=os.getenv("LOG_LEVEL", "INFO"))
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.debug("Starting up...")
    app.state.inverted_index = get_index()
    app.state.spell_corrector = get_spell_corrector()
    yield
    logger.debug("Shutting down...")


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/search", response_model=SearchResults)
async def search(
    q: Annotated[
        str,
        Query(min_length=1, max_length=MAX_QUERY_LENGTH, description="Search query"),
    ],
    limit: Annotated[
        int, Query(ge=1, le=500, description="Maximum number of results")
    ] = 10,
) -> SearchResults:
    if app.state.inverted_index is None or app.state.spell_corrector is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Search index or spell corrector not loaded",
        )

    try:
        qe = QueryEngine(q)
        results = qe.search_results(limit)

        return results
    except InvalidOperatorError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid query syntax: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search operation failed: {str(e)}",
        )
