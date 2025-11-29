from typing import Annotated
import os
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Query, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from backend.logging_config import setup_logging, get_logger
from backend.search_engine.models.index import SearchResult
from backend.search_engine.index.index_loader import get_index
from backend.search_engine.query.query_engine import QueryEngine
from backend.search_engine.error_handling import InvalidOperatorError

setup_logging(level=os.getenv("LOG_LEVEL", "INFO"))
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.debug("Loading search index...")
    start = time.time()
    app.state.inverted_index = get_index()
    end = time.time()
    logger.debug(f"Search index loaded in {end - start}s")
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


@app.get("/search", response_model=list[SearchResult])
async def search(
    q: Annotated[str, Query(min_length=1, max_length=50, description="Search query")],
    limit: Annotated[
        int, Query(ge=1, le=100, description="Maximum number of results")
    ] = 10,
) -> list[SearchResult]:
    if app.state.inverted_index is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Search index not loaded",
        )

    try:
        qe = QueryEngine(q)

        return qe.search_results(limit)
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
