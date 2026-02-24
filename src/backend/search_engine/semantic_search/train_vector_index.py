import time
from functools import lru_cache
from math import sqrt
from pathlib import Path

import faiss  # faiss-cpu

from backend.logging_config import get_logger
from backend.search_engine.index_builder.create_embeddings import (
    TOTAL_DOCS,
    NumpyIndexer,
)

logger = get_logger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT = PROJECT_DIR / "models" / "IVFPQ.faiss"
SAMPLE_SIZE = 50000


@lru_cache(maxsize=1)
def train_or_load_ivfpq():
    """
    Trains index if not trained yet, stores and returns it.

    Note: Stores total index with all documents, so storage
    requirements: numpy arrays (.npy) for embeddings and ids +
    embeddings and ids in the index
    """
    if CHECKPOINT.exists():
        logger.debug(f"Loading trained IVFPQ model from {CHECKPOINT}...")
        load_start = time.perf_counter()
        index = faiss.read_index(str(CHECKPOINT))
        logger.debug(f"Loaded in {time.perf_counter() - load_start:.6f}s")
        return index

    numpy_indexer = NumpyIndexer()
    embeddings, ids = numpy_indexer.load()
    d = numpy_indexer.dim
    nlist = int(sqrt(TOTAL_DOCS))  # number of clusters
    m = 8  # sub quantizers per vector
    bits = 8  # bits per sub quantizer
    assert d % m == 0

    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFPQ(quantizer, d, nlist, m, bits)

    logger.debug("Starting training...")
    start = time.time()
    index.train(embeddings[:SAMPLE_SIZE])
    logger.debug(f"Training finished in {time.time() - start}s")

    logger.debug("Starting indexing...")
    indexing_start = time.perf_counter()
    index.add_with_ids(embeddings, ids)
    logger.debug(
        f"Indexing finished in {time.perf_counter() - indexing_start:.6f}s. Documents: {index.ntotal}"
    )

    logger.debug("Starting to persist...")
    persisting_start = time.perf_counter()
    faiss.write_index(index, str(CHECKPOINT))
    logger.debug(
        f"Persisting finished in {time.perf_counter() - persisting_start:.6f}s"
    )

    return index
