import time
from functools import lru_cache
from math import sqrt
from pathlib import Path

# faiss-cpu
import faiss  # type: ignore [import-untyped]

from backend.logging_config import get_logger
from backend.memory_tracer import trace_memory
from backend.search_engine.index_builder.create_embeddings import (
    NumpyIndexer,
)
from backend.utils import TempOMPThreads

logger = get_logger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT = PROJECT_DIR / "models" / "IVFPQ.faiss"
MIN_POINTS_PER_CENTROID = 39


@lru_cache(maxsize=1)
@trace_memory
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

    with TempOMPThreads(1):  # macOs workaround
        numpy_indexer = NumpyIndexer()
        embeddings, ids = numpy_indexer.load()
        d = numpy_indexer.dim
        num_vectors = embeddings.shape[0]
        nlist = int(sqrt(num_vectors))  # number of clusters
        sample_size = min(num_vectors, max(MIN_POINTS_PER_CENTROID * nlist, nlist))
        m = 8  # sub quantizers per vector
        bits = 8  # bits per sub quantizer
        assert d % m == 0
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFPQ(quantizer, d, nlist, m, bits)
        index.nprobe = 5  # how many clusters to look into

        logger.debug(
            f"Training with {sample_size} points for {nlist} centroids "
            f"({sample_size / nlist:.0f} points/centroid)"
        )
        start = time.time()
        index.train(embeddings[:sample_size])
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
