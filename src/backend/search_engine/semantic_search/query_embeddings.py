import time

from backend.logging_config import get_logger
from backend.search_engine.semantic_search.embedding_model import get_embedding_model
from backend.search_engine.semantic_search.train_vector_index import train_or_load_ivfpq

logger = get_logger(__name__)


class SemanticSearcher:
    def __init__(self):
        self.model = get_embedding_model()
        self.index = train_or_load_ivfpq()

    def search(self, query: str, top_n):
        logger.debug(f"Semantic search for: '{query}'")
        start = time.perf_counter()

        query_vector = self.model.embed_query(query).reshape(1, -1)
        scores, ids = self.index.search(query_vector, top_n)

        results = list(zip(ids[0], scores[0]))

        logger.debug(
            f"Semantic search found {len(results)} results in {time.perf_counter() - start:.4f}s"
        )

        return results
