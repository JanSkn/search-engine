import argparse
import time

from backend.logging_config import get_logger, setup_logging
from backend.memory_tracer import trace_torch
from backend.search_engine.semantic_search.embedding_model import get_embedding_model
from backend.search_engine.semantic_search.train_vector_index import train_or_load_ivfpq

logger = get_logger(__name__)


class SemanticSearcher:
    @trace_torch
    def __init__(self):
        self.model = get_embedding_model()
        self.index = train_or_load_ivfpq()

    @trace_torch
    def search(self, query: str, top_n):
        logger.debug(f"Semantic search for: '{query}'")
        start = time.perf_counter()

        query_vector = self.model.embed_query(query).reshape(1, -1)
        scores, ids = self.index.search(query_vector, top_n)

        results = list(zip(ids[0], scores[0]))

        logger.debug(
            f"Search found {len(results)} results in {time.perf_counter() - start:.4f}s"
        )

        return results


if __name__ == "__main__":
    setup_logging(level="DEBUG")

    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str, help="Search query")
    parser.add_argument("--top-n", type=int, default=5, help="Number of results")

    args = parser.parse_args()

    searcher = SemanticSearcher()
    scores = searcher.search(args.query, top_n=args.top_n)

    print("\nSearch Results:")
    for id_, score in scores:
        print(f"ID: {id_:<10} Score: {score:.4f}")
