import argparse
import time

from backend.logging_config import get_logger, setup_logging
from backend.search_engine.semantic_search.embedding_model import get_embedding_model
from backend.search_engine.semantic_search.train_vector_index import train_or_load_ivfpq

logger = get_logger(__name__)


class SemanticSearcher:
    def __init__(self):
        self.model = get_embedding_model()
        self.index = train_or_load_ivfpq()

    def search(self, query: str, top_n: int = 10):
        logger.debug(f"Semantic serach for: '{query}'")
        start = time.perf_counter()

        query_vector = self.model.embed_query(query)
        scores, ids = self.index.search(query_vector, top_n)

        logger.debug(
            f"Search found {len(ids[0])} results in {time.perf_counter() - start:.4f}s"
        )

        return scores[0], ids[0]


if __name__ == "__main__":
    setup_logging(level="DEBUG")

    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str, help="Search query")
    parser.add_argument("--top-n", type=int, default=5, help="Number of results")

    args = parser.parse_args()

    searcher = SemanticSearcher()
    scores, ids = searcher.search(args.query, top_n=args.top_n)

    print("\nSearch Results:")
    for score, id_ in zip(scores, ids):
        print(f"ID: {id_:<10} Score: {score:.4f}")
