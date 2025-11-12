import time
import argparse
from backend.search_engine.index.inverted_index import InvertedIndex
from backend.search_engine.query.query_engine import QueryEngine, inverted_index
from backend.logging_config import setup_logging

setup_logging(level="DEBUG")


def main():
    ap = argparse.ArgumentParser(
        description="Run a query against a pre-built inverted index."
    )
    ap.add_argument("--query", type=str, required=True, help="Search query.")
    ap.add_argument("--index", type=str, required=True, help="Path to JSON.")
    ap.add_argument("--limit", type=int, default=10, help="Max results.")

    args = ap.parse_args()

    start = time.time()
    inverted_index_loaded = InvertedIndex.from_json(args.index)

    # TODO global for now until not loaded from JSON anymore
    inverted_index.index = inverted_index_loaded.index
    inverted_index.doc_store = inverted_index_loaded.doc_store
    inverted_index.all_doc_ids = inverted_index_loaded.all_doc_ids

    search_start = time.time()
    qe = QueryEngine(args.query)
    results = qe.search_results(limit=args.limit)
    end = time.time()

    for r in results:
        print(f"[{r.document_id}] {r.title} — {r.url}")

    print(f"Total time: {end - start}s, search time: {end - search_start}s")


if __name__ == "__main__":
    main()
