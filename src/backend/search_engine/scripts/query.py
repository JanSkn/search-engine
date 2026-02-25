import time
import argparse
from backend.search_engine.query.query_engine import QueryEngine
from backend.logging_config import setup_logging

setup_logging(level="DEBUG")


def main():
    ap = argparse.ArgumentParser(
        description="Run a query against a pre-built inverted index."
    )
    ap.add_argument("--query", type=str, required=True, help="Search query.")
    ap.add_argument("--limit", type=int, default=10, help="Max results.")

    args = ap.parse_args()

    start = time.time()
    qe = QueryEngine(args.query)
    results = qe.search_results(limit=args.limit)
    end = time.time()

    for r in results.search_results:
        print(f"[{r.document_id}] {r.title} — {r.url} — {r.score}")

    print(f"Total time: {end - start}s")

    correction = results.correction
    if correction:
        print(f"Did you mean '{correction}'?")
        user_input = input("[y/n] ").strip().lower()

        if user_input in ["y", "yes"]:
            start_corr = time.time()
            qe_corr = QueryEngine(correction)
            results_corr = qe_corr.search_results(limit=args.limit)
            end_corr = time.time()

            for r in results_corr.search_results:
                print(f"[{r.document_id}] {r.title} — {r.url}")

            print(f"Total time: {end_corr - start_corr}s")


if __name__ == "__main__":
    main()
