import argparse
from backend.search_engine.query.query_engine import QueryEngine


def main():
    ap = argparse.ArgumentParser(
        description="Run a query against a pre-built inverted index."
    )
    ap.add_argument("--query", required=True, help="Search query.")
    ap.add_argument("--limit", type=int, default=10, help="Max results.")

    args = ap.parse_args()

    qe = QueryEngine(args.query)
    results = qe.search_results(limit=args.limit)

    for r in results:
        print(f"[{r.document_id}] {r.title} — {r.url}")


if __name__ == "__main__":
    main()
