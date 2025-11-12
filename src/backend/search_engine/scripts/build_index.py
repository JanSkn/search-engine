import argparse
import json
import sys

from backend.search_engine.models.index import PostingList
from backend.search_engine.index.inverted_index import InvertedIndex
from backend.search_engine.indexer.index_builder import build_index


def postinglist_to_dict(pl: PostingList) -> dict:
    return {
        "postings": pl.postings.tolist(),
        "term_frequencies": pl.term_frequencies,
        "positions": {doc_id: pos.tolist() for doc_id, pos in pl.positions.items()},
        "skip_pointers": pl.skip_pointers,
        "doc_freq": pl.doc_freq,
    }


def inverted_index_to_dict(inv: InvertedIndex) -> dict:
    return {
        "index": {term: postinglist_to_dict(pl) for term, pl in inv.index.items()},
        "doc_store": inv.doc_store,
        "num_docs": getattr(inv, "_num_docs", len(inv.doc_store)),
    }


def main():
    ap = argparse.ArgumentParser(
        description="Build inverted index from JSONL(.gz) and store it as JSON."
    )
    ap.add_argument("--jsonl", required=True, help="Path to JSONL/JSONL.GZ file.")
    ap.add_argument("--out", required=True, help="Target file (e.g. index.json).")
    ap.add_argument(
        "--limit", type=int, default=None, help="Max. number of documents)."
    )
    ap.add_argument("--lang", default="en", help="Language (default: en).")
    ap.add_argument("--no-lemma", action="store_true", help="Deactivate lemmatization.")
    args = ap.parse_args()

    print(f"[INFO] Building inverted index from: {args.jsonl}", file=sys.stderr)
    inv = build_index(
        jsonl_path=args.jsonl,
        limit=args.limit,
        lang=args.lang,
        use_lemma=(not args.no_lemma),
    )

    print(f"[INFO] Storing index as JSON: {args.out}", file=sys.stderr)
    data = inverted_index_to_dict(inv)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print("[INFO] Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
