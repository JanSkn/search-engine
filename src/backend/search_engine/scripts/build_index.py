import argparse
import json
import sys

from backend.search_engine.indexer.index_builder import build_index


def postinglist_to_dict(pl):

    return {
        "urls": pl.urls,
        "titles": pl.titles,
        "doc_freq": pl.doc_freq,
        "term_frequencies": pl.term_frequencies,
        "positions": pl.positions,
        "postings": pl.postings,
    }


def inverted_index_to_dict(inv):
    """Konvertiert den gesamten InvertedIndex in ein JSON-kompatibles Dict."""
    return {
        "index": {term: postinglist_to_dict(pl) for term, pl in inv._index.items()},
        "doc_store": inv._doc_store,
        "num_docs": getattr(inv, "_num_docs", len(inv._doc_store)),
    }


def main():
    ap = argparse.ArgumentParser(
        description="Baue einen Inverted Index aus JSONL(.gz) und speichere ihn als JSON."
    )
    ap.add_argument("--jsonl", required=True, help="Pfad zur JSONL/JSONL.GZ-Datei (mit doc_id, url, title, body).")
    ap.add_argument("--out", required=True, help="Zieldatei für den JSON-Index (z. B. index.json).")
    ap.add_argument("--limit", type=int, default=None, help="Max. Anzahl Dokumente (optional).")
    ap.add_argument("--spacy-model", default="en_core_web_sm", help="spaCy-Modell (default: en_core_web_sm).")
    ap.add_argument("--lang", default="en", help="Sprache (default: en).")
    ap.add_argument("--no-lemma", action="store_true", help="Lemmatisierung ausschalten.")
    args = ap.parse_args()

    print(f"[INFO] Baue Inverted Index aus: {args.jsonl}", file=sys.stderr)
    inv = build_index(
        jsonl_path=args.jsonl,
        limit=args.limit,
        lang=args.lang,
        spacy_model=args.spacy_model,
        use_lemma=(not args.no_lemma),
    )

    print(f"[INFO] Speichere Index als JSON: {args.out}", file=sys.stderr)
    data = inverted_index_to_dict(inv)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    print("[INFO] Fertig.", file=sys.stderr)


if __name__ == "__main__":
    main()
