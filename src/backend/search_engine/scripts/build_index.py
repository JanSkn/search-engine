import argparse
import pickle
from ..indexer.dataloader import stream_documents
from ..indexer.index_builder import build_index


def main() -> None:
    parser = argparse.ArgumentParser(description="build inverted index with positions")
    parser.add_argument(
        "--corpus", type=str, required=True, help="path to tsv gz or jsonl"
    )
    parser.add_argument("--limit", type=int, default=None, help="optional doc limit")
    parser.add_argument(
        "--lang", type=str, default="en", help="language code for tokenizer"
    )
    parser.add_argument(
        "--out", type=str, default="index.pkl", help="path to output pickle"
    )
    args = parser.parse_args()

    docs = stream_documents(args.corpus)
    inv = build_index(docs, limit=args.limit, lang=args.lang)
    with open(args.out, "wb") as f:
        pickle.dump(inv, f)

    print(f"ok built index with {inv.num_docs} docs to {args.out}")


if __name__ == "__main__":
    main()
