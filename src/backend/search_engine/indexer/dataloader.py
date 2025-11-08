import argparse
import gzip
import io
import json
import random
import sys
from typing import Iterable, Tuple, List, Dict

import requests
from tqdm import tqdm


def open_stream(tsv_gz: str) -> io.BufferedReader:
    if tsv_gz.startswith(("http://", "https://")):
        resp = requests.get(tsv_gz, stream=True, timeout=60)
        resp.raise_for_status()
        return gzip.GzipFile(fileobj=resp.raw)
    else:
        return gzip.open(tsv_gz, "rb")


def iter_docs(tsv_stream: io.BufferedReader) -> Iterable[Tuple[str, str, str, str]]:
    # get (docid, url, title, body)
    for line in tsv_stream:
        try:
            line = line.decode("utf-8", errors="replace")
        except Exception:
            line = line.decode("latin-1", errors="replace")
        parts = line.rstrip("\n").split("\t", maxsplit=3)
        while len(parts) < 4:
            parts.append("")
        docid, url, title, body = parts[0], parts[1], parts[2], parts[3]
        yield docid, url, title, body


def reservoir_sample(stream: Iterable[Tuple[str, str, str, str]], k: int, seed: int = 42):
    rng = random.Random(seed)
    sample: List[Tuple[str, str, str, str]] = []
    for i, item in enumerate(stream, start=1):
        if i <= k:
            sample.append(item)
        else:
            j = rng.randint(1, i)
            if j <= k:
                sample[j - 1] = item
    return sample


def write_jsonl(records: Iterable[Dict], out_path: str):
    is_gz = out_path.endswith(".gz")
    open_func = gzip.open if is_gz else open
    mode = "wt" if is_gz else "w"
    with open_func(out_path, mode, encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Sample n MS MARCO docs and save as JSONL (.jsonl or .jsonl.gz).")
    ap.add_argument("--tsv-gz", required=True, help="Path or URL to ms-marco-docs.tsv.gz")
    ap.add_argument("--n", type=int, default=15000, help="Sample size (default: 15000)")
    ap.add_argument("--out", required=True, help="Output file, e.g., msmarco_15k.jsonl.gz")
    ap.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = ap.parse_args()

    print(f"Opening source: {args.tsv_gz}", file=sys.stderr)
    with open_stream(args.tsv_gz) as gz_stream:
        sampled = reservoir_sample(
            tqdm(iter_docs(gz_stream), desc="Scanning & sampling", unit="line", mininterval=1.0),
            k=args.n,
            seed=args.seed,
        )

    print(f"Sampled {len(sampled)} docs. Writing JSONL to: {args.out}", file=sys.stderr)
    records = ({"doc_id": d, "url": u, "title": t, "body": b} for d, u, t, b in sampled)
    write_jsonl(records, args.out)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()