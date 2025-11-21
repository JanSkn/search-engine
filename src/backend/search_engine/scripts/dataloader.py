import argparse
import gzip
import io
import os
import random
import sys
from typing import Iterable

import requests
from tqdm import tqdm

OUTPUT_DIR = "index_builder/data"
OUTPUT_FILE = "msmarco.tsv.gz"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUT_PATH = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", OUTPUT_DIR, OUTPUT_FILE
    )
)


def open_stream(tsv_gz: str) -> io.TextIOBase:
    if tsv_gz.startswith(("http://", "https://")):
        resp = requests.get(tsv_gz, stream=True, timeout=60)
        resp.raise_for_status()
        raw = resp.raw
        if tsv_gz.endswith(".gz"):
            return gzip.GzipFile(fileobj=raw, mode="rb")
        else:
            return raw
    else:
        if tsv_gz.endswith(".gz"):
            return gzip.open(tsv_gz, "rb")
        else:
            return open(tsv_gz, "rb")


def iter_docs(tsv_stream: io.BufferedReader) -> Iterable[tuple[str, str, str, str]]:
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


def reservoir_sample(
    stream: Iterable[tuple[str, str, str, str]], k: int, seed: int = 42
) -> list[tuple[str, str, str, str]]:
    rng = random.Random(seed)
    sample: list[tuple[str, str, str, str]] = []
    for i, item in enumerate(stream, start=1):
        if i <= k:
            sample.append(item)
        else:
            j = rng.randint(1, i)
            if j <= k:
                sample[j - 1] = item
    return sample


def write_tsv(records: Iterable[tuple[str, str, str, str]], out_path: str) -> None:
    """Write sampled records as gzipped TSV."""
    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        for docid, url, title, body in records:
            title_safe = title.replace("\t", " ").replace("\n", " ")
            body_safe = body.replace("\t", " ").replace("\n", " ")
            f.write(f"{docid}\t{url}\t{title_safe}\t{body_safe}\n")


def main():
    ap = argparse.ArgumentParser(
        description="Sample n MS MARCO docs and save as gzipped TSV."
    )
    ap.add_argument("--tsv", required=True, help="Path or URL to ms-marco-docs")
    ap.add_argument("--n", type=int, default=15000, help="Sample size (default: 15000)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = ap.parse_args()

    print(f"Opening source: {args.tsv}", file=sys.stderr)
    with open_stream(args.tsv) as gz_stream:
        sampled = reservoir_sample(
            tqdm(
                iter_docs(gz_stream),
                desc="Scanning & sampling",
                unit="line",
                mininterval=1.0,
            ),
            k=args.n,
            seed=args.seed,
        )

    print(
        f"Sampled {len(sampled)} docs. Writing TSV (gzipped) to: {OUT_PATH}",
        file=sys.stderr,
    )
    write_tsv(sampled, OUT_PATH)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
