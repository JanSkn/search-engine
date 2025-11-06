#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sample 15k MS MARCO docs, tokenize + lemmatize with spaCy, save as JSONL (.jsonl or .jsonl.gz).
"""

import argparse
import gzip
import io
import json
import random
import sys
from typing import Iterable, Tuple, List, Dict

import requests
import spacy
from tqdm import tqdm


def open_stream(tsv_gz: str) -> io.BufferedReader:
    """Return a binary file-like for the gzipped TSV (local path or HTTP URL)."""
    if tsv_gz.startswith("http://") or tsv_gz.startswith("https://"):
        resp = requests.get(tsv_gz, stream=True, timeout=60)
        resp.raise_for_status()
        return gzip.GzipFile(fileobj=resp.raw)  # type: ignore
    else:
        return gzip.open(tsv_gz, "rb")


def iter_docs(tsv_stream: io.BufferedReader) -> Iterable[Tuple[str, str, str, str]]:
    """Yield (docid, url, title, body) from gzipped TSV (robust parsing)."""
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
    """Reservoir sampling for uniform random sample."""
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


def to_record(docid: str, url: str, title: str, lemmas: List[str], tokens: List[str]) -> Dict:
    return {
        "doc_id": docid,
        "url": url,
        "title": title,
        "tokens": tokens,
        "lemmas": lemmas,
    }


def spaCy_process(
    nlp,
    samples: List[Tuple[str, str, str, str]],
    batch_size: int = 1000,
    n_process: int = 1,
    keep_punct: bool = False,
    keep_spaces: bool = False,
) -> Iterable[Dict]:
    """Tokenize + lemmatize samples with spaCy."""
    texts = []
    metas = []
    for docid, url, title, body in samples:
        text = (title + "\n\n" + body).strip()
        texts.append(text)
        metas.append((docid, url, title))

    for doc, (docid, url, title) in tqdm(
        zip(nlp.pipe(texts, batch_size=batch_size, n_process=n_process), metas),
        total=len(texts),
        desc="spaCy processing",
        unit="doc",
    ):
        toks, lems = [], []
        for tok in doc:
            if (not keep_spaces and tok.is_space) or (not keep_punct and tok.is_punct):
                continue
            if tok.is_stop:
                continue
            if tok.is_alpha or tok.like_num or tok.is_currency or tok.is_ascii:
                toks.append(tok.text.lower())
                lemma = tok.lemma_.lower() if tok.lemma_ else tok.text.lower()
                lems.append(lemma)
        yield to_record(docid, url, title, lems, toks)


def write_jsonl(records: Iterable[Dict], out_path: str):
    """Write one JSON object per line (optionally gzip)."""
    open_func = gzip.open if out_path.endswith(".gz") else open
    mode = "wt" if out_path.endswith(".gz") else "w"
    with open_func(out_path, mode, encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Sample 15k MS MARCO docs, tokenize+lemmatize with spaCy, save as JSONL (.jsonl or .jsonl.gz).")
    ap.add_argument("--tsv-gz", required=True, help="Path or URL to ms-marco-docs.tsv.gz")
    ap.add_argument("--n", type=int, default=15000, help="Sample size (default: 15000)")
    ap.add_argument("--out", required=True, help="Output file (e.g., msmarco_15k_spacy.jsonl.gz)")
    ap.add_argument("--spacy-model", default="en_core_web_sm", help="spaCy model (default: en_core_web_sm)")
    ap.add_argument("--batch-size", type=int, default=1000)
    ap.add_argument("--n-process", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--keep-punct", action="store_true")
    ap.add_argument("--keep-spaces", action="store_true")
    args = ap.parse_args()

    print(f"Opening source: {args.tsv_gz}", file=sys.stderr)
    with open_stream(args.tsv_gz) as gz_stream:
        sampled = reservoir_sample(
            tqdm(iter_docs(gz_stream), desc="Scanning & sampling", unit="line", mininterval=1.0),
            k=args.n,
            seed=args.seed,
        )

    print(f"Loaded {len(sampled)} docs. Loading spaCy model: {args.spacy_model}", file=sys.stderr)
    nlp = spacy.load(args.spacy_model, disable=["ner","parser","textcat","senter"])

    records_iter = spaCy_process(
        nlp,
        sampled,
        batch_size=args.batch_size,
        n_process=args.n_process,
        keep_punct=args.keep_punct,
        keep_spaces=args.keep_spaces,
    )

    print(f"Writing JSONL to: {args.out}", file=sys.stderr)
    write_jsonl(records_iter, args.out)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
