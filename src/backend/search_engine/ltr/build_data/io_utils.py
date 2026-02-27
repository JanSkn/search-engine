from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Query:
    qid: int
    query: str


def iter_queries(path: Path) -> Iterable[Query]:
    # doctrain-queries.tsv: qid \t query
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", maxsplit=1)
            if len(parts) != 2:
                continue
            qid_s, q = parts
            try:
                yield Query(qid=int(qid_s), query=q)
            except ValueError:
                continue


def sample_qids(
    queries_path: Path, *, seed: int, max_queries: int | None
) -> list[Query]:
    rnd = random.Random(seed)

    if max_queries is None:
        return list(iter_queries(queries_path))

    k = int(max_queries)
    if k <= 0:
        return []

    reservoir: list[Query] = []
    n = 0
    for q in iter_queries(queries_path):
        n += 1
        if len(reservoir) < k:
            reservoir.append(q)
        else:
            j = rnd.randrange(n)
            if j < k:
                reservoir[j] = q
    return reservoir


def sample_qids_from_qrels(
    qrels_path: Path, *, seed: int, max_queries: int | None
) -> list[int]:
    """
    Sample qids from doctrain-qrels.tsv (space-separated).
    Uses reservoir sampling if max_queries is set.
    """
    rnd = random.Random(seed)

    def iter_qrel_qids() -> Iterable[int]:
        with qrels_path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 3:
                    continue
                try:
                    yield int(parts[0])
                except ValueError:
                    continue

    if max_queries is None:
        # all qids (might be large)
        return list(iter_qrel_qids())

    k = int(max_queries)
    if k <= 0:
        return []

    reservoir: list[int] = []
    n = 0
    for qid in iter_qrel_qids():
        n += 1
        if len(reservoir) < k:
            reservoir.append(qid)
        else:
            j = rnd.randrange(n)
            if j < k:
                reservoir[j] = qid
    return reservoir


def load_qrels_for_qids(qrels_path: Path, qids: set[int]) -> dict[int, int]:
    # doctrain-qrels: qid dummy docid dummy (space-separated)
    # example: 211691 0 D1499345 1  (label=1)
    out: dict[int, int] = {}
    with qrels_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                qid = int(parts[0])
            except ValueError:
                continue
            if qid not in qids:
                continue
            docid = parts[2]
            # docid is like "D12345"
            if not docid or docid[0] != "D":
                continue
            try:
                out[qid] = int(docid[1:])
            except ValueError:
                continue
    return out


@dataclass
class Top100Selection:
    hard: list[int]
    soft_candidates: dict[int, int]  # rank -> docid


def load_top100_selection(
    top100_path: Path,
    qids: set[int],
    *,
    hard_k: int,
    soft_rank: int,
    soft_fallback_from: int,
    soft_fallback_to: int,
) -> dict[int, Top100Selection]:
    """
    Streaming parse msmarco-doctrain-top100.tsv:
      qid \t docid \t rank \t score
    We only keep:
      - hard ranks: 1..hard_k
      - soft ranks: [soft_fallback_from..soft_fallback_to] (so we can pick best available)
    """
    soft_low = min(soft_fallback_from, soft_fallback_to)
    soft_high = max(soft_fallback_from, soft_fallback_to)

    out: dict[int, Top100Selection] = {}

    with top100_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            # TREC format: qid Q0 docid rank score run_tag (space-separated)
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                qid = int(parts[0])
            except ValueError:
                continue
            if qid not in qids:
                continue

            docid_s = parts[2]
            rank_s = parts[3]

            if not docid_s.startswith("D"):
                continue
            try:
                docid = int(docid_s[1:])
                rank = int(rank_s)
            except ValueError:
                continue

            sel = out.get(qid)
            if sel is None:
                sel = Top100Selection(hard=[], soft_candidates={})
                out[qid] = sel

            if 1 <= rank <= hard_k:
                sel.hard.append(docid)

            if soft_low <= rank <= soft_high:
                sel.soft_candidates[rank] = docid

    return out
