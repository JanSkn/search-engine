from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Top100Entry:
    docid: int
    rank: int
    score: float

def docid_str_to_int(docid: str) -> int:
    # "D12345" -> 12345
    # if w/o "D" already:
    return int(docid[1:]) if docid.startswith("D") else int(docid)

def load_queries(path: Path) -> dict[str, str]:
    # qid \t query
    out: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            qid, query = line.rstrip("\n").split("\t", 1)
            out[qid] = query
    return out

def load_qrels(path: Path) -> dict[str, set[int]]:
    pos = defaultdict(set)
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()  # whitespace split
            if len(parts) != 4:
                raise ValueError(f"qrels bad line {lineno}: expected 4 fields, got {len(parts)}: {line[:200]}")

            qid, _dummy, docid, rel = parts
            if int(rel) > 0:
                pos[qid].add(docid_str_to_int(docid))
    return dict(pos)

def load_top100(path: Path) -> dict[str, list[Top100Entry]]:
    # qid dummy docID rank score algorithm
    out: dict[str, list[Top100Entry]] = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            qid, _q0, docid, rank, score, _alg = line.rstrip("\n").split()
            out[qid].append(Top100Entry(docid=docid_str_to_int(docid), rank=int(rank), score=float(score)))
    # sort by rank
    for qid in out:
        out[qid].sort(key=lambda e: e.rank)
    return dict(out)

def iter_top100_for_qids(path: Path, qids_keep: set[str]):
    """
    Streamt top100 und liefert nur Einträge für gewünschte QIDs.
    Yields: (qid, Top100Entry)
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            qid, _q0, docid, rank, score, _alg = line.rstrip("\n").split()
            if qid not in qids_keep:
                continue
            yield qid, Top100Entry(
                docid=docid_str_to_int(docid),
                rank=int(rank),
                score=float(score),
            )
            