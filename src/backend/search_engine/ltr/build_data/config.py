from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetPaths:
    queries_tsv: Path
    qrels_tsv: Path
    top100_tsv: Path

    out_dir: Path

    train_jsonl: Path
    val_jsonl: Path
    test_jsonl: Path

    @staticmethod
    def from_base(data_dir: Path, out_dir: Path) -> "DatasetPaths":
        return DatasetPaths(
            queries_tsv=data_dir / "msmarco-doctrain-queries.tsv",
            qrels_tsv=data_dir / "msmarco-doctrain-qrels.tsv",
            top100_tsv=data_dir / "msmarco-doctrain-top100.tsv",
            out_dir=out_dir,
            train_jsonl=out_dir / "train.jsonl",
            val_jsonl=out_dir / "val.jsonl",
            test_jsonl=out_dir / "test.jsonl",
        )


@dataclass(frozen=True)
class BuildConfig:
    seed: int = 13

    # how many queries (qid) to sample from doctrain-queries
    max_queries: int | None = 10

    # negatives
    hard_negatives: int = 10  # take ranks 1..hard_negatives
    soft_negatives: int = 1   # take from bottom area / rank=100 by default

    # soft negative strategy
    soft_rank: int = 100              # try rank==100 first
    soft_fallback_from: int = 80      # if rank==100 conflicts, search ranks 99..80
    soft_fallback_to: int = 100

    # output format
    pretty_json: bool = False