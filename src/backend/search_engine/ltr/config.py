from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class LTRPaths:
    data_dir: Path = Path("data/msmarco")
    queries_tsv: Path = data_dir / "msmarco-doctrain-queries.tsv"
    qrels_tsv: Path = data_dir / "msmarco-doctrain-qrels.tsv"
    top100_tsv: Path = data_dir / "msmarco-doctrain-top100.tsv"
    docs_tsv: Path =  Path("backend/search_engine/index_builder/data/msmarco-docs.tsv")

    out_dir: Path = Path("data/ltr_out")
    train_jsonl: Path = out_dir / "train.jsonl"
    val_jsonl: Path = out_dir / "val.jsonl"
    test_jsonl: Path = out_dir / "test.jsonl"

@dataclass(frozen=True)
class SamplingConfig:
    hard_top_k: int = 20
    easy_from_rank_low: int = 60
    easy_from_rank_high: int = 100
    easy_n: int = 12
    max_pos_per_qid: int = 1

@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    list_size: int = 1 + SamplingConfig().hard_top_k + SamplingConfig().easy_n  # pos + negatives
    batch_size: int = 256
    epochs: int = 3
    lr: float = 1e-3

@dataclass(frozen=True)
class DatasetBuildConfig:
    tfidf_fit_qids: int = 10_000      # smoke: 500, full: z.B. 50_000
    smoke_limit_qids: int | None = None  # None for full run
    max_features: int = 100_000