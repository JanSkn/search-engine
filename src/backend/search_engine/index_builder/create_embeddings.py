from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
import torch
from backend.logging_config import get_logger, setup_logging
from backend.memory_tracer import trace_memory
from backend.search_engine.semantic_search.embedding_model import get_embedding_model
from backend.utils import measure_time
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from tqdm import tqdm

if __name__ == "__main__":
    setup_logging(level=os.getenv("LOG_LEVEL", "DEBUG"))
logger = get_logger(__name__)

MSMARCO_DIR = Path(__file__).resolve().parent / "data"
TSV_PATH = MSMARCO_DIR / "msmarco-docs.tsv"
TARGET_DIR = Path(__file__).resolve().parent.parent / "index" / "bin"
EMBEDDING_PATH = TARGET_DIR / "embeddings.npy"
DOCID_PATH = TARGET_DIR / "doc_ids.npy"
TOTAL_DOCS = 3_213_835


class MSMarcoDataset(IterableDataset):
    def __init__(
        self, tsv_path: Path, tokenizer, max_rows: int, task: str = "document"
    ):
        self.tsv_path = tsv_path
        self.tokenizer = tokenizer
        self.max_rows = max_rows
        # see https://huggingface.co/nomic-ai/nomic-embed-text-v1.5
        self.prefix = "search_document: " if task == "document" else "search_query: "

    def _parse_docid(self, docid_str: str) -> int:
        return int(docid_str[1:])

    def __iter__(self):
        worker_info = get_worker_info()

        lf = (
            pl.scan_csv(
                self.tsv_path,
                separator="\t",
                has_header=False,
            )
            .select(
                [
                    pl.col("column_1").alias("docid"),
                    pl.col("column_4").alias("text"),
                ]  # column_ index starts with 1
            )
            .filter(
                pl.col("text").is_not_null() & (pl.col("text").str.strip_chars() != "")
            )
            .limit(self.max_rows)
        )

        # give each worker subset of the data
        if worker_info is not None:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            lf = lf.with_row_index("idx").filter(
                pl.col("idx") % num_workers == worker_id
            )

        for batch in lf.collect_batches():
            for row_idx in range(len(batch)):
                docid_str = batch["docid"][row_idx]
                text = batch["text"][row_idx]
                docid = self._parse_docid(docid_str)

                # on CPU
                encoded = self.tokenizer(
                    self.prefix + text,
                    padding="max_length",
                    truncation=True,
                    max_length=128,
                    return_tensors="pt",
                )
                encoded = {k: v.squeeze(0) for k, v in encoded.items()}
                yield docid, encoded


class NumpyIndexer:
    def __init__(
        self, num_workers: int = 8, batch_size: int = 64, max_docs: int | None = None
    ):
        self.tsv_path = TSV_PATH
        self.target_dir = TARGET_DIR
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_docs = max_docs if max_docs and max_docs != -1 else TOTAL_DOCS
        self.embedder = get_embedding_model()
        self.dim = self.embedder.matryoshka_dim
        self.dtype = self.embedder.embedding_dtype

    @measure_time
    def run(self):
        self.target_dir.mkdir(parents=True, exist_ok=True)
        if EMBEDDING_PATH.exists() or DOCID_PATH.exists():
            logger.info("Embeddings exist already. Skipping...")
            return

        dataset = MSMarcoDataset(self.tsv_path, self.embedder.tokenizer, self.max_docs)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
            shuffle=False,
        )

        doc_id_fp = np.lib.format.open_memmap(
            DOCID_PATH,
            mode="w+",
            dtype=np.int64,
            shape=(self.max_docs,),
        )
        embeddings_fp = np.lib.format.open_memmap(
            EMBEDDING_PATH,
            mode="w+",
            dtype=self.embedder.embedding_dtype,
            shape=(self.max_docs, self.dim),
        )

        logger.info(f"Starting embedding generation with {self.num_workers} workers...")

        start_idx = 0
        try:
            for docids, encoded_batch in tqdm(
                dataloader,
                mininterval=10.0,
                desc="Embedding docs",
                total=self.max_docs // self.batch_size,
            ):
                # docids is a tensor of shape (batch_size,)
                # encoded_batch is a dict of tensors of shape (batch_size, seq_len)
                device = self.embedder.model.device
                encoded_batch = {k: v.to(device) for k, v in encoded_batch.items()}

                embeddings = self.embedder.embed_tokenized(encoded_batch)
                end_idx = start_idx + len(docids)
                embeddings_fp[start_idx:end_idx, :] = (
                    embeddings.cpu().float().numpy()
                    if torch.is_tensor(embeddings)
                    else embeddings
                )
                doc_id_fp[start_idx:end_idx] = docids.numpy()

                start_idx = end_idx
        finally:
            del doc_id_fp
            del embeddings_fp

        logger.info(f"Finished. Embeddings written to {EMBEDDING_PATH}")

    @classmethod
    @trace_memory
    def load(cls, mmap: bool = False):
        """
        Args:
            mmap:
                if True, only memory map (memory efficient).
                if False, load everything into memory (faster).
        """
        if not EMBEDDING_PATH.exists() or not DOCID_PATH.exists():
            raise FileNotFoundError(
                f"Embeddings file '{EMBEDDING_PATH}' or doc IDs file '{DOCID_PATH}' not found. "
                "Run `just build-index` to create them first."
            )

        mode: Literal["r+", "r", "w+", "c"] | None = "r" if mmap else None

        embeddings = np.load(EMBEDDING_PATH, mmap_mode=mode)
        ids = np.load(DOCID_PATH, mmap_mode=mode)

        assert embeddings.dtype == np.float32, "IVFPQ requires float32 for embeddings"
        assert ids.dtype == np.int64, "IVFPQ requires int64 for ids"

        return embeddings, ids


if __name__ == "__main__":
    """
    called by `just build-index` shell script
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "max_docs",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    indexer = NumpyIndexer(1, 32, args.max_docs)
    indexer.run()
