from functools import lru_cache
from pathlib import Path
from time import time

import numpy as np
import polars as pl
from backend.logging_config import get_logger
from backend.search_engine.semantic_search.embedding_model import get_embedding_model
from tqdm import tqdm

logger = get_logger(__name__)

MSMARCO_DIR = Path(__file__).resolve().parent / "data"
TSV_PATH = MSMARCO_DIR / "msmarco-docs.tsv"
TARGET_DIR = Path(__file__).resolve().parent.parent / "index" / "bin"
EMBEDDING_PATH = TARGET_DIR / "embeddings.npy"
DOCID_PATH = TARGET_DIR / "doc_ids.npy"
TOTAL_DOCS = 3_213_835


class NumpyIndexer:
    def __init__(
        self,
        batch_size: int = 64,  # depending on device (GPU/CPU)
    ):
        self.tsv_path = TSV_PATH
        self.target_dir = TARGET_DIR

        self.batch_size = batch_size

        self.embedding_path = self.target_dir / "embeddings.npy"
        self.docid_path = self.target_dir / "doc_ids.npy"

        self.embedder = get_embedding_model()
        self.dim = self.embedder.matryoshka_dim
        self.dtype = self.embedder.embedding_dtype

    # TODO if invalid id format, retrieval could fail later
    # skip invalid ids, but then numpy array loaded from file needs to be trimmed
    def _parse_docid(self, docid_str: str) -> int:
        return int(docid_str[1:])  # see index_builder, id starts with 'D'

    def _get_reader(self):
        return pl.scan_csv(
            self.tsv_path, separator="\t", has_header=False
        ).collect_batches(chunk_size=self.batch_size)

    def run(self):
        self.target_dir.mkdir(parents=True, exist_ok=True)
        if EMBEDDING_PATH.exists() or DOCID_PATH.exists():
            logger.info("Embeddings exist already. Skipping...")
            return

        doc_id_fp = np.lib.format.open_memmap(
            DOCID_PATH,
            mode="w+",
            dtype=np.int64,  # FAISS requires int64
            shape=(TOTAL_DOCS,),
        )
        embeddings_fp = np.lib.format.open_memmap(
            EMBEDDING_PATH,
            mode="w+",
            dtype=self.embedder.embedding_dtype,
            shape=(TOTAL_DOCS, self.dim),
        )

        logger.debug(f"Starting to write embeddings to {EMBEDDING_PATH}")

        start_idx = 0

        start = time()
        try:
            for batch in tqdm(
                self._get_reader(), mininterval=30.0, desc="Embedding docs"
            ):
                # colum indices in the tsv
                texts = batch[:, 3].to_list()
                docid_strs = batch[:, 0].to_list()
                docids = [self._parse_docid(id_) for id_ in docid_strs]
                docids = np.array(docids, dtype=np.int64)
                embeddings = self.embedder.embed(texts)
                end_idx = start_idx + len(docids)
                embeddings_fp[start_idx:end_idx, :] = embeddings
                doc_id_fp[start_idx:end_idx] = docids
                start_idx = end_idx
        finally:
            # should happen automatically, but mmaps close when reference gets deleted
            del doc_id_fp
            del embeddings_fp

        logger.debug(
            f"Embeddings written to {EMBEDDING_PATH} ({TOTAL_DOCS} docs, {self.dim} dims), took {time() - start}s"
        )

    @classmethod
    @lru_cache(maxsize=1)
    def load(mmap: bool = False):
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

        mode = "r" if mmap else None

        embeddings = np.load(EMBEDDING_PATH, mmap_mode=mode)
        ids = np.load(DOCID_PATH, mmap_mode=mode)

        assert embeddings.dtype == np.float32, "IVFPQ requires float32 for embeddings"
        assert ids.dtype == np.int64, "IVFPQ requires int64 for ids"

        return embeddings, ids


if __name__ == "__main__":
    """
    called by `just build-index` shell script
    """
    indexer = NumpyIndexer()
    indexer.run()
