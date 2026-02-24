from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from backend.logging_config import get_logger

logger = get_logger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_DIR / "models" / "nomic-embed-text"
CACHE_DIR = CACHE_DIR.resolve()

# see https://huggingface.co/nomic-ai/nomic-embed-text-v1.5
TASK_PREFIX: dict[str, str] = {
    "query": "search_query: ",
    "document": "search_document: ",
}


@dataclass
class EmbeddingModel:
    tokenizer: AutoTokenizer
    model: AutoModel
    device: torch.device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("mps")
        if torch.has_mps
        else torch.device("cpu")
    )
    embedding_dtype: np.dtype = np.float32
    matryoshka_dim: int = 64

    @classmethod
    def load(cls) -> EmbeddingModel:
        logger.debug("Loading nomic-embed-text-v1.5 model...")
        start = time.perf_counter()

        tokenizer = AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v1.5")
        model = AutoModel.from_pretrained(
            "nomic-ai/nomic-embed-text-v1.5",
            trust_remote_code=True,
            cache_dir=CACHE_DIR,
        )
        model.to(cls.device)
        model.eval()
        logger.debug(
            f"Model loaded on {cls.device} in {time.perf_counter() - start:.4f}s"
        )

        return cls(tokenizer=tokenizer, model=model, device=cls.device)

    def embed(
        self,
        texts: list[str],
        task: Literal["query", "document"] = "document",
    ) -> np.ndarray:
        """
        Returns:
            C-contiguous float32 numpy array of shape (len(texts), matryoshka_dim)
        """
        prefix = TASK_PREFIX[task]
        prefixed = [prefix + t for t in texts if t is not None]

        # MSMarco avg. tokens per doc: 1131
        # truncate edge cases
        encoded = self.tokenizer(
            prefixed,
            padding=True,  # for batching
            truncation=True,  # truncate too long texts
            max_length=8192,  # Nomic's max context
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            output = self.model(**encoded)

        embeddings = self._mean_pool(
            output.last_hidden_state, encoded["attention_mask"]
        )
        embeddings = F.layer_norm(embeddings, normalized_shape=(embeddings.shape[1],))
        assert self.matryoshka_dim <= embeddings.shape[1], (
            f"matryoshka_dim={self.matryoshka_dim} greater than model output "
            f"dimension={embeddings.shape[1]}"
        )
        embeddings = embeddings[:, : self.matryoshka_dim]
        # for dot product in similarity search
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy().astype(np.float32)

    @staticmethod
    def _mean_pool(
        token_embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Remove padding tokens, create mean of all token vectors to get single vector
        for semantic search
        """
        mask_expanded = attention_mask.unsqueeze(-1).float()
        summed = (token_embeddings * mask_expanded).sum(dim=1)
        counts = mask_expanded.sum(dim=1).clamp(min=1e-9)
        return summed / counts


@lru_cache(maxsize=1)
def get_embedding_model() -> EmbeddingModel:
    return EmbeddingModel.load()
