from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from backend.logging_config import get_logger

logger = get_logger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_DIR / "models" / "nomic-embed-text"
CACHE_DIR = CACHE_DIR.resolve()
MAX_QUERY_LENGTH = 50


@dataclass
class EmbeddingModel:
    tokenizer: AutoTokenizer
    model: AutoModel
    device: torch.device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("mps")
        if torch.backends.mps.is_built()
        else torch.device("cpu")
    )
    embedding_dtype: np.dtype = np.float32
    matryoshka_dim: int = 64

    @classmethod
    def load(cls) -> EmbeddingModel:
        logger.debug("Loading nomic-embed-text-v1.5 model...")
        start = time.perf_counter()

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = AutoModel.from_pretrained(
            "nomic-ai/nomic-embed-text-v1.5",
            trust_remote_code=True,
            cache_dir=CACHE_DIR,
        )
        model.eval()
        model.to(cls.device)
        logger.debug(
            f"Model loaded on {cls.device} in {time.perf_counter() - start:.4f}s"
        )

        return cls(tokenizer=tokenizer, model=model, device=cls.device)

    def embed_query(
        self,
        q: str,
    ) -> np.ndarray:
        prefixed = "search_query: " + q

        encoded = self.tokenizer(
            prefixed,
            max_length=MAX_QUERY_LENGTH,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            output = self.model(**encoded)

            embeddings = self._mean_pool(
                output.last_hidden_state, encoded["attention_mask"]
            )

            embeddings = F.layer_norm(
                embeddings, normalized_shape=(embeddings.shape[1],)
            )
            embeddings = embeddings[:, : self.matryoshka_dim]
            # for dot product in similarity search
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings.squeeze(0).cpu().numpy()

    def embed_tokenized(self, encoded: dict[str, torch.Tensor]) -> np.ndarray:
        # ensure tensors are on the correct device
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.no_grad():
            output = self.model(**encoded)
            embeddings = self._mean_pool(
                output.last_hidden_state, encoded["attention_mask"]
            )
            embeddings = F.layer_norm(
                embeddings, normalized_shape=(embeddings.shape[1],)
            )
            embeddings = embeddings[:, : self.matryoshka_dim]
            # for dot product in similarity search
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings

    @staticmethod
    def _mean_pool(
        token_embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Remove padding tokens, create mean of all token vectors to get single vector
        for semantic search
        """
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )


@lru_cache(maxsize=1)
def get_embedding_model() -> EmbeddingModel:
    return EmbeddingModel.load()
