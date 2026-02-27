from __future__ import annotations

import time
from pathlib import Path

import torch

from backend.search_engine.ltr.model import TinyLTRModel, DEFAULT_FEATURE_ORDER
from backend.search_engine.ltr.build_data.features import (
    build_postings_by_term,
    compute_features_for_doc,
    parse_query_terms,
)
from backend.logging_config import get_logger

logger = get_logger(__name__)

_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "ltr_model.pt"


class LTRReranker:
    def __init__(self, model_path: Path = _MODEL_PATH) -> None:
        logger.info(f"Loading LTR model from {model_path}")
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

        self.feature_order: list[str] = checkpoint.get(
            "feature_order", DEFAULT_FEATURE_ORDER
        )
        hidden: int = checkpoint.get("hidden", 64)
        dropout: float = checkpoint.get("dropout", 0.0)

        self.model = TinyLTRModel(
            num_features=len(self.feature_order),
            hidden=hidden,
            dropout=dropout,
        )
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        logger.info(
            f"LTR model loaded (features={self.feature_order}, hidden={hidden})"
        )

    @torch.no_grad()
    def rerank(
        self,
        ranked_candidates: list[tuple[int, float]],
        raw_query: str,
        inverted_index,
    ) -> list[tuple[int, float]]:
        """Re-rank BM25 candidates using the LTR model.

        Uses ``parse_query_terms`` from the training pipeline so that
        feature computation is identical to how the training data was built.
        """
        if not ranked_candidates:
            return []

        start = time.perf_counter()

        # same tokenisation as training (no stop-word removal, no boolean ops)
        query_terms = parse_query_terms(raw_query)
        if not query_terms:
            return ranked_candidates

        postings_by_term = build_postings_by_term(inverted_index, query_terms)

        features_list: list[list[float]] = []
        doc_ids: list[int] = []
        for doc_id, _ in ranked_candidates:
            doc_id = int(doc_id)
            fv = compute_features_for_doc(
                inverted_index,
                doc_id=doc_id,
                query_terms=query_terms,
                postings_by_term=postings_by_term,
            )
            feat_dict = fv.as_dict()
            features_list.append(
                [float(feat_dict.get(name, 0.0)) for name in self.feature_order]
            )
            doc_ids.append(doc_id)

        x = torch.tensor([features_list], dtype=torch.float32)  # (1, L, F)
        mask = torch.ones(1, len(doc_ids), dtype=torch.bool)

        scores = self.model(x, mask).squeeze(0)  # (L,)

        result = sorted(
            zip(doc_ids, scores.tolist()),
            key=lambda pair: pair[1],
            reverse=True,
        )

        logger.debug(
            f"LTR rerank time: {time.perf_counter() - start:.6f}s "
            f"({len(doc_ids)} docs)"
        )
        return result


_reranker: LTRReranker | None = None


def get_reranker() -> LTRReranker:
    global _reranker
    if _reranker is None:
        _reranker = LTRReranker()
    return _reranker
