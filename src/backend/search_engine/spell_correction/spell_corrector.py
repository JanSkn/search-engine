from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from neuspell import SclstmChecker  # type: ignore [import-untyped]

from backend.logging_config import get_logger
from backend.memory_tracer import trace_memory

logger = get_logger(__name__)

_THIS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = _THIS_DIR.parent
CHECKPOINT_DIR = PROJECT_DIR / "models" / "neuspell-scrnn-probwordnoise"
MODEL_PATH = CHECKPOINT_DIR.resolve()
HF_REPO_ID = "pszemraj/neuspell-scrnn-probwordnoise"


def _ensure_model_exists():
    """Ensure the checkpoint directory exists, otherwise download it from HF Hub."""
    if MODEL_PATH.exists():
        logger.debug(f"Model directory exists: {MODEL_PATH}")
        return

    logger.warning(f"Model directory missing. Downloading from {HF_REPO_ID}...")
    try:
        snapshot_download(
            repo_id=HF_REPO_ID,
            local_dir=str(MODEL_PATH),
        )
        logger.info("Model successfully downloaded.")
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        raise RuntimeError("Model could not be downloaded automatically.") from e


@dataclass
class SpellCorrector:
    checker: SclstmChecker

    @classmethod
    def load(cls) -> SpellCorrector:
        _ensure_model_exists()
        logger.debug("Load neuspell SCLSTM checker...")
        start = time.perf_counter()
        checker = SclstmChecker()

        # monkey-patch the load to use weights_only=False for this specific model
        original_load = torch.load

        def patched_load(*args, **kwargs):
            # force weights_only=False for trusted checkpoint
            kwargs["weights_only"] = False
            return original_load(*args, **kwargs)

        # temporarily replace torch.load
        torch.load = patched_load
        try:
            checker.from_pretrained(MODEL_PATH)
        finally:
            # restore original torch.load
            torch.load = original_load

        logger.debug(f"Model loaded in {time.perf_counter() - start:.6f}s")
        return cls(checker=checker)

    def correct(self, text: str) -> str:
        return self.checker.correct(text)


@lru_cache(maxsize=1)
@trace_memory
def get_spell_corrector() -> SpellCorrector:
    return SpellCorrector.load()
