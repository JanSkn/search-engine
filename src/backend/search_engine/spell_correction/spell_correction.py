import time
import torch
from backend.search_engine.spell_correction.spell_corrector import SpellCorrector
from backend.logging_config import get_logger

logger = get_logger(__name__)


def patch_torch_load_for_neuspell() -> None:
    if getattr(torch.load, "__name__", "") == "_torch_load_legacy":
        return

    orig_torch_load = torch.load

    def _torch_load_legacy(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return orig_torch_load(*args, **kwargs)

    torch.load = _torch_load_legacy


patch_torch_load_for_neuspell()


def repl(corrector: SpellCorrector, query: str) -> str | None:
    start = time.perf_counter()
    corrected = corrector.correct(query)
    logger.debug(f"Query spell correction took {time.perf_counter() - start:.6f}")

    if query != corrected:
        return corrected
    else:
        return None
