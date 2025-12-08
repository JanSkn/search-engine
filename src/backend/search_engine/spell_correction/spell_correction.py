import time
import torch
from .spell_corrector import SpellCorrector

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
    print("NeuSpell SCLSTM spell correction\n")

    #start = time.perf_counter()
    corrected = corrector.correct(query)
    #elapsed_ms = (time.perf_counter() - start) * 1000

    if query != corrected:
        return corrected
    else:
        return None