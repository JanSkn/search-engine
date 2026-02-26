import gc
import os
import time
from functools import wraps
from pathlib import Path

import numpy as np
import psutil  # type: ignore [import-untyped]

LOG_FILE = Path(__file__).resolve().parent / "memory_log.txt"

MB = 1024**2


def _object_size_mb(obj) -> tuple[str, float]:
    """Measure the actual memory footprint of a returned object."""
    import torch

    # torch model (nn.Module or dataclass containing one)
    model = None
    if isinstance(obj, torch.nn.Module):
        model = obj
    elif hasattr(obj, "model") and isinstance(getattr(obj, "model"), torch.nn.Module):
        model = obj.model

    if model is not None:
        params = sum(p.numel() * p.element_size() for p in model.parameters())
        buffers = sum(b.numel() * b.element_size() for b in model.buffers())
        total = (params + buffers) / MB
        return "torch_model", total

    # faiss index
    try:
        import faiss  # type: ignore [import-untyped]

        if isinstance(obj, faiss.Index):
            # write to a temporary buffer to measure serialized size
            writer = faiss.VectorIOWriter()
            faiss.write_index(obj, writer)
            total = len(faiss.vector_to_array(writer.data)) / MB
            return "faiss_index", total
    except ImportError:
        pass

    # numpy array
    if isinstance(obj, np.ndarray):
        return "numpy", obj.nbytes / MB

    return "unknown", 0.0


def trace_memory(func):
    """Universal memory tracer: measures RSS delta + actual object size."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        proc = psutil.Process(os.getpid())
        gc.collect()

        mem_before = proc.memory_info().rss
        start_time = time.perf_counter()

        result = func(*args, **kwargs)

        duration = time.perf_counter() - start_time
        gc.collect()
        if not LOG_FILE.exists() or LOG_FILE.stat().st_size == 0:
            with open(LOG_FILE, "a") as f:
                f.write(f"Process PID: {os.getpid()}\n")
                f.write("------------------------------------\n\n")

        mem_after = proc.memory_info().rss

        obj_type, obj_size = _object_size_mb(result)

        with open(LOG_FILE, "a") as f:
            f.write(f"--- {func.__name__} ---\n")
            f.write(f"Time:          {duration:.4f}s\n")
            f.write(f"RSS before:    {mem_before / MB:.2f} MB\n")
            f.write(f"RSS after:     {mem_after / MB:.2f} MB\n")
            f.write(f"RSS delta:     {(mem_after - mem_before) / MB:.2f} MB\n")
            if obj_size > 0:
                f.write(f"Object type:   {obj_type}\n")
                f.write(f"Object size:   {obj_size:.2f} MB\n")
            f.write("------------------------------------\n\n")

        return result

    return wrapper
