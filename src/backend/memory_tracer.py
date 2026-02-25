import gc
import os
from functools import wraps
from pathlib import Path

import numpy as np
import psutil  # type: ignore [import-untyped]

LOG_FILE = Path(__file__).resolve().parent / "memory_log.txt"


def trace_cpp_bindings(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        process = psutil.Process(os.getpid())

        gc.collect()
        mem_before = process.memory_info().rss

        result = func(*args, **kwargs)

        gc.collect()
        mem_after = process.memory_info().rss

        diff = mem_after - mem_before

        with open(LOG_FILE, "a") as f:
            f.write(f"--- Inverted Index (C++): {func.__name__} ---\n")
            f.write(f"Before:       {mem_before / 1024**2:.2f} MB\n")
            f.write(f"After:      {mem_after / 1024**2:.2f} MB\n")
            f.write(f"Delta:      {diff / 1024**2:.2f} MB\n")
            f.write("------------------------------------\n\n")

        return result

    return wrapper


def trace_torch(func):
    """Decorator: misst CPU- und GPU-Speicher und Zeit für Torch-Operationen und schreibt ins Logfile."""
    import time

    import torch

    @wraps(func)
    def wrapper(*args, **kwargs):
        process = os.getpid()
        gc.collect()

        import psutil

        proc = psutil.Process(process)
        mem_before = proc.memory_info().rss

        gpu_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        start_time = time.perf_counter()

        result = func(*args, **kwargs)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end_time = time.perf_counter()
        gc.collect()

        mem_after = proc.memory_info().rss
        gpu_after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        delta_cpu = (mem_after - mem_before) / 1024**2
        delta_gpu = (gpu_after - gpu_before) / 1024**2
        duration = end_time - start_time

        with open(LOG_FILE, "a") as f:
            f.write(f"--- Torch Trace: {func.__name__} ---\n")
            f.write(f"Time:        {duration:.4f}s\n")
            f.write(f"CPU Before:  {mem_before / 1024**2:.2f} MB\n")
            f.write(f"CPU After:   {mem_after / 1024**2:.2f} MB\n")
            f.write(f"CPU Delta:   {delta_cpu:.2f} MB\n")
            if torch.cuda.is_available():
                f.write(f"GPU Before:  {gpu_before / 1024**2:.2f} MB\n")
                f.write(f"GPU After:   {gpu_after / 1024**2:.2f} MB\n")
                f.write(f"GPU Delta:   {delta_gpu:.2f} MB\n")
            f.write("------------------------------------\n\n")

        return result

    return wrapper


def trace_numpy(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        process = psutil.Process(os.getpid())
        gc.collect()
        mem_before = process.memory_info().rss

        result = func(*args, **kwargs)

        gc.collect()
        mem_after = process.memory_info().rss
        diff = mem_after - mem_before

        numpy_mem = 0

        def sum_numpy(obj):
            nonlocal numpy_mem
            if isinstance(obj, np.ndarray):
                numpy_mem += obj.nbytes
            elif isinstance(obj, (list, tuple, dict, set)):
                for v in obj.values() if isinstance(obj, dict) else obj:
                    sum_numpy(v)

        sum_numpy(result)

        with open(LOG_FILE, "a") as f:
            f.write(f"--- NumPy memory trace: {func.__name__} ---\n")
            f.write(f"RAM Before: {mem_before / 1024**2:.2f} MB\n")
            f.write(f"RAM After:  {mem_after / 1024**2:.2f} MB\n")
            f.write(f"RAM Delta:          {diff / 1024**2:.2f} MB\n")
            f.write(f"NumPy arrays total: {numpy_mem / 1024**2:.2f} MB\n")
            f.write("------------------------------------\n\n")

        return result

    return wrapper
