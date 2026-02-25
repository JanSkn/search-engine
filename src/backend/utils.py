import functools
import os
import time

from backend.logging_config import get_logger

logger = get_logger(__name__)


class TempOMPThreads:
    """Temporarily set OMP_NUM_THREADS to a given value."""

    def __init__(self, num_threads: int):
        self.num_threads = str(num_threads)
        self.original = os.environ.get("OMP_NUM_THREADS", None)

    def __enter__(self):
        os.environ["OMP_NUM_THREADS"] = self.num_threads

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.original is None:
            del os.environ["OMP_NUM_THREADS"]
        else:
            os.environ["OMP_NUM_THREADS"] = self.original


def measure_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        duration = end - start

        if duration < 1:
            logger.debug(f"{func.__name__} took {duration * 1000:.2f} ms")
        elif duration < 60:
            logger.debug(f"{func.__name__} took {duration:.2f} s")
        else:
            minutes = int(duration // 60)
            seconds = duration % 60
            logger.debug(f"{func.__name__} took {minutes}m {seconds:.2f}s")

        return result

    return wrapper
