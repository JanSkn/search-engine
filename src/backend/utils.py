import functools
import time

from backend.logging_config import get_logger

logger = get_logger(__name__)


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
