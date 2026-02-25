import os
import time
from functools import lru_cache

from cpp_utils import InvertedIndex  # type: ignore [import-untyped]

from backend.logging_config import get_logger
from backend.memory_tracer import trace_cpp_bindings

logger = get_logger(__name__)


def _index_path() -> str:
    return os.path.normpath(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "bin",
        )
    )


@trace_cpp_bindings
@lru_cache(maxsize=1)
def get_index() -> InvertedIndex:
    logger.debug(f"Loading inverted index from {_index_path()}...")
    start = time.perf_counter()
    index = InvertedIndex(_index_path())
    logger.debug(f"Inverted index loaded in {time.perf_counter() - start:.6f}s")
    return index
