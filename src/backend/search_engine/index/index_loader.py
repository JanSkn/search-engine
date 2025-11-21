import os
from functools import lru_cache
from cpp_utils import InvertedIndex  # type: ignore [import-untyped]


def _index_path() -> str:
    return os.path.normpath(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "bin",
        )
    )


@lru_cache(maxsize=1)
def get_index() -> InvertedIndex:
    return InvertedIndex(_index_path())
