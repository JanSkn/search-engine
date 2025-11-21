from __future__ import annotations

from ._core import (
    DocInfo,
    DocStore,
    InvertedIndex,
    PostingList,
    normalize_search_query,
    list_union,
    list_diff,
)

__all__ = [
    "InvertedIndex",
    "PostingList",
    "DocStore",
    "DocInfo",
    "normalize_search_query",
    "list_union",
    "list_diff",
]