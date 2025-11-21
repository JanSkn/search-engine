"""
CPP utils for search engine
"""
from __future__ import annotations
import typing
__all__: list[str] = ['DocInfo', 'DocStore', 'IndexAccessor', 'InvertedIndex', 'PostingList', 'list_diff', 'list_union', 'normalize_search_query']
class DocInfo:
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, url: str, title: str) -> None:
        ...
    @property
    def title(self) -> str:
        ...
    @property
    def url(self) -> str:
        ...
class DocStore:
    def get(self, doc_id: int) -> DocInfo | None:
        ...
class IndexAccessor:
    def get(self, term: str) -> PostingList | None:
        ...
class InvertedIndex:
    def __init__(self, arg0: str) -> None:
        ...
    @property
    def doc_store(self) -> DocStore:
        ...
    @property
    def index(self) -> IndexAccessor:
        ...
class PostingList:
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, postings: list[int], term_frequencies: dict[int, int], positions: dict[int, list[int]]) -> None:
        ...
    def build_skip_pointers(self) -> None:
        ...
    @property
    def positions(self) -> dict[int, list[int]]:
        ...
    @property
    def postings(self) -> list[int]:
        ...
    @property
    def skip_pointers(self) -> dict[int, int]:
        ...
    @property
    def term_frequencies(self) -> dict[int, int]:
        ...
def list_diff(postings_1: list[int], postings_2: list[int]) -> list[int]:
    """
    Difference of two sorted posting lists (postings_1 - postings_2)
    """
def list_union(postings_1: list[int], postings_2: list[int]) -> list[int]:
    """
    Union of two sorted posting lists
    """
def normalize_search_query(text: str) -> list[str]:
    """
    Normalize and stem search query into tokens, but keep logical operators and parentheses as is
    """
