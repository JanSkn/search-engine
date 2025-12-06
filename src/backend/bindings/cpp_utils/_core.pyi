"""
CPP utils for search engine
"""
from __future__ import annotations
import typing
__all__: list[str] = ['DocInfo', 'DocStore', 'Metadata', 'IndexAccessor', 'InvertedIndex', 'PostingList', 'normalize_search_query', 'positional_intersect', 'find_docs']
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
class Metadata:
    @property
    def num_docs(self) -> int: ...
    @property
    def avg_doc_length(self) -> float: ...
    @property
    def doc_lengths(self) -> dict[int, int]: ...
    def get_doc_length(self, doc_id: int) -> int: ...
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
def normalize_search_query(text: str) -> list[str]:
    """
    Normalize and stem search query into tokens, but keep logical operators and parentheses as is
    """
def positional_intersect(pl1: PostingList, pl2: PostingList, distance: int = 1) -> PostingList:
    """
    Positional intersection of two posting lists with given distance
    """
def find_docs(pl1: PostingList, pl2: PostingList, mode: typing.Literal["AND", "OR", "NOT"]) -> PostingList:
    """
    Find documents that are in both posting lists
    """