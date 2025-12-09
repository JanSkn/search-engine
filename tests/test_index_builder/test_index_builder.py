import pytest

from cpp_utils import DocInfo
from backend.search_engine.index.index_loader import get_index

EXPECTED = {
    "alpha": {
        "postings": [0, 1, 4, 7, 9],
        "tf": {
            0: 2,
            1: 3,
            4: 1,
            7: 1,
            9: 1,
        },
        "pos": {
            0: [0, 2],
            1: [0, 1, 2],
            4: [1],
            7: [0],
            9: [0],
        },
    },
    "beta": {
        "postings": [0, 2, 4, 5, 8, 9],
        "tf": {
            0: 2,
            2: 2,
            4: 1,
            5: 1,
            8: 1,
            9: 1,
        },
        "pos": {
            0: [1, 3],
            2: [0, 1],
            4: [2],
            5: [0],
            8: [0],
            9: [1],
        },
    },
    "gamma": {
        "postings": [2, 3, 6, 8, 9],
        "tf": {
            2: 2,
            3: 3,
            6: 1,
            8: 1,
            9: 1,
        },
        "pos": {
            2: [2, 3],
            3: [0, 1, 2],
            6: [0],
            8: [1],
            9: [2],
        },
    },
    "delta": {
        "postings": [3, 4, 6, 7, 9],
        "tf": {
            3: 1,
            4: 1,
            6: 1,
            7: 2,
            9: 1,
        },
        "pos": {
            3: [3],
            4: [0],
            6: [1],
            7: [1, 2],
            9: [3],
        },
    },
}


def test_real_index_metadata():
    inverted_index = get_index()
    metadata = inverted_index.metadata

    expected_num_docs = 10
    # counting title + body
    expected_doc_lengths = {
        0: 6,
        1: 5,
        2: 6,
        3: 6,
        4: 5,
        5: 3,
        6: 4,
        7: 5,
        8: 4,
        9: 6,
    }
    expected_avg_length = sum(expected_doc_lengths.values()) / expected_num_docs

    assert metadata.num_docs == expected_num_docs
    assert metadata.avg_doc_length == expected_avg_length
    assert metadata.doc_lengths == expected_doc_lengths

    for doc_id, length in expected_doc_lengths.items():
        assert metadata.get_doc_length(doc_id) == length


@pytest.mark.parametrize("term", ["alpha", "beta", "gamma", "delta"])
def test_real_index_postings(term):
    inverted_index = get_index()
    result = inverted_index.index.get(term)

    assert result.postings == EXPECTED[term]["postings"]
    assert result.term_frequencies == EXPECTED[term]["tf"]
    assert result.positions == EXPECTED[term]["pos"]

    assert result.postings == sorted(result.postings)
    for doc_id, pos in result.positions.items():
        assert len(pos) == result.term_frequencies[doc_id]


def test_real_index_docstore():
    inverted_index = get_index()
    doc_store = inverted_index.doc_store

    expected_docs = {
        0: DocInfo("http://example.com/0", "Title One"),
        1: DocInfo("http://example.com/1", "Title Two"),
        2: DocInfo("http://example.com/2", "Title Three"),
        3: DocInfo("http://example.com/3", "Title Four"),
        4: DocInfo("http://example.com/4", "Title Five"),
        5: DocInfo("http://example.com/5", "Title Six"),
        6: DocInfo("http://example.com/6", "Title Seven"),
        7: DocInfo("http://example.com/7", "Title Eight"),
        8: DocInfo("http://example.com/8", "Title Nine"),
        9: DocInfo("http://example.com/9", "Title Ten"),
    }

    for doc_id, doc_info in expected_docs.items():
        assert doc_store.get(doc_id).url == doc_info.url
        assert doc_store.get(doc_id).title == doc_info.title
