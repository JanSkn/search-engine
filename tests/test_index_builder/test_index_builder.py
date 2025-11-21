import pytest

import pytest
from backend.search_engine.index.index_loader import get_index

EXPECTED = {
    "alpha": {
        "postings": [1, 2, 5, 8, 10],
        "tf": {
            1: 2, 2: 3, 5: 1, 8: 1, 10: 1,
        },
        "pos": {
            1: [0, 2],
            2: [0, 1, 2],
            5: [1],
            8: [0],
            10: [0],
        },
    },

    "beta": {
        "postings": [1, 3, 5, 6, 9, 10],
        "tf": {
            1: 2, 3: 2, 5: 1, 6: 1, 9: 1, 10: 1,
        },
        "pos": {
            1: [1, 3],
            3: [0, 1],
            5: [2],
            6: [0],
            9: [0],
            10: [1],
        },
    },

    "gamma": {
        "postings": [3, 4, 7, 9, 10],
        "tf": {
            3: 2, 4: 3, 7: 1, 9: 1, 10: 1,
        },
        "pos": {
            3: [2, 3],
            4: [0, 1, 2],
            7: [0],
            9: [1],
            10: [2],
        },
    },

    "delta": {
        "postings": [4, 5, 7, 8, 10],
        "tf": {
            4: 1, 5: 1, 7: 1, 8: 2, 10: 1,
        },
        "pos": {
            4: [3],
            5: [0],
            7: [1],
            8: [1, 2],
            10: [3],
        },
    },
}


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
