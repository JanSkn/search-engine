import pytest

import pytest
from backend.search_engine.index.index_loader import get_index

EXPECTED = {
    "alpha": {
        "postings": [0, 1, 4, 7, 9],
        "tf": {
            0: 2, 1: 3, 4: 1, 7: 1, 9: 1,
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
            0: 2, 2: 2, 4: 1, 5: 1, 8: 1, 9: 1,
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
            2: 2, 3: 3, 6: 1, 8: 1, 9: 1,
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
            3: 1, 4: 1, 6: 1, 7: 2, 9: 1,
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
