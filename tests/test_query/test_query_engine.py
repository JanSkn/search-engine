import pytest
from unittest.mock import Mock, patch
from cpp_utils import (  # type: ignore [import-untyped]
    PostingList,
    DocInfo,
    normalize_search_query,
    find_docs,
    positional_intersect,
)
from backend.search_engine.models.index import SearchResult
from backend.search_engine.query.query_engine import QueryEngine
from backend.search_engine.query.query_preprocessing import Node


@pytest.fixture
def mock_query_tree():
    with patch("backend.search_engine.query.query_engine.QueryTree") as mock_qt_class:
        mock_qt_instance = Mock()
        mock_qt_class.return_value = mock_qt_instance
        mock_qt_instance.root = Node(value="test")
        yield mock_qt_instance


@pytest.fixture
def mock_inverted_index():
    mock_index_instance = Mock()
    mock_index_instance.index = Mock()
    mock_index_instance.doc_store = Mock()

    with patch(
        "backend.search_engine.query.query_engine.get_index"
    ) as mock_get_index_function:
        mock_get_index_function.return_value = mock_index_instance
        yield mock_index_instance


class TestPositionalIntersect:
    def test_positional_intersect_basic(self):
        postings1 = PostingList(
            postings=[1, 2, 3],
            term_frequencies={1: 2, 2: 1, 3: 2},
            positions={1: [0, 5], 2: [3], 3: [1, 7]},
        )
        postings2 = PostingList(
            postings=[1, 2, 3],
            term_frequencies={1: 1, 2: 1, 3: 1},
            positions={1: [1], 2: [4], 3: [2]},
        )
        postings1.build_skip_pointers()
        postings2.build_skip_pointers()

        result = positional_intersect(postings1, postings2, distance=1)

        # Doc 1: term1 at [0,5], term2 at [1] -> match at position 0
        # Doc 2: term1 at [3], term2 at [4] -> match at position 3
        # Doc 3: term1 at [1,7], term2 at [2] -> match at position 1
        expected = [1, 2, 3]
        assert result.postings == expected
        assert result.term_frequencies[1] == 1
        assert result.term_frequencies[2] == 1
        assert result.term_frequencies[3] == 1

    def test_positional_intersect_no_match(self):
        postings1 = PostingList(
            postings=[1, 2], term_frequencies={1: 1, 2: 1}, positions={1: [0], 2: [5]}
        )
        postings2 = PostingList(
            postings=[1, 2], term_frequencies={1: 1, 2: 1}, positions={1: [5], 2: [0]}
        )
        postings1.build_skip_pointers()
        postings2.build_skip_pointers()

        result = positional_intersect(postings1, postings2, distance=1)

        assert len(result.postings) == 0
        assert len(result.term_frequencies) == 0
        assert len(result.positions) == 0

    def test_positional_intersect_multiple_matches_same_doc(self):
        postings1 = PostingList(
            postings=[1], term_frequencies={1: 3}, positions={1: [0, 3, 6]}
        )
        postings2 = PostingList(
            postings=[1], term_frequencies={1: 3}, positions={1: [1, 4, 7]}
        )
        postings1.build_skip_pointers()
        postings2.build_skip_pointers()

        result = positional_intersect(postings1, postings2, distance=1)

        expected = [1]
        assert result.postings == expected
        assert result.term_frequencies[1] == 3  # All three positions match
        assert result.positions[1] == [0, 3, 6]

    def test_positional_intersect_different_distance(self):
        postings1 = PostingList(
            postings=[1, 2],
            term_frequencies={1: 2, 2: 1},
            positions={1: [0, 5], 2: [10]},
        )
        postings2 = PostingList(
            postings=[1, 2], term_frequencies={1: 1, 2: 1}, positions={1: [3], 2: [15]}
        )
        postings1.build_skip_pointers()
        postings2.build_skip_pointers()

        result = positional_intersect(postings1, postings2, distance=3)

        # Doc 1: term1 at [0,5], term2 at [3] -> match at position 0 (0+3=3)
        # Doc 2: term1 at [10], term2 at [15] -> no match (10+3=13≠15)
        expected = [1]
        assert result.postings == expected
        assert result.term_frequencies[1] == 1

    def test_positional_intersect_empty_postings(self):
        postings1 = PostingList(postings=[], term_frequencies={}, positions={})
        postings2 = PostingList(
            postings=[1, 2], term_frequencies={1: 1, 2: 1}, positions={1: [0], 2: [1]}
        )
        postings1.build_skip_pointers()
        postings2.build_skip_pointers()

        result = positional_intersect(postings1, postings2, distance=1)

        assert len(result.postings) == 0


class TestPositionalPhraseSearch:
    def test_phrase_search_basic(self, mock_inverted_index):
        postings1 = PostingList(
            postings=[1, 2, 3],
            term_frequencies={1: 1, 2: 1, 3: 1},
            positions={1: [0], 2: [5], 3: [10]},
        )
        postings2 = PostingList(
            postings=[1, 2, 3],
            term_frequencies={1: 1, 2: 1, 3: 1},
            positions={1: [1], 2: [6], 3: [15]},
        )

        def get_side_effect(term):
            if term == "hello":
                return postings1
            elif term == "world":
                return postings2
            return None

        mock_inverted_index.index.get.side_effect = get_side_effect

        qe = QueryEngine("'hello world'")
        result = qe._positional_phrase_search(["hello", "world"])

        # Doc 1: "hello" at 0, "world" at 1 -> match
        # Doc 2: "hello" at 5, "world" at 6 -> match
        # Doc 3: no match (10+1 ≠ 15)
        expected = [1, 2]
        assert result.postings == expected

    def test_phrase_search_three_terms(self, mock_inverted_index):
        postings1 = PostingList(
            postings=[1, 2], term_frequencies={1: 1, 2: 1}, positions={1: [0], 2: [5]}
        )
        postings2 = PostingList(
            postings=[1, 2], term_frequencies={1: 1, 2: 1}, positions={1: [1], 2: [6]}
        )
        postings3 = PostingList(
            postings=[1, 2], term_frequencies={1: 1, 2: 1}, positions={1: [2], 2: [8]}
        )

        def get_side_effect(term):
            if term == "the":
                return postings1
            elif term == "quick":
                return postings2
            elif term == "fox":
                return postings3
            return None

        mock_inverted_index.index.get.side_effect = get_side_effect

        qe = QueryEngine("'the quick fox'")
        result = qe._positional_phrase_search(["the", "quick", "fox"])

        # Doc 1: "the" at 0, "quick" at 1, "fox" at 2 -> match
        # Doc 2: "the" at 5, "quick" at 6, "fox" at 8 -> no match (7≠8)
        expected = [1]
        assert result.postings == expected

    def test_phrase_search_term_not_found(self, mock_inverted_index):
        postings1 = PostingList(
            postings=[1, 2], term_frequencies={1: 1, 2: 1}, positions={1: [0], 2: [5]}
        )

        def get_side_effect(term):
            if term == "hello":
                return postings1
            elif term == "nonexistent":
                return None
            return None

        mock_inverted_index.index.get.side_effect = get_side_effect

        qe = QueryEngine("hello nonexistent")
        result = qe._positional_phrase_search(["hello", "nonexistent"])

        assert len(result.postings) == 0

    def test_phrase_search_empty_query(self, mock_inverted_index):
        qe = QueryEngine("")
        result = qe._positional_phrase_search([])

        assert len(result.postings) == 0
        assert len(result.term_frequencies) == 0
        assert len(result.positions) == 0

    def test_phrase_search_single_term(self, mock_inverted_index):
        postings = PostingList(
            postings=[1, 2, 3],
            term_frequencies={1: 1, 2: 1, 3: 1},
            positions={1: [0], 2: [5], 3: [10]},
        )

        mock_inverted_index.index.get.return_value = postings

        qe = QueryEngine("'hello'")
        result = qe._positional_phrase_search(["hello"])

        expected = [1, 2, 3]
        assert result.postings == expected

    def test_phrase_search_no_positional_match(self, mock_inverted_index):
        postings1 = PostingList(
            postings=[1, 2],
            term_frequencies={1: 2, 2: 2},
            positions={1: [0, 10], 2: [5, 15]},
        )
        postings2 = PostingList(
            postings=[1, 2],
            term_frequencies={1: 2, 2: 2},
            positions={1: [5, 20], 2: [8, 25]},
        )

        def get_side_effect(term):
            if term == "word1":
                return postings1
            elif term == "word2":
                return postings2
            return None

        mock_inverted_index.index.get.side_effect = get_side_effect

        qe = QueryEngine("word1 word2")
        result = qe._positional_phrase_search(["word1", "word2"])

        # Neither doc has consecutive positions
        assert len(result.postings) == 0


class TestFindDocsUtils:
    def test_normalize_search_query_phrase_query(self):
        text = "The quick brown fox is running!"
        result = normalize_search_query(text)
        expected = ["the", "quick", "brown", "fox", "is", "run"]
        assert result == expected

    def test_normalize_search_query_bool_query(self):
        text = "(Cats AND dogs) OR birds"
        result = normalize_search_query(text)
        expected = ["(", "cat", "AND", "dog", ")", "OR", "bird"]
        assert result == expected


class TestFindDocsAND:
    def test_and_with_common_elements(self):
        postings1 = PostingList(
            postings=[1, 3, 5, 7, 9],
            term_frequencies={1: 2, 3: 1, 5: 3, 7: 1, 9: 2},
            positions={1: [0, 5], 3: [2], 5: [1, 3, 7], 7: [4], 9: [6, 8]},
        )
        postings2 = PostingList(
            postings=[2, 3, 5, 8, 10],
            term_frequencies={2: 1, 3: 2, 5: 1, 8: 1, 10: 1},
            positions={2: [1], 3: [0, 4], 5: [2], 8: [3], 10: [5]},
        )
        result = find_docs(postings1, postings2, "AND")

        expected = [3, 5]
        assert result.postings == expected
        assert result.term_frequencies[3] == 3
        assert result.term_frequencies[5] == 4

    def test_and_with_no_common_elements(self):
        postings1 = PostingList(
            postings=[1, 3, 5],
            term_frequencies={1: 1, 3: 1, 5: 1},
            positions={1: [0], 3: [1], 5: [2]},
        )
        postings2 = PostingList(
            postings=[2, 4, 6],
            term_frequencies={2: 1, 4: 1, 6: 1},
            positions={2: [0], 4: [1], 6: [2]},
        )
        postings1.build_skip_pointers()
        postings2.build_skip_pointers()

        result = find_docs(postings1, postings2, "AND")

        assert len(result.postings) == 0
        assert len(result.term_frequencies) == 0
        assert len(result.positions) == 0

    def test_and_with_all_common_elements(self):
        postings1 = PostingList(
            postings=[1, 2, 3],
            term_frequencies={1: 2, 2: 1, 3: 3},
            positions={1: [0, 5], 2: [1], 3: [2, 4, 6]},
        )
        postings2 = PostingList(
            postings=[1, 2, 3],
            term_frequencies={1: 1, 2: 2, 3: 1},
            positions={1: [3], 2: [7, 9], 3: [8]},
        )
        postings1.build_skip_pointers()
        postings2.build_skip_pointers()

        result = find_docs(postings1, postings2, "AND")

        expected = [1, 2, 3]
        assert result.postings == expected
        assert result.term_frequencies[1] == 3
        assert result.term_frequencies[2] == 3
        assert result.term_frequencies[3] == 4

    def test_and_skip_pointers_optimization(self):
        postings1 = PostingList(
            postings=[1, 5, 10, 15, 20, 25, 30],
            term_frequencies={1: 1, 5: 1, 10: 1, 15: 1, 20: 1, 25: 1, 30: 1},
            positions={i: [0] for i in [1, 5, 10, 15, 20, 25, 30]},
        )
        postings2 = PostingList(
            postings=[3, 5, 7, 10, 12, 15, 18, 20],
            term_frequencies={3: 1, 5: 1, 7: 1, 10: 1, 12: 1, 15: 1, 18: 1, 20: 1},
            positions={i: [0] for i in [3, 5, 7, 10, 12, 15, 18, 20]},
        )
        postings1.build_skip_pointers()
        postings2.build_skip_pointers()

        result = find_docs(postings1, postings2, "AND")

        expected = [5, 10, 15, 20]
        assert result.postings == expected


class TestFindDocsOR:
    def test_or_with_common_elements(self):
        postings1 = PostingList(
            postings=[1, 3, 5, 7, 9],
            term_frequencies={1: 1, 3: 2, 5: 1, 7: 1, 9: 1},
            positions={1: [0], 3: [1, 3], 5: [2], 7: [4], 9: [5]},
        )
        postings2 = PostingList(
            postings=[2, 3, 5, 8, 10],
            term_frequencies={2: 1, 3: 1, 5: 2, 8: 1, 10: 1},
            positions={2: [0], 3: [6], 5: [7, 9], 8: [8], 10: [10]},
        )
        result = find_docs(postings1, postings2, "OR")

        expected = [1, 2, 3, 5, 7, 8, 9, 10]
        assert result.postings == expected
        assert result.term_frequencies[3] == 3
        assert result.term_frequencies[5] == 3

    def test_or_with_no_common_elements(self):
        postings1 = PostingList(
            postings=[1, 3, 5],
            term_frequencies={1: 1, 3: 1, 5: 1},
            positions={1: [0], 3: [1], 5: [2]},
        )
        postings2 = PostingList(
            postings=[2, 4, 6],
            term_frequencies={2: 1, 4: 1, 6: 1},
            positions={2: [0], 4: [1], 6: [2]},
        )
        postings1.build_skip_pointers()
        postings2.build_skip_pointers()

        result = find_docs(postings1, postings2, "OR")

        expected = [1, 2, 3, 4, 5, 6]
        assert result.postings == expected

    def test_or_with_all_common_elements(self):
        postings1 = PostingList(
            postings=[1, 2, 3],
            term_frequencies={1: 1, 2: 1, 3: 1},
            positions={1: [0], 2: [1], 3: [2]},
        )
        postings2 = PostingList(
            postings=[1, 2, 3],
            term_frequencies={1: 1, 2: 1, 3: 1},
            positions={1: [3], 2: [4], 3: [5]},
        )
        postings1.build_skip_pointers()
        postings2.build_skip_pointers()

        result = find_docs(postings1, postings2, "OR")

        expected = [1, 2, 3]
        assert result.postings == expected


class TestFindDocsNOT:
    def test_not_operation_basic_difference(self):
        postings1 = PostingList(
            postings=[2, 4, 6, 8],
            term_frequencies={2: 1, 4: 1, 6: 1, 8: 1},
            positions={},
        )
        postings2 = PostingList(
            postings=[4, 8, 10], term_frequencies={4: 1, 8: 1, 10: 1}, positions={}
        )

        postings1.build_skip_pointers()
        postings2.build_skip_pointers()

        result = find_docs(postings1, postings2, "NOT")

        expected = [2, 6]
        assert result.postings == expected
        assert set(result.term_frequencies.keys()) == {2, 6}

    def test_not_with_empty_right(self):
        postings1 = PostingList(
            postings=[1, 2, 3, 4],
            term_frequencies={1: 1, 2: 1, 3: 1, 4: 1},
            positions={},
        )
        postings2 = PostingList(postings=[], term_frequencies={}, positions={})

        postings1.build_skip_pointers()
        postings2.build_skip_pointers()

        result = find_docs(postings1, postings2, "NOT")

        expected = [1, 2, 3, 4]
        assert result.postings == expected
        assert len(result.term_frequencies) == 4

    def test_not_with_empty_left(self):
        postings1 = PostingList(postings=[], term_frequencies={}, positions={})
        postings2 = PostingList(
            postings=[1, 2, 3, 4],
            term_frequencies={1: 1, 2: 1, 3: 1, 4: 1},
            positions={},
        )

        postings1.build_skip_pointers()
        postings2.build_skip_pointers()

        result = find_docs(postings1, postings2, "NOT")

        expected = []
        assert result.postings == expected
        assert len(result.term_frequencies) == 0


class TestEvaluate:
    def test_evaluate_leaf_node(self, mock_inverted_index):
        posting_list = PostingList(
            postings=[1, 2, 3],
            term_frequencies={1: 1, 2: 1, 3: 1},
            positions={1: [0], 2: [1], 3: [2]},
        )
        mock_inverted_index.index.get.return_value = posting_list

        node = Node(value="test")
        result = QueryEngine("")._bool_search(node)

        mock_inverted_index.index.get.assert_called_once_with("test")
        assert result == posting_list

    def test_evaluate_and_node(self, mock_inverted_index):
        mock_inverted_index.index.get.side_effect = [
            PostingList(
                postings=[1, 2, 3],
                term_frequencies={1: 1, 2: 1, 3: 1},
                positions={1: [0], 2: [1], 3: [2]},
            ),
            PostingList(
                postings=[2, 3, 4],
                term_frequencies={2: 1, 3: 1, 4: 1},
                positions={2: [0], 3: [1], 4: [2]},
            ),
        ]

        left_node = Node(value="term1")
        right_node = Node(value="term2")
        and_node = Node(value="AND", left=left_node, right=right_node)

        result = QueryEngine("")._bool_search(and_node)

        expected = [2, 3]
        assert result.postings == expected

    def test_evaluate_or_node(self, mock_inverted_index):
        mock_inverted_index.index.get.side_effect = [
            PostingList(
                postings=[1, 2],
                term_frequencies={1: 1, 2: 1},
                positions={1: [0], 2: [1]},
            ),
            PostingList(
                postings=[3, 4],
                term_frequencies={3: 1, 4: 1},
                positions={3: [0], 4: [1]},
            ),
        ]

        left_node = Node(value="term1")
        right_node = Node(value="term2")
        or_node = Node(value="OR", left=left_node, right=right_node)

        result = QueryEngine("")._bool_search(or_node)

        expected = [1, 2, 3, 4]
        assert result.postings == expected

    def test_evaluate_and_not_node(self, mock_inverted_index):
        postings1 = PostingList(
            postings=[1, 2, 3, 4, 5],
            term_frequencies={i: 1 for i in [1, 2, 3, 4, 5]},
            positions={},
        )
        postings2 = PostingList(
            postings=[3, 4, 5, 6, 7],
            term_frequencies={i: 1 for i in [3, 4, 5, 6, 7]},
            positions={},
        )

        def get_side_effect(term):
            if term == "A":
                return postings1
            elif term == "B":
                return postings2
            return None

        mock_inverted_index.index.get.side_effect = get_side_effect

        postings1.build_skip_pointers()
        postings2.build_skip_pointers()

        node_a = Node(value="A")
        node_b = Node(value="B")
        not_b = Node(value="NOT", right=node_b)
        root = Node(value="AND", left=node_a, right=not_b)

        result = QueryEngine("")._bool_search(root)

        expected = [1, 2]
        assert result.postings == expected
        assert set(result.term_frequencies.keys()) == {1, 2}

    def test_evaluate_complex_query(self, mock_inverted_index):
        # (term1 AND term2) OR term3
        mock_inverted_index.index.get.side_effect = [
            PostingList(
                postings=[1, 2, 3],
                term_frequencies={1: 1, 2: 1, 3: 1},
                positions={1: [0], 2: [1], 3: [2]},
            ),
            PostingList(
                postings=[2, 3, 4],
                term_frequencies={2: 1, 3: 1, 4: 1},
                positions={2: [0], 3: [1], 4: [2]},
            ),
            PostingList(
                postings=[5, 6],
                term_frequencies={5: 1, 6: 1},
                positions={5: [0], 6: [1]},
            ),
        ]

        term1 = Node(value="term1")
        term2 = Node(value="term2")
        and_node = Node(value="AND", left=term1, right=term2)
        term3 = Node(value="term3")
        or_node = Node(value="OR", left=and_node, right=term3)

        result = QueryEngine("")._bool_search(or_node)

        expected = [2, 3, 5, 6]
        assert result.postings == expected


class TestSearchResults:
    def test_search_results_basic(self, mock_query_tree, mock_inverted_index):
        mock_inverted_index.index.get.return_value = PostingList(
            postings=[1, 2, 3],
            term_frequencies={1: 1, 2: 1, 3: 1},
            positions={1: [0], 2: [1], 3: [2]},
        )
        mock_inverted_index.doc_store.get.side_effect = [
            DocInfo(url="http://example.com/1", title="Doc 1"),
            DocInfo(url="http://example.com/2", title="Doc 2"),
            DocInfo(url="http://example.com/3", title="Doc 3"),
        ]

        qe = QueryEngine("test")
        with patch(
            "backend.search_engine.query.query_engine.normalize_search_query",
            return_value=["test"],
        ):
            results = qe.search_results(limit=10)

        assert len(results) == 3
        assert all(isinstance(r, SearchResult) for r in results)

    def test_search_results_with_limit(self, mock_query_tree, mock_inverted_index):
        mock_inverted_index.index.get.return_value = PostingList(
            postings=[1, 2, 3, 4, 5],
            term_frequencies={i: 1 for i in range(1, 6)},
            positions={i: [0] for i in range(1, 6)},
        )
        mock_inverted_index.doc_store.get.side_effect = [
            DocInfo(url=f"http://example.com/{i}", title=f"Doc {i}")
            for i in range(1, 6)
        ]

        qe = QueryEngine("test")
        with patch(
            "backend.search_engine.query.query_engine.normalize_search_query",
            return_value=["test"],
        ):
            results = qe.search_results(limit=2)

        assert len(results) == 2

    def test_search_results_empty(self, mock_query_tree, mock_inverted_index):
        mock_inverted_index.index.get.return_value = PostingList(
            postings=[], term_frequencies={}, positions={}
        )

        qe = QueryEngine("test")
        with patch(
            "backend.search_engine.query.query_engine.normalize_search_query",
            return_value=["test"],
        ):
            results = qe.search_results(limit=10)

        assert len(results) == 0


class TestEdgeCases:
    def test_find_docs_empty_postings_and(self):
        postings1 = PostingList(postings=[], term_frequencies={}, positions={})
        postings2 = PostingList(
            postings=[1, 2, 3],
            term_frequencies={1: 1, 2: 1, 3: 1},
            positions={1: [0], 2: [1], 3: [2]},
        )
        postings1.build_skip_pointers()
        postings2.build_skip_pointers()

        result = find_docs(postings1, postings2, "AND")

        assert len(result.postings) == 0

    def test_find_docs_single_element(self):
        postings1 = PostingList(
            postings=[5], term_frequencies={5: 1}, positions={5: [0]}
        )
        postings2 = PostingList(
            postings=[5], term_frequencies={5: 2}, positions={5: [1, 3]}
        )
        postings1.build_skip_pointers()
        postings2.build_skip_pointers()

        result = find_docs(postings1, postings2, "AND")

        expected = [5]
        assert result.postings == expected
        assert result.term_frequencies[5] == 3


class TestIntegration:
    def test_full_workflow(self, mock_query_tree, mock_inverted_index):
        # (term1 AND term2) OR term3
        mock_inverted_index.index.get.side_effect = [
            PostingList(
                postings=[1, 2, 3],
                term_frequencies={1: 1, 2: 1, 3: 1},
                positions={1: [0], 2: [1], 3: [2]},
            ),
            PostingList(
                postings=[2, 3, 4],
                term_frequencies={2: 1, 3: 1, 4: 1},
                positions={2: [0], 3: [1], 4: [2]},
            ),
            PostingList(postings=[5], term_frequencies={5: 1}, positions={5: [0]}),
        ]

        mock_inverted_index.doc_store.get.side_effect = [
            DocInfo(url=f"http://example.com/{i}", title=f"Doc {i}") for i in [2, 3, 5]
        ]

        term1 = Node(value="term1")
        term2 = Node(value="term2")
        and_node = Node(value="AND", left=term1, right=term2)
        term3 = Node(value="term3")
        or_node = Node(value="OR", left=and_node, right=term3)

        mock_query_tree.root = or_node

        qe = QueryEngine("(term1 AND term2) OR term3")
        with patch(
            "backend.search_engine.query.query_engine.normalize_search_query",
            return_value=["(", "term1", "AND", "term2", ")", "OR", "term3"],
        ):
            results = qe.search_results(limit=10)

        assert len(results) == 3
        assert results[0].document_id == 2
        assert results[1].document_id == 3
        assert results[2].document_id == 5
