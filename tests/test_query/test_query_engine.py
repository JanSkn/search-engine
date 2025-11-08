import pytest
import numpy as np
from unittest.mock import Mock, patch
from backend.search_engine.models.index import PostingList, SearchResult
from backend.search_engine.query.query_engine import QueryEngine
from backend.search_engine.query.query_preprocessing import Node, AND, OR, NOT


@pytest.fixture
def mock_query_tree():
    with patch('backend.search_engine.query.query_engine.QueryTree') as mock_qt_class:
        mock_qt_instance = Mock()
        mock_qt_class.return_value = mock_qt_instance
        mock_qt_instance.root = Node(value="test")
        yield mock_qt_instance


@pytest.fixture
def mock_inverted_index():
    with patch('backend.search_engine.query.query_engine.inverted_index') as mock_inverted_index:
        mock_inverted_index.all_doc_ids = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        mock_inverted_index.index = {}
        mock_inverted_index.doc_store = Mock()
        yield mock_inverted_index


class TestFindDocsAND:
    def test_and_with_common_elements(self):
        postings1 = PostingList(postings=np.array([1, 3, 5, 7, 9]))
        postings2 = PostingList(postings=np.array([2, 3, 5, 8, 10]))
        result = QueryEngine._find_docs(postings1, postings2, "AND")
        
        expected = np.array([3, 5])
        np.testing.assert_array_equal(result.postings, expected)
    
    def test_and_with_no_common_elements(self):
        postings1 = PostingList(postings=np.array([1, 3, 5]))
        postings2 = PostingList(postings=np.array([2, 4, 6]))
        postings1.build_skip_pointers()
        postings2.build_skip_pointers()
        
        result = QueryEngine._find_docs(postings1, postings2, "AND")
        
        assert len(result.postings) == 0
    
    def test_and_with_all_common_elements(self):
        postings1 = PostingList(postings=np.array([1, 2, 3]))
        postings2 = PostingList(postings=np.array([1, 2, 3]))
        postings1.build_skip_pointers()
        postings2.build_skip_pointers()
        
        result = QueryEngine._find_docs(postings1, postings2, "AND")
        
        expected = np.array([1, 2, 3])
        np.testing.assert_array_equal(result.postings, expected)
    
    def test_and_skip_pointers_optimization(self):
        # Large posting lists to test skip pointer optimization
        postings1 = PostingList(postings=np.array([1, 5, 10, 15, 20, 25, 30]))
        postings2 = PostingList(postings=np.array([3, 5, 7, 10, 12, 15, 18, 20]))
        postings1.build_skip_pointers()
        postings2.build_skip_pointers()
        
        result = QueryEngine._find_docs(postings1, postings2, "AND")
        
        expected = np.array([5, 10, 15, 20])
        np.testing.assert_array_equal(result.postings, expected)


class TestFindDocsOR:
    def test_or_with_common_elements(self):
        postings1 = PostingList(postings=np.array([1, 3, 5, 7, 9]))
        postings2 = PostingList(postings=np.array([2, 3, 5, 8, 10]))
        result = QueryEngine._find_docs(postings1, postings2, "OR")
        
        expected = np.array([1, 2, 3, 5, 7, 8, 9, 10])
        np.testing.assert_array_equal(result.postings, expected)
    
    def test_or_with_no_common_elements(self):
        postings1 = PostingList(postings=np.array([1, 3, 5]))
        postings2 = PostingList(postings=np.array([2, 4, 6]))
        postings1.build_skip_pointers()
        postings2.build_skip_pointers()
        
        result = QueryEngine._find_docs(postings1, postings2, "OR")
        
        expected = np.array([1, 2, 3, 4, 5, 6])
        np.testing.assert_array_equal(result.postings, expected)
    
    def test_or_with_all_common_elements(self):
        postings1 = PostingList(postings=np.array([1, 2, 3]))
        postings2 = PostingList(postings=np.array([1, 2, 3]))
        postings1.build_skip_pointers()
        postings2.build_skip_pointers()
        
        result = QueryEngine._find_docs(postings1, postings2, "OR")
        
        expected = np.array([1, 2, 3])
        np.testing.assert_array_equal(result.postings, expected)


class TestFindDocsNOT:
    def test_not_operation(self, mock_inverted_index):
        postings = PostingList(postings=np.array([2, 4, 6, 8]))
        postings.build_skip_pointers()
        
        result = QueryEngine._find_docs(None, postings, "NOT")
        
        expected = np.array([1, 3, 5, 7, 9, 10])
        np.testing.assert_array_equal(result.postings, expected)
    
    def test_not_with_empty_postings(self, mock_inverted_index):
        postings = PostingList(postings=np.array([]))
        postings.build_skip_pointers()
        
        result = QueryEngine._find_docs(None, postings, "NOT")
        
        expected = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        np.testing.assert_array_equal(result.postings, expected)
    
    def test_not_with_all_docs(self, mock_inverted_index):
        postings = PostingList(postings=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
        postings.build_skip_pointers()
        
        result = QueryEngine._find_docs(None, postings, "NOT")
        
        assert len(result.postings) == 0


class TestEvaluate:
    def test_evaluate_leaf_node(self, mock_inverted_index):
        posting_list = PostingList(postings=np.array([1, 2, 3]))
        mock_inverted_index.index.get.return_value = posting_list
        
        node = Node(value="test")
        result = QueryEngine.evaluate(node)
        
        mock_inverted_index.index.get.assert_called_once_with("test")
        assert result == posting_list
    
    def test_evaluate_and_node(self, mock_inverted_index):
        mock_inverted_index.index.get.side_effect = [
            PostingList(postings=np.array([1, 2, 3])),
            PostingList(postings=np.array([2, 3, 4]))
        ]
        
        left_node = Node(value="term1")
        right_node = Node(value="term2")
        and_node = Node(value="AND", left=left_node, right=right_node)
        
        result = QueryEngine.evaluate(and_node)
        
        expected = np.array([2, 3])
        np.testing.assert_array_equal(result.postings, expected)
    
    def test_evaluate_or_node(self, mock_inverted_index):
        mock_inverted_index.index.get.side_effect = [
            PostingList(postings=np.array([1, 2])),
            PostingList(postings=np.array([3, 4]))
        ]
        
        left_node = Node(value="term1")
        right_node = Node(value="term2")
        or_node = Node(value="OR", left=left_node, right=right_node)
        
        result = QueryEngine.evaluate(or_node)
        
        expected = np.array([1, 2, 3, 4])
        np.testing.assert_array_equal(result.postings, expected)
    
    def test_evaluate_not_node(self, mock_inverted_index):
        mock_inverted_index.index.get.return_value = PostingList(postings=np.array([2, 4, 6, 8, 10]))
        
        right_node = Node(value="term1")
        not_node = Node(value="NOT", right=right_node)
        
        result = QueryEngine.evaluate(not_node)
        
        expected = np.array([1, 3, 5, 7, 9])
        np.testing.assert_array_equal(result.postings, expected)
    
    def test_evaluate_complex_query(self, mock_inverted_index):
        # (term1 AND term2) OR term3
        mock_inverted_index.index.get.side_effect = [
            PostingList(postings=np.array([1, 2, 3])),
            PostingList(postings=np.array([2, 3, 4])),
            PostingList(postings=np.array([5, 6]))
        ]
        
        term1 = Node(value="term1")
        term2 = Node(value="term2")
        and_node = Node(value="AND", left=term1, right=term2)
        term3 = Node(value="term3")
        or_node = Node(value="OR", left=and_node, right=term3)
        
        result = QueryEngine.evaluate(or_node)
        
        expected = np.array([2, 3, 5, 6])
        np.testing.assert_array_equal(result.postings, expected)


class TestSearchResults:
    def test_search_results_basic(self, mock_query_tree, mock_inverted_index):
        mock_inverted_index.index.get.return_value = PostingList(postings=np.array([1, 2, 3]))
        mock_inverted_index.doc_store.get.side_effect = [
            {"url": "http://example.com/1", "title": "Doc 1"},
            {"url": "http://example.com/2", "title": "Doc 2"},
            {"url": "http://example.com/3", "title": "Doc 3"}
        ]
        
        qe = QueryEngine("test")
        with patch.object(qe, '_normalized_query', return_value=["test"]):
            results = qe.search_results(limit=10)
        
        assert len(results) == 3
        assert all(isinstance(r, SearchResult) for r in results)
    
    def test_search_results_with_limit(self, mock_query_tree, mock_inverted_index):
        mock_inverted_index.index.get.return_value = PostingList(postings=np.array([1, 2, 3, 4, 5]))
        mock_inverted_index.doc_store.get.side_effect = [
            {"url": f"http://example.com/{i}", "title": f"Doc {i}"}
            for i in range(1, 6)
        ]
        
        qe = QueryEngine("test")
        with patch.object(qe, '_normalized_query', return_value=["test"]):
            results = qe.search_results(limit=2)
        
        assert len(results) == 2
    
    def test_search_results_empty(self, mock_query_tree, mock_inverted_index):
        mock_inverted_index.index.get.return_value = PostingList(postings=np.array([]))
        
        qe = QueryEngine("test")
        with patch.object(qe, '_normalized_query', return_value=["test"]):
            results = qe.search_results(limit=10)
        
        assert len(results) == 0


class TestEdgeCases:
    def test_find_docs_empty_postings_and(self):
        postings1 = PostingList(postings=np.array([]))
        postings2 = PostingList(postings=np.array([1, 2, 3]))
        postings1.build_skip_pointers()
        postings2.build_skip_pointers()
        
        result = QueryEngine._find_docs(postings1, postings2, "AND")
        
        assert len(result.postings) == 0
    
    def test_find_docs_single_element(self):
        postings1 = PostingList(postings=np.array([5]))
        postings2 = PostingList(postings=np.array([5]))
        postings1.build_skip_pointers()
        postings2.build_skip_pointers()
        
        result = QueryEngine._find_docs(postings1, postings2, "AND")
        
        expected = np.array([5])
        np.testing.assert_array_equal(result.postings, expected)


class TestIntegration:
    def test_full_workflow(self, mock_query_tree, mock_inverted_index):
        # (term1 AND term2) OR term3
        mock_inverted_index.index.get.side_effect = [
            PostingList(postings=np.array([1, 2, 3])),
            PostingList(postings=np.array([2, 3, 4])),
            PostingList(postings=np.array([5]))
        ]
        
        mock_inverted_index.doc_store.get.side_effect = [
            {"url": f"http://example.com/{i}", "title": f"Document {i}"}
            for i in [2, 3, 5]
        ]
        
        # Build query tree
        term1 = Node(value="term1")
        term2 = Node(value="term2")
        and_node = Node(value="AND", left=term1, right=term2)
        term3 = Node(value="term3")
        or_node = Node(value="OR", left=and_node, right=term3)
        
        mock_query_tree.root = or_node
        
        qe = QueryEngine("(term1 AND term2) OR term3")
        with patch.object(qe, '_normalized_query', return_value=["term1", "AND", "term2", "OR", "term3"]):
            results = qe.search_results(limit=10)
        
        assert len(results) == 3
        assert results[0].document_id == 2
        assert results[1].document_id == 3
        assert results[2].document_id == 5
