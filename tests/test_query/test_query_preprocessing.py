import pytest
from backend.search_engine.error_handling import ParenthesesWarning
from backend.search_engine.query.query_preprocessing import Node, QueryTree


class TestQueryTree:
    @pytest.fixture
    def query_tree(self) -> QueryTree:
        return QueryTree()

    def test_parse_query_returns_node(self, query_tree: QueryTree):
        query_tree.parse_query(["A"])

        assert query_tree.root is not None
        assert isinstance(query_tree.root, Node)

    def test_parse_query_missing_closing_parenthesis(self, query_tree: QueryTree):
        with pytest.warns(ParenthesesWarning) as record:
            query_tree.parse_query(["(", "A", "AND", "B"])

        assert len(record) == 2

        w_1 = record.list[0].message
        w_2 = record.list[1].message
        assert isinstance(w_1, ParenthesesWarning)
        assert str(w_1) == "Query may not be parsed as intended: Unbalanced parentheses: 1 '(' vs 0 ')'"
        assert isinstance(w_2, ParenthesesWarning)
        assert str(w_2) == "Missing closing parenthesis: query may not be parsed as intended"

    def _assert_tree_equal(self, node: Node, expected: dict[str, any]):
        assert node is not None
        assert node.value == expected["value"]

        if "left" in expected:
            self._assert_tree_equal(node.left, expected["left"])
        else:
            assert node.left is None

        if "right" in expected:
            self._assert_tree_equal(node.right, expected["right"])
        else:
            assert node.right is None

    @pytest.mark.parametrize("query, expected_tree", [
        (
            ["A", "AND", "B"],
            {"value": "AND", "left": {"value": "A"}, "right": {"value": "B"}}
        ),
        (
            ["A", "AND", "B", "AND", "C"],    # more than binary
            {"value": "AND", "left": {"value": "AND", "left": {"value": "A"}, "right": {"value": "B"}}, "right": {"value": "C"}}
        ),
        (
            ["A", "&", "B"],
            {"value": "AND", "left": {"value": "A"}, "right": {"value": "B"}}
        ),
        (
            ["A"],  # one word phrase query
            {"value": "A"}
        ),
        (
            ["A", "B", "C"],  # phrase query with biword chain
            {
                "value": "AND",
                "left": {
                    "value": "AND",
                    "left": {"value": "A"},
                    "right": {"value": "B"},
                },
                "right": {
                    "value": "AND",
                    "left": {"value": "B"},
                    "right": {"value": "C"},
                },
            },
        ),
        (
            ["A", "OR", "B"],
            {"value": "OR", "left": {"value": "A"}, "right": {"value": "B"}}
        ),
        (
            ["A", "|", "B"],
            {"value": "OR", "left": {"value": "A"}, "right": {"value": "B"}}
        ),
        (
            ["NOT", "A"],
            {"value": "NOT", "right": {"value": "A"}}
        ),
        (
            ["-", "A"],
            {"value": "NOT", "right": {"value": "A"}}
        ),
        (
            ["A", "OR", "NOT", "B"],
            {"value": "OR", "left": {"value": "A"}, "right": {"value": "NOT", "right": {"value": "B"}}}
        ),
        (
            ["A", "AND", "(", "B", "OR", "C", ")"],
            {
                "value": "AND",
                "left": {"value": "A"},
                "right": {
                    "value": "OR",
                    "left": {"value": "B"},
                    "right": {"value": "C"}
                }
            }
        ),
        (
            ["-", "(", "A", "OR", "B", ")"],
            {
                "value": "NOT",
                "right": {
                    "value": "OR",
                    "left": {"value": "A"},
                    "right": {"value": "B"}
                }
            }
        ),
        (
            ["(", "A", "OR", "B", ")", "AND", "(", "C", "OR", "D", ")"],
            {
                "value": "AND",
                "left": {
                    "value": "OR",
                    "left": {"value": "A"},
                    "right": {"value": "B"}
                },
                "right": {
                    "value": "OR",
                    "left": {"value": "C"},
                    "right": {"value": "D"}
                }
            }
        ),
        (
            ["A", "AND", "(", "B", "OR", "(", "C", "AND", "D", ")", ")"],
            {
                "value": "AND",
                "left": {"value": "A"},
                "right": {
                    "value": "OR",
                    "left": {"value": "B"},
                    "right": {
                        "value": "AND",
                        "left": {"value": "C"},
                        "right": {"value": "D"}
                    }
                }
            }
        ),
        (
            ["NOT", "(", "A", "AND", "(", "B", "OR", "NOT", "C", ")", ")", "OR", "D"],
            {
                "value": "OR",
                "left": {
                    "value": "NOT",
                    "right": {
                        "value": "AND",
                        "left": {"value": "A"},
                        "right": {
                            "value": "OR",
                            "left": {"value": "B"},
                            "right": {"value": "NOT", "right": {"value": "C"}}
                        }
                    }
                },
                "right": {"value": "D"}
            }
        ),
    ]
    )
    def test_parse_query(self, query_tree: QueryTree, query: list[str], expected_tree: dict[str, any]):
        query_tree.parse_query(query)
        self._assert_tree_equal(query_tree.root, expected_tree)
        