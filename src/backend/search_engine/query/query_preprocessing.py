from __future__ import annotations
import warnings
from backend.search_engine.error_handling import (
    ParenthesesWarning,
    InvalidOperatorError,
)

# TODO considering reordering of terms by frequency

# if adjusting connectors, also add in CONNECTOR_MAPPING
AND = {"AND", "&"}
OR = {"OR", "|"}
# - without whitespace gets treated as part of the term, e.g. check-in
NOT = {"NOT", "-"}

# normalize connectors
CONNECTOR_MAPPING = {"&": "AND", "|": "OR", "-": "NOT"}


class Node:
    def __init__(
        self,
        value: str,
        left: Node | None = None,
        right: Node | None = None,
    ) -> None:
        self.value: str = CONNECTOR_MAPPING.get(
            value, value
        )  # map values if mapping exists, otherwise original
        self.left: Node | None = left
        self.right: Node | None = right

    def __repr__(self) -> str:
        return f"Node(value={self.value!r}, left={self.left!r}, right={self.right!r})"


class QueryTree:
    def __init__(self) -> None:
        self._root: Node | None = None
        self._warnings_stack: list[ParenthesesWarning] = []

    @property
    def root(self) -> Node | None:
        return self._root

    @property
    def warnings_stack(self) -> list[ParenthesesWarning]:
        return self._warnings_stack

    def _check_parentheses(self, tokens: list[str]) -> None:
        open_count = tokens.count("(")
        close_count = tokens.count(")")
        if open_count != close_count:
            msg = f"Query may not be parsed as intended: Unbalanced parentheses: {open_count} '(' vs {close_count} ')'"
            w = ParenthesesWarning(msg, open_count, close_count)
            warnings.warn(msg, category=ParenthesesWarning, stacklevel=2)
            self._warnings_stack.append(w)

    @staticmethod
    def _has_operators(tokens: list[str]) -> bool:
        operator_tokens = AND | OR | NOT | {"(", ")"}
        return any(token in operator_tokens for token in tokens)

    def _validate_not_usage(self, node: Node, parent: Node | None = None) -> None:
        if node is None:
            return

        if node.value in NOT:
            # NOT without a parent -> not allowed
            if parent is None:
                raise InvalidOperatorError("NOT must be combined with AND")

            # parent must be AND
            if parent.value not in AND:
                raise InvalidOperatorError("NOT must be combined with AND")

        # OR also cannot have NOT children
        if node.value in OR:
            if node.left and node.left.value in NOT:
                raise InvalidOperatorError("NOT cannot be combined with OR")
            if node.right and node.right.value in NOT:
                raise InvalidOperatorError("NOT cannot be combined with OR")

        if node.left:
            self._validate_not_usage(node.left, node)
        if node.right:
            self._validate_not_usage(node.right, node)

    def parse_query(self, tokens: list[str]) -> None:
        self._warnings_stack.clear()
        self._check_parentheses(tokens)
        self._root = self._parse_query(tokens)
        self._validate_not_usage(self._root)

    def _parse_query(self, tokens: list[str]) -> Node:
        node = self._parse_term(tokens)

        while tokens and tokens[0] in (AND | OR):
            connector = tokens.pop(0)
            right = self._parse_term(tokens)
            node = Node(connector, left=node, right=right)

        return node

    # handle NOT and parentheses
    def _parse_term(self, tokens: list[str]) -> Node:
        if not tokens:
            raise ValueError("Unexpected end of tokens while parsing term")

        token = tokens[0]

        if token in NOT:
            tokens.pop(0)
            word_node = self._parse_term(tokens)
            return Node(token, left=None, right=word_node)

        if token == "(":
            tokens.pop(0)
            node = self._parse_query(tokens)

            if not tokens or tokens[0] != ")":
                msg = "Missing closing parenthesis: query may not be parsed as intended"
                w = ParenthesesWarning(msg)
                warnings.warn(msg, ParenthesesWarning, stacklevel=2)
                self._warnings_stack.append(w)
            # not popping here can cause wrong trees -> remove else, raise Exception to prevent
            else:
                tokens.pop(0)
            return node

        if token == ")":
            msg = "Unexpected closing parenthesis: query may not be parsed as intended"
            w = ParenthesesWarning(msg)
            warnings.warn(msg, ParenthesesWarning, stacklevel=2)
            self._warnings_stack.append(w)

        # leaf node/actual word
        if token not in (AND | OR | NOT):
            tokens.pop(0)
            return Node(token)

        raise ValueError("Unexpected token")

    def __repr__(self) -> str:
        return repr(self._root)

    def __str__(self) -> str:
        return self.tree_to_str(self._root)

    def tree_to_str(
        self, node: Node | None, prefix: str = "", is_left: bool = True
    ) -> str:
        if node is None:
            return ""

        connector = "├── " if is_left else "└── "
        s = prefix + connector + node.value + "\n"

        if node.left or node.right:
            new_prefix = prefix + ("│   " if is_left else "    ")
            if node.left:
                s += self.tree_to_str(node.left, new_prefix, True)
            if node.right:
                s += self.tree_to_str(node.right, new_prefix, False)

        return s
