from __future__ import annotations

import math

import cpp_utils as c
import pytest
from backend.search_engine.error_handling import InvalidOperatorError
from backend.search_engine.query.query_engine import QueryEngine
from backend.search_engine.query.query_preprocessing import AND, NOT, OR, QueryTree
from backend.search_engine.scoring.bm25 import bm25_idf
from hypothesis import given, settings
from hypothesis import strategies as st

# Strategies

_LOWER = "abcdefghijklmnopqrstuvwxyz"
_ALNUM = _LOWER + _LOWER.upper() + "0123456789"

# A lowercase word: the tokenizer can never confuse it with an operator
# (operators in KEEP_TOKENS are uppercase: AND/OR/NOT).
lower_word = st.text(alphabet=_LOWER, min_size=1, max_size=8)
# A mixed-case alphanumeric word (can collide with reserved words).
alnum_word = st.text(alphabet=_ALNUM, min_size=1, max_size=8)


@st.composite
def terms_and_ops_tokens(draw):
    terms = draw(st.lists(lower_word, min_size=1, max_size=5))
    n_ops = len(terms) - 1
    ops = draw(st.lists(st.sampled_from(["AND", "OR"]), min_size=n_ops, max_size=n_ops))
    out: list[str] = []
    for i, term in enumerate(terms):
        out.append(term)
        if i < len(ops):
            out.append(ops[i])
    return out


# Sorted, unique doc-id lists = the precondition find_docs actually relies on.
doc_ids = st.lists(st.integers(min_value=0, max_value=10_000), unique=True, max_size=40)


def _pl(ids: list[int]) -> c.PostingList:
    """A valid posting list: sorted ids, every id present in term_frequencies."""
    ids = sorted(ids)
    pl = c.PostingList(postings=ids, term_frequencies={i: 1 for i in ids}, positions={})
    pl.build_skip_pointers()
    return pl


# 1. INVARIANTS - hold for every input


class TestInvariants:
    @settings(max_examples=300)
    @given(terms_and_ops_tokens())
    def test_unique_terms_subset_of_input(self, tokens: list[str]) -> None:
        """Positive (non-negated) terms collected by the parser must all come
        from the input tokens."""
        qt = QueryTree()
        input_terms = {t for t in tokens if t not in (AND | OR | NOT)}
        qt.parse_query(list(tokens))
        assert qt.unique_terms <= input_terms

    ranked_list = st.lists(
        st.tuples(st.integers(0, 1000), st.floats(0, 1)), max_size=20
    )

    @settings(max_examples=300)
    @given(
        lists=st.lists(ranked_list, max_size=4),
        top_n=st.integers(min_value=0, max_value=10),
    )
    def test_rrf_output_sorted_and_bounded(
        self, lists: list[list[tuple[int, float]]], top_n: int
    ) -> None:
        out = QueryEngine._reciprocal_rank_fusion(lists, top_n=top_n, k=60)
        assert len(out) <= top_n
        scores = [s for _, s in out]
        assert scores == sorted(scores, reverse=True)


# 2. METAMORPHIC - relation among related inputs, we dont know result of input A and of input B,
# but we know the relation between the results


class TestMetamorphic:
    @settings(max_examples=400)
    @given(st.lists(lower_word, min_size=1, max_size=6))
    def test_lowercase_input_never_yields_operators(self, words: list[str]) -> None:
        """Lowercase alphanumeric input must never produce a boolean operator."""
        tokens = c.normalize_search_query(" ".join(words))
        assert not (set(tokens) & {"AND", "OR", "NOT", "&", "|", "-", "(", ")"})

    @settings(max_examples=300)
    @given(doc_ids, doc_ids)
    def test_find_docs_and_commutative(self, a: list[int], b: list[int]) -> None:
        left, right = _pl(a), _pl(b)
        # find_docs is one of the most important functions --> filters docs from 2 different PostingLists
        assert set(c.find_docs(left, right, "AND").postings) == set(
            c.find_docs(right, left, "AND").postings
        )

    @settings(max_examples=300)
    @given(doc_ids, doc_ids)
    def test_find_docs_and_subset_of_inputs(self, a: list[int], b: list[int]) -> None:
        result = set(c.find_docs(_pl(a), _pl(b), "AND").postings)
        assert result <= set(a) and result <= set(b)


# 3. DIFFERENTIAL - 2 implementations must deliver same result
# we do this because we have a wild mixture of python and cpp code


class TestDifferential:
    @settings(max_examples=500)
    @given(
        num_docs=st.integers(min_value=1, max_value=2**31 - 1),
        df=st.data(),
    )
    def test_python_idf_matches_cpp_formula(self, num_docs: int, df) -> None:
        """For a VALID index (df <= num_docs), the Python bm25_idf must equal the
        C++ bm25_idf_cpp formula bit-for-bit (within float epsilon)."""
        d = df.draw(st.integers(min_value=1, max_value=num_docs))
        py = bm25_idf(num_docs, d, clamp_negative=True)
        val = math.log((num_docs - d + 0.5) / (d + 0.5))  # mirror of C++ impl
        cpp = 0.0 if val < 0.0 else val
        assert py == pytest.approx(cpp, rel=1e-12, abs=1e-12)

    @settings(max_examples=300)
    @given(num_docs=st.integers(1, 1_000_000), df=st.data())
    def test_idf_non_negative_when_clamped(self, num_docs: int, df) -> None:
        d = df.draw(st.integers(min_value=1, max_value=num_docs))
        assert bm25_idf(num_docs, d, clamp_negative=True) >= 0.0

    @settings(max_examples=400)
    @given(doc_ids, doc_ids)
    def test_and_matches_set_intersection(self, a: list[int], b: list[int]) -> None:
        assert set(c.find_docs(_pl(a), _pl(b), "AND").postings) == (set(a) & set(b))

    @settings(max_examples=400)
    @given(doc_ids, doc_ids)
    def test_or_matches_set_union(self, a: list[int], b: list[int]) -> None:
        assert set(c.find_docs(_pl(a), _pl(b), "OR").postings) == (set(a) | set(b))

    @settings(max_examples=400)
    @given(doc_ids, doc_ids)
    def test_not_matches_set_difference(self, a: list[int], b: list[int]) -> None:
        assert set(c.find_docs(_pl(a), _pl(b), "NOT").postings) == (set(a) - set(b))


# FINDINGS AS PROPERTIES
# Hypothesis discovers a failing input and shrinks it
# to the minimal counterexample


def _pl_raw(ids: list[int]) -> c.PostingList:
    """A posting list that does NOT sort its ids (violates find_docs' precondition)."""
    pl = c.PostingList(
        postings=list(ids), term_frequencies={i: 1 for i in ids}, positions={}
    )
    pl.build_skip_pointers()
    return pl


# A word containing an interior hyphen, e.g. 'check-in'.
hyphenated_word = st.tuples(lower_word, lower_word).map(lambda t: f"{t[0]}-{t[1]}")
# Alphabet of non-ASCII latin letters the ASCII-only tokenizer drops.
accented_word = st.text(alphabet="àáâäéèêëíìîïóòôöúùûüñç", min_size=1, max_size=6)
# Tokens that may form a *malformed* boolean query.
maybe_malformed = st.lists(
    st.one_of(lower_word, st.sampled_from(["AND", "OR", "NOT", "(", ")"])),
    max_size=6,
)
# Tokens that may contain stray parentheses.
maybe_parens = st.lists(
    st.one_of(lower_word, st.sampled_from(["(", ")"])), min_size=1, max_size=6
)


class TestFindingsProperties:
    # tokenizer (C++)

    # @pytest.mark.xfail(strict=True, reason="hyphen inside a word becomes a NOT token")
    @settings(max_examples=200, deadline=None)
    @given(hyphenated_word)
    def test_hyphen_never_becomes_operator(self, text: str) -> None:
        # A hyphen between two letters should stay part of the term, never a NOT.
        assert "-" not in c.normalize_search_query(text)

    # @pytest.mark.xfail(strict=True, reason="non-ASCII characters are silently dropped")
    @settings(max_examples=200, deadline=None)
    @given(accented_word)
    def test_non_ascii_survives_normalization(self, word: str) -> None:
        # Every non-ASCII letter of the input should appear somewhere in the output.
        joined = "".join(c.normalize_search_query(word))
        assert all(ch in joined for ch in word)

    # @pytest.mark.xfail(strict=True, reason="stemming is not idempotent: i0se->i0s->i0")
    @settings(max_examples=500, deadline=None)
    @given(alnum_word)
    def test_normalize_is_idempotent(self, word: str) -> None:
        # stemming happens twice when _to_boolean_normalized_query
        # --> word potentially not found in index becuase of idempotency
        once = c.normalize_search_query(word)
        twice = c.normalize_search_query(" ".join(once))
        assert once == twice

    # @pytest.mark.xfail(
    #     strict=True, reason="reserved words: the literal word 'AND' is unsearchable"
    # )
    @settings(max_examples=50, deadline=None)
    @given(st.sampled_from(["AND", "OR", "NOT"]))
    def test_reserved_word_searchable(self, word: str) -> None:
        # A user searching for the literal word should get it back as a term.
        assert c.normalize_search_query(word) == [word.lower()]

    # parser (pure Python)

    # @pytest.mark.xfail(
    #     strict=True, reason="stray ')' leaks into unique_terms as a search term"
    # )
    @settings(max_examples=300)
    @given(maybe_parens)
    def test_paren_never_a_term(self, tokens: list[str]) -> None:
        qt = QueryTree()
        try:
            qt.parse_query(list(tokens))
        except (InvalidOperatorError, ValueError):
            return  # failed to parse for unrelated reasons; not what we test here
        assert not ({"(", ")"} & qt.unique_terms)

    # @pytest.mark.xfail(
    #     strict=True, reason="parse_query mutates (empties) its input list"
    # )
    @settings(max_examples=300)
    @given(terms_and_ops_tokens())
    def test_parse_query_no_side_effect(self, tokens: list[str]) -> None:
        qt = QueryTree()
        original = list(tokens)
        qt.parse_query(tokens)
        assert tokens == original

    # scoring (Python)

    # @pytest.mark.xfail(
    #     strict=True, reason="bm25_idf raises math domain error when df>num_docs"
    # )
    @settings(max_examples=300)
    @given(num_docs=st.integers(1, 10_000), extra=st.integers(1, 10_000))
    def test_idf_handles_df_greater_than_n(self, num_docs: int, extra: int) -> None:
        # An over-counted df (df > N) must clamp to 0.0, not crash on log(negative).
        assert (
            bm25_idf(num_docs=num_docs, df=num_docs + extra, clamp_negative=True) == 0.0
        )

    # boolean set algebra (C++)

    # @pytest.mark.xfail(
    #     strict=True, reason="find_docs assumes sorted input; unsorted -> wrong result"
    # )
    @settings(max_examples=300)
    @given(st.lists(st.integers(0, 60), unique=True, min_size=2, max_size=20))
    def test_find_docs_handles_unsorted_input(self, ids: list[int]) -> None:
        # Same ids in both lists -> AND must return all of them, regardless of order.
        # in theory, this will never happen because assumption is that doc ids are sorted,
        # but becomes problematic in larger system
        ids = sorted(ids)
        left = _pl_raw(list(reversed(ids)))  # descending: violates the precondition
        right = _pl(ids)
        assert set(c.find_docs(left, right, "AND").postings) == set(ids)

    # @pytest.mark.xfail(
    #     strict=True, reason="find_docs AND uses .at() -> IndexError on missing tf key"
    # )
    @settings(max_examples=300)
    @given(st.lists(st.integers(0, 40), unique=True, min_size=2, max_size=15))
    def test_find_docs_missing_tf_no_crash(self, ids: list[int]) -> None:
        ids = sorted(ids)
        tf = {i: 1 for i in ids[1:]}  # drop the tf entry for the first posting
        x = c.PostingList(postings=ids, term_frequencies=tf, positions={})
        x.build_skip_pointers()
        c.find_docs(x, _pl(ids), "AND")  # should degrade gracefully, not raise

    # fusion (Python)

    # @pytest.mark.xfail(
    #     strict=True, reason="RRF double-counts a doc duplicated within one list"
    # )
    @settings(max_examples=300)
    # less docs than elements --> duplicates
    @given(
        st.lists(st.tuples(st.integers(0, 5), st.floats(0, 1)), min_size=2, max_size=10)
    )
    def test_rrf_no_double_count(self, single_list: list[tuple[int, float]]) -> None:
        # Within ONE list a doc can contribute at most its best rank: 1/(k+0+1).
        out = QueryEngine._reciprocal_rank_fusion([single_list], top_n=50, k=60)
        for _, score in out:
            assert score <= 1 / 61 + 1e-12  # + 1e-12 is rounding buffer
