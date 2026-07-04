# Property-Based Testing a Hybrid Search Engine


---

## 1. What we built

A search engine over ~3 million documents that blends two very different retrieval
philosophies and fuses their results:

- **Lexical search** — a classic inverted index with BM25 ranking, written in C++ for speed.
- **Semantic search** — neural embeddings + an approximate nearest-neighbour vector index.

On top of that sit query parsing, spell correction,etc. 

The interesting part for testing: Hot paths are
C++; orchestration, parsing, and fusion are Python. The interface is pybind11 (Python bindings of C++ code)

---

## 2. Query engine structure


```
   query string
        │
        ▼
  ┌───────────────┐   normalize + stem, keep operators
  │ 1. Normalize  │   (C++ tokenizer)
  └───────┬───────┘
          ▼
  ┌───────────────┐   suggest a corrected spelling
  │ 2. Spell-fix  │   (neural model, informational only)
  └───────┬───────┘
          ▼
  ┌───────────────┐   build a boolean parse tree
  │ 3. Parse      │   AND / OR / NOT / "quoted phrase"
  └───────┬───────┘
          ▼
  ┌───────────────┐   pick the most distinctive terms,
  │ 4. Candidates │   gather a bounded set of docs to score
  └───────┬───────┘
          ▼
  ┌───────────────────────────┐
  │ 5. Lexical retrieval + BM25│  boolean match + fielded BM25   (C++)
  │ 6. Semantic search         │  embed query, ANN lookup        (vectors)
  └───────────────┬───────────┘
                  ▼
  ┌───────────────┐   Reciprocal Rank Fusion — merge the two
  │ 7. Fuse (RRF) │   ranked lists by rank, not by score
  └───────┬───────┘
          ▼
  ┌───────────────┐   build result snippets, assemble response
  │ 8. Present    │   (C++ snippet generator)
  └───────────────┘
```

**Two phases overall:**

| Phase | What happens |
|-------|--------------|
| **Offline / build** | Build the inverted index, generate embeddings, train the vector index and the re-ranker. |
| **Online / query** | The pipeline above, per request. |

---

## 3. Components

| Component | Language | Role |
|-----------|----------|------|
| API layer | Python | One `/search` endpoint; loads all models once at startup |
| Query parser | Python | Turns a raw string into a boolean tree; validates operator usage |
| Tokenizer / normalizer | C++ | Lowercasing, stemming, operator handling |
| Inverted index | C++ | Posting lists, boolean set algebra, positional phrase search |
| BM25 scorer | C++ | Fielded scoring (title vs. body emphasis) |
| Rank fusion (RRF) | Python | Combines lexical + semantic rankings |
| Embedding model | Python | Turns text into vectors |
| Vector index | Python (FAISS) | Approximate nearest-neighbour search |
| Spell correction | Python | Neural checker; suggests corrections |
| Learning-to-rank | Python | Optional feature-based re-ranker |

Remark: **the same logic sometimes exists in two places** — once in C++ for
production speed, once in Python for feature extraction or as a reference. That is something we test as well.

---

## 4. Why property-based testing here

Example-based tests check the cases the developer thought of. This system actively resists
that approach for three reasons:

1. **The input space is unbounded.** A query is an arbitrary string: operators, quotes,
   parentheses, Unicode, hyphens, empty.

2. **There is an Python<->C++ seam.** Python constructs objects, sets attributes, and passes data
   into C++ that assumes invariants (sorted lists, present keys, valid ranges). Whether
   those assumptions hold for *every* input is a property, not an example.

3. **Duplicate implementations.** Where the same operation
   lives in both C++ and Python (or where a trivially-correct reference exists (Python's
   built-in `set` for boolean algebra)) we can assert the two must always agree.


### The four property archetypes we used

| Type | Idea | Example from this project |
|------|----------------------|---------------------------|
| **Invariant** | What is true of *every single* call? | Fused output is always sorted and never longer than requested |
| **Round-trip** | Does encode→decode return the original? | Normalizing already-normalized text should change nothing |
| **Metamorphic** | How do outputs of *related* inputs relate? | `A AND B` must equal `B AND A` |
| **Differential** | Do two implementations agree? | C++ boolean algebra vs. Python's `set`; Python phrase-match vs. C++ phrase-match |

---

## 5. Interesting findings

Every one of these was found by the fuzzer and shrunk to a minimal trigger.

### Tokenizer / normalization
- **A hyphen inside a word becomes a NOT operator.** `check-in` is read as `check NOT in`.
  A comment in the code claimed the opposite.
- **Non-ASCII is silently dropped.** `café` becomes `caf`; `café naïve` splits into three
  fragments. The engine is quietly ASCII-only.
- **Stemming is not idempotent.** A token gets shortened on the first pass and
  shortened *again* on a second pass. This matters because one query path normalizes the
  same terms *twice* while the index normalizes them *once*, so a doubly-stemmed query
  term can silently fail to match the index and drop the whole lexical result.
- **Reserved words are unsearchable.** You cannot search for the literal word "AND", it is
  always parsed as an operator.

### Parser
- **Malformed queries crash with the wrong error type.** Some inputs raise a bare,
  uncontrolled error that surfaces to the user as a 500 (server error) instead of a clean
  400 (bad request).
- **A stray closing parenthesis leaks in as a search term.**
- **The parser mutates its input list** as a side effect (it empties it).

### Scoring
- **BM25's IDF crashes on impossible input.** When document frequency exceeds the corpus
  size (a corrupt-index scenario), the Python version raises a math-domain error. 

### Boolean set algebra (C++)
- **`find_docs` silently assumes sorted input** and returns wrong results otherwise.
- **`find_docs` crashes** (index-out-of-bounds) when a posting lacks its term-frequency
  entry — and this affects more operators than the code comments suggested.
- **The same operation is implemented twice** with *divergent* robustness: the fast
  two-pointer version is fragile (needs sorted input, crashes on missing keys); a second,
  hash-set version is robust. Different call paths use different ones.

### Python setting state on C++ objects
- **Value semantics surprise.** Reading a C++ collection attribute from Python returns a
  *copy*. Mutating it in place (append) silently does nothing to the C++ object.

### Fusion (RRF)
- **A document duplicated within one ranked list gets double-counted**, inflating its fused
  score. The property "within one list a document contributes at most its best rank" catches it.

---


## 6. Limitations — where PBT fell short

- **Crashes kill shrinking.** A hard crash in native code (a segfault vs a catchable
  exception) takes the whole interpreter down, so there is nothing left to shrink. Native
  boundaries need defensive handling or subprocess isolation to stay fuzzable.
- **Heavy models don't scale to thousands of examples.** The embedding and spell-correction
  models are slow to load and run. Running hundreds of inputs against them is too slow. Here, we would
  need classical integration tests.
- **There is no oracle for ranking quality.** PBT checks *structural* truths: sorted,
  deterministic, bounded, order-independent. It cannot tell whether the results are
  relevant.

---

## 7. When PBT 

| Part of the system | Best-fit testing approach | Why |
|--------------------|---------------------------|-----|
| Pure functions: parser, fusion, set algebra, IDF | **Property-based** | Large input space, clear invariants, fast, shrinkable |
| C++/Python  boundary | **Property-based (differential + invariants)** | Two implementations to cross-check; type/lifetime edges are input-driven |
| Concurrency & shared state | **Stress / concurrency tests** | Single-threaded fuzzing cannot see races |
| Model behaviour (embeddings, spell-fix) | **Curated example sets / evaluation suites** | Too slow for mass sampling; correctness is subjective |
| Wiring, config, HTTP contract | **Example / integration tests** | Few, known, discrete cases — enumeration is fine |


