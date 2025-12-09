# Seekr.

Seekr is a minimal search engine based on **~3 million** websites.

## Components
Seekr consists of several core subsystems working together:

* **Inverted Index (CPP, lazy-loaded)** — a high-performance inverted index implemented in C++ that loads posting lists on demand from binary files.
* **Boolean Query Engine** — supports AND/OR/NOT evaluation over posting lists.
* **Phrase Query Engine** — supports ordered phrase matching.
* **Positional Phrase Queries** — supports exact positional constraints using positional indexes.
* **Top-N Scoring** — ranks documents using a custom scoring pipeline to retrieve the top-N most relevant results.
* **Spell Correction (ML)** — ML-based correction and normalization of search queries.
* **Snippeting** — extracts meaningful snippets from documents based on matched terms.
* **ML-based Ranking** — integrates machine-learning models for ranking refinement.
* **Python Core Logic** — the main orchestration logic is implemented in Python.
* **CPP Bindings via Pybind** — performance-critical components are exposed to Python through pybind for fast execution.

## Prerequisites
* uv
* Node.js
* npm
* Docker (for containerized integration/unit tests)
* LFS (downloading ML models from GitHub)
* CMake (building and compiling the CPP components)
* Just (command runner)

## Entrypoints

### Build the Index
Before running the system, build the index with a memory limit:
```bash
just build-index <memory-limit>
```

Start both frontend and backend in dev environment (`http://localhost:8080`):
```bash
just local <uvicorn-args>
```

Query Search Engine from CLI:
```bash
cd src && uv run --project backend python -m backend.search_engine.scripts.query --query <query> --limit <limit>
```