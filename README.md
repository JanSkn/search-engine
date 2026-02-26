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
* Docker (for containerized integration/unit tests & deployment)
* CMake (building and compiling the CPP components)
* Just (command runner)

## Entrypoints
### Docker

```bash
just deploy
```
Automated build process. Will download the dataset and build the index if
it does not exist yet. This preprocessing can take up to 2 hours.

Afterwards, it spins up a frontend and a backend container. 

- Access search engine frontend via `http://localhost:8080`.
- API-only: `http://localhost:8000`.
    
    **Search Endpoint**

    **GET** `/search`
    
    Query parameters:

    | Parameter | Type   | Description                                    |
    | --------- | ------ | ---------------------------------------------- |
    | `q`       | string | Search query (1–50 characters)                 |
    | `limit`   | int    | Maximum number of results (1–500, default: 10) |


### Manual usage
Download the dataset:
```bash
cd src && uv run --project backend python -m backend.search_engine.scripts.download_dataset
```

Build the index with a memory limit:
```bash
just build-index <memory-limit> [<max-docs>]
```

Start both frontend and backend in dev environment (`http://localhost:8080`):
```bash
just local [<uvicorn-args>]
```

Query Search Engine from CLI:
```bash
cd src && uv run --project backend python -m backend.search_engine.scripts.query --query <query> --limit <limit>
```
