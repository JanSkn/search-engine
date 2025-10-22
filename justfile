build-frontend:
    cd src/frontend && npm run build

install-frontend:
    cd src/frontend && npm install

install-backend:
    cd src/backend && uv sync

# install frontend and backend
install:
    just install-frontend
    just install-backend

local uvicorn-args="":
    chmod +x local.sh
    ./local.sh {{uvicorn-args}}

lint:
    @echo "Linting with Ruff..."
    cd src/backend && uv run ruff check api/ search_engine/
    cd src/backend && uv run ruff format --check --diff api/ search_engine/

mypy:
    @echo "Type checking with MyPy..."
    cd src/backend && uv run mypy api/
    cd src/backend && uv run mypy search_engine/

test args="":
    just -f tests/justfile test {{args}}