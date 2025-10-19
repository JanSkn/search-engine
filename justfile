build-frontend:
    cd src/frontend && npm run build

install-frontend:
    cd src/frontend && npm install

install-backend:
    cd src/backend
    @poetry install

# install frontend and backend
install:
    just install-frontend
    just install-backend

local uvicorn-args:
    bash local.sh {{uvicorn-args}}

lint:
    @echo "Linting with Ruff..."
    poetry run ruff check src/backend/api src/backend/search_engine

mypy:
    @echo "Running type checks..."
    poetry run mypy src/backend/api
    poetry run mypy src/backend/search_engine

test args="":
    cd tests
    just -f tests/justfile test {{args}}