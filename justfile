build-frontend:
    cd src/frontend && npm run build

install-frontend:
    cd src/frontend && npm install

install-backend:
    cd src/backend && poetry install --no-root

# install frontend and backend
install:
    just install-frontend
    just install-backend

local uvicorn-args="":
    chmod +x local.sh
    ./local.sh {{uvicorn-args}}

lint:
    @echo "Linting with Ruff..."
    poetry run --directory src/backend ruff check api/ search_engine/
    poetry run --directory src/backend ruff format --check --diff api/ search_engine/

mypy:
    @echo "Type checking with MyPy..."
    poetry run --directory src/backend mypy api/
    poetry run --directory src/backend mypy search_engine/

test args="":
    cd tests
    just -f tests/justfile test {{args}}