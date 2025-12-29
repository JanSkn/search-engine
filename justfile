build-frontend:
    cd src/frontend && npm run build

install-frontend:
    cd src/frontend && npm install

install-backend:
    cd src/backend && uv sync

uv-add *args:
    cd src/backend && uv add {{args}}

install:
    just install-frontend
    just install-backend

local *uvicorn-args:
    chmod +x local.sh && \
    ./local.sh {{uvicorn-args}}

deploy:
    chmod +x deploy.sh && \
    ./deploy.sh

build-index memory-limit="1024" max-docs="-1":
    cd src/backend/search_engine/scripts/ && \
    chmod +x build-index.sh && \
    ./build-index.sh {{memory-limit}} {{max-docs}}

remove-index-files:
    rm -rf src/backend/search_engine/index/bin
    rm -rf src/backend/search_engine/index_builder/data/docstore
    rm -rf src/backend/search_engine/index_builder/data/index
    rm -rf src/backend/search_engine/index_builder/data/partial_indices

# from installed package
# caution, will override existing stubs
generate-stubs:
    cd src/backend/bindings/ && \
    chmod +x generate-stubs.sh && \
    ./generate-stubs.sh

lint:
    @echo "Linting Python code..."
    cd src/backend && uv run ruff check api/ search_engine/ tests/
    cd src/backend && uv run ruff format --check --diff api/ search_engine/ tests/
    @echo "Linting C++ code..." # only format-check instead of linting to avoid dependency-related failures
    clang-format --dry-run --Werror \
    src/backend/bindings/utils.cpp \
        src/backend/search_engine/index_builder/index_builder.cpp \
        src/backend/search_engine/index_builder/merge_partial_indices.cpp

format:
    @echo "Formatting Python code..."
    cd src/backend && uv run ruff format api/ search_engine/ tests/
    @echo "Formatting C++ code..."
    clang-format -i \
        src/backend/bindings/utils.cpp \
        src/backend/search_engine/index_builder/index_builder.cpp \
        src/backend/search_engine/index_builder/merge_partial_indices.cpp

mypy:
    @echo "Type checking with MyPy..."
    cd src/backend && uv run mypy api/
    cd src/backend && uv run mypy search_engine/

test:
    just -f src/backend/tests/justfile test