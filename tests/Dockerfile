FROM python:3.13-slim

# system dependencies
# build-essential for packages using c extensions
# git for packages installed from git
# zlib and libstemmer for CMake build of index_builder
RUN apt-get update && apt-get install -y \
    build-essential \   
    git \
    curl \
    cmake \
    just \
    zlib1g-dev \
    libstemmer-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# set workdir to where pyproject.toml is located for uv
WORKDIR /app/src/backend

# copy first to cache dependencies
COPY src/backend/pyproject.toml .
COPY src/backend/uv.lock .
COPY src/backend/bindings/ ./bindings/

RUN uv sync

COPY src/backend/ .

COPY tests/ /app/tests/

COPY justfile /app/justfile

RUN chmod +x /app/tests/entrypoint.sh

ENTRYPOINT ["/app/tests/entrypoint.sh"]
