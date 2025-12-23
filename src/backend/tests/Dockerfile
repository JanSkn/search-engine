FROM python:3.13-slim

# system dependencies
# build-essential for packages using c extensions
# git for packages installed from git
# libstemmer for CMake build of index_builder
RUN apt-get update && apt-get install -y \
    build-essential \   
    git \
    curl \
    cmake \
    just \
    libstemmer-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# set workdir to where pyproject.toml is located for uv
WORKDIR /app/src/backend

# copy first to cache dependencies
COPY pyproject.toml .
COPY uv.lock .
COPY bindings/ ./bindings/

RUN uv sync

COPY . .

RUN chmod +x /app/src/backend/tests/entrypoint.sh

ENTRYPOINT ["/app/src/backend/tests/entrypoint.sh"]