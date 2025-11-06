FROM python:3.13-slim

# system dependencies
# build-essential for packages using c extensions
# git for packages installed from git
RUN apt-get update && apt-get install -y \
    build-essential \   
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# set workdir to where pyproject.toml is located for uv
WORKDIR /app/src/backend

# copy only dependency files first (for caching)
COPY src/backend/pyproject.toml .
COPY src/backend/uv.lock .

RUN uv sync

COPY src/backend/ .
COPY tests/ /app/tests/

RUN mv /app/tests/entrypoint.sh /entrypoint.sh \
    && chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
