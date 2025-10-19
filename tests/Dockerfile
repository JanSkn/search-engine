FROM python:3.10-slim

# system dependencies
# build-essential for packages using c extensions
# git for packages installed from git
RUN apt-get update && apt-get install -y \
    build-essential \   
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install poetry

WORKDIR /app

# copy only dependency files first (for caching)
COPY src/backend/pyproject.toml .
COPY src/backend/poetry.lock .

# install dependencies (no-root so it doesn't try to install the package itself)
RUN poetry install --no-root

COPY src/backend/ .

CMD ["poetry", "run", "pytest"]