# Seekr - Setup Guide

This guide explains how to set up the development environment for **Seekr**.

## Table of Contents
* [Prerequisites](#prerequisites)
* [Backend Setup](#backend-setup)
* [Mac](#mac)
* [Windows](#windows)
* [Frontend Setup](#frontend-setup)
* [Mac](#mac-1)
* [Windows](#windows-1)
* [Running Tests](#running-tests)

---

## Prerequisites
* uv
* Node.js >= 20.x
* npm >= 9.x
* Docker (for containerized integration/unit tests)

---

## Backend Setup

The backend uses **uv** for Python version management and dependency management, **Just** for task automation, **mypy** for type checking, and **pytest** for tests.

### Mac

1. **Install uv**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version
```

2. **Install Just (task runner)**
```bash
brew install just
just --version
```

3. **Install Python and dependencies**
```bash
cd src/backend
uv sync
# or
just install-backend
```

---

### Windows

1. **Install uv**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
uv --version
```

2. **Install Just**
```powershell
choco install just -y
```

3. **Install Python and dependencies**
```powershell
cd src/backend
uv sync
# or
just install-backend
```

> 💡 Note for Windows: To run Just commands properly, it is recommended to use Git Bash or WSL instead of the default PowerShell, as some recipes may rely on Unix-like shell behavior.

---

### Justfile Commands

The project uses Just as a task runner to automate common development tasks. Below is an overview of the available commands:

| Command | Purpose |
| ------------------ | ------------------------------------------------ |
| `build-frontend` | Build the production frontend bundle |
| `install-frontend` | Install frontend dependencies via npm |
| `install-backend` | Install backend dependencies via uv |
| `install` | Install both frontend and backend dependencies |
| `local` | Start frontend and backend for local development |
| `lint` | Run Python style, formatting, and import checks |
| `mypy` | Run static type checks on Python code |
| `test` | Run integration/unit tests |

### Using uv

To manage Python dependencies and versions, follow these steps:

1. **Add dependencies**
```bash
cd src/backend
uv add <package_name>

# Add a development dependency
uv add --dev <package_name>
```

2. **Manage Python versions**
```bash
# Install a specific Python version
uv python install 3.13

# List installed Python versions
uv python list

# List all available Python versions
uv python list --all-versions
```

3. **Check installed dependencies (optional)**
```bash
uv pip list              # List all installed packages
uv tree                  # Show dependency tree
```

---

## Frontend Setup

Frontend uses **Node.js** and **npm**.

### Mac

```bash
brew install node
node -v
npm -v
```

### Windows

```powershell
choco install nodejs-lts
node -v
npm -v
```

---

## Running Tests

### Backend Unit Tests
```bash
just test
```

### Type Checking
```bash
just mypy
```

### Linting and Formatting
```bash
just lint
```