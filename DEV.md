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

* Python >= 3.10 (recommended: use **pyenv** to manage Python versions)
* Node.js >= 20.x
* npm >= 9.x
* Docker (for containerized integration/unit tests)

---

## Backend Setup

The backend uses **Poetry** for dependency management, **Just** for task automation, **mypy** for type checking, and **pytest** for tests.

### Mac

1. **Install pyenv (optional but recommended)**

```bash
brew install pyenv
pyenv install 3.10.0
pyenv global 3.10.0
python --version
```

2. **Install Poetry**

```bash
curl -sSL https://install.python-poetry.org | python3 -
poetry --version
```

3. **Install Just (task runner)**

```bash
brew install just
just --version
```

---

### Windows

1. **Install pyenv-win (optional but recommended)**

```powershell
choco install pyenv-win -y
```

* Install a specific Python version and set it globally:

```powershell
pyenv install 3.10.0
pyenv global 3.10.0
python --version
```

2. **Install Poetry**

```powershell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
poetry --version
```

3. **Install Just**

```powershell
choco install just -y
```

> 💡 Note for Windows users: To run Just commands properly, it is recommended to use Git Bash or WSL instead of the default PowerShell, as some recipes may rely on Unix-like shell behavior.

---

### Justfile Commands

The project uses Just as a task runner to automate common development tasks. Below is an overview of the available commands:

| Command            | Purpose                                          |
| ------------------ | ------------------------------------------------ |
| `build-frontend`   | Build the production frontend bundle             |
| `install-frontend` | Install frontend dependencies via npm            |
| `install-backend`  | Install backend dependencies via Poetry          |
| `install`          | Install both frontend and backend dependencies   |
| `local`            | Start frontend and backend for local development |
| `lint`             | Run Python style, formatting, and import checks  |
| `mypy`             | Run static type checks on Python code            |
| `test`             | Run integration/unit tests                       |

### Using Poetry

To manage Python dependencies, follow these steps:

1. **Add dependencies**

   ```bash
   # Add a regular dependency
   poetry add <package_name>

   # Add a development dependency
   poetry add --dev <package_name>
   ```

2. **Install all dependencies**

   ```bash
   poetry install
   ```

   * Installs all dependencies defined in `pyproject.toml` and `poetry.lock`.
   * Creates a virtual environment automatically (isolated from system Python).

3. **Activate the virtual environment**

   ```bash
   poetry shell
   ```

   * After this, `python` and `pip` commands will point to the virtual environment.
   * To exit, simply type:

     ```bash
     exit
     ```

4. **Check installed dependencies (optional)**

   ```bash
   poetry show          # List all installed packages
   poetry show --tree   # Show dependency tree
   ```

---

## Frontend Setup

Frontend uses **Node.js** and **npm**.

### Mac

```bash
brew install node
node -v
npm -v

# Install frontend dependencies
cd frontend
npm install
```

### Windows

1. **Simplest installation (recommended)**

```powershell
choco install nodejs-lts
node -v
npm -v
```

2. **Install frontend dependencies**

```powershell
cd frontend
npm install
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