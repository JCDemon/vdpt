# VDPT MVP

This repository hosts a minimal FastAPI application that exposes a todo-focused API together with automated tests and CI configuration.

## Features

- **Health endpoint** to verify the service availability.
- **Todo endpoints** for listing, creating, and completing items stored in memory.
- **Pytest suite** that exercises the core API behaviour using FastAPI's `TestClient`.
- **GitHub Actions workflow** that installs dependencies and runs the tests on every push and pull request targeting `main`.

## Getting started

1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r backend/requirements.txt pytest
   ```

## Running the API locally

Use the helper script to start a local development server:

```bash
./dev-run.sh
```

The server listens on [http://localhost:8000](http://localhost:8000). The automatically generated OpenAPI docs are available at `/docs`.

## Running tests

Execute the test suite with:

```bash
pytest
```

## Project layout

```
backend/           # FastAPI application and domain services
backend/requirements.txt  # Runtime dependencies for the API
.github/workflows/ # GitHub Actions workflow configuration
tests/             # Pytest-based automated tests
```

## Additional notes

- The API stores data in memory to keep the MVP lightweight. Persistence can be added later.
- See [`SANITY.md`](SANITY.md) for manual verification steps.
