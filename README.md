# AI Chat Backend

FastAPI backend for an AI-powered chatbot SaaS.

This service provides:
- HTTP APIs
- LLM orchestration
- document ingestion and retrieval
- evaluation and observability

This repo intentionally starts minimal.

## Run

Default dev run script:

```bash
./scripts/run-dev.sh
```

With Poetry:

```bash
poetry install
poetry run ./scripts/run-dev.sh
```

With pip/venv:

```bash
pip install -e ".[dev]"
./scripts/run-dev.sh
```

## Test

```bash
poetry run pytest
# or, with venv activated: python -m pytest
```
