# AI Chat Backend

FastAPI backend for an AI-powered chatbot SaaS.

This service provides:
- HTTP APIs
- LLM orchestration
- document ingestion and retrieval
- evaluation and observability

This repo intentionally starts minimal.

## Environment

- **`OPENAI_API_KEY`** (optional): If set, the app uses OpenAI chat completions instead of the stub. Get a key from [OpenAI API keys](https://platform.openai.com/api-keys).
- **`OPENAI_MODEL`** (optional): Model to use (default: `gpt-4o-mini`). Examples: `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`.

Without `OPENAI_API_KEY`, the backend uses a deterministic stub that echoes the user message.

**Keeping the key out of git:** Put secrets in a `.env` file in the backend repo root. The app loads `.env` on startup via `python-dotenv`. `.env` is in `.gitignore` and is never committed. Copy `.env.example` to `.env`, uncomment the line, and set your real key.

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
