# AI Chat Backend

FastAPI backend for an AI-powered chatbot SaaS.

This service provides:
- HTTP APIs
- LLM orchestration
- document ingestion and retrieval
- evaluation and observability

This repo intentionally starts minimal.

## Run

```bash
pip install -e .
uvicorn app.main:app --reload
```

## Test

```bash
pytest
```