# AskData

Chat with your business data. Upload a CSV or Excel file (or connect a database), ask a question in plain English, and get back a table, a chart and a short summary.

**Core rule:** the LLM interprets the question, and code computes the answer. The model only produces a structured query plan. The backend validates that plan and runs it with DuckDB/pandas, so every number shown comes from executed code.

Everything runs on free tools: SQLite, a local Ollama model, and the free tiers of Gemini or Groq.

> Status: **Milestone 1 (skeleton)**. Health check, config, DB + migrations, tests, Docker and CI are in place.

## Quick start (local)

Requires [uv](https://docs.astral.sh/uv/) (`pip install uv`). uv downloads Python 3.12 automatically.

```bash
cp .env.example .env
uv sync
uv run alembic upgrade head
uv run uvicorn app.main:app --reload
```

Open http://localhost:8000/health or http://localhost:8000/docs.

## With Docker

```bash
docker compose up --build                     # API only
docker compose --profile ollama up --build    # API + local Ollama
docker compose exec ollama ollama pull qwen2.5:7b-instruct
```

## Development

```bash
uv run pytest            # tests
uv run ruff check .      # lint
uv run ruff format .     # format
uv run mypy              # type check
uv run alembic revision --autogenerate -m "message"   # new migration
```

CI (GitHub Actions) runs ruff, mypy and pytest on every push, then builds the Docker image and smoke-tests `/health`.

## Configuration

Every setting is listed in [.env.example](.env.example). The main ones:

| Variable | Default | Notes |
|---|---|---|
| `DATABASE_URL` | `sqlite:///./data/askdata.db` | Set a Postgres URL to switch. |
| `LLM_PROVIDER` | `ollama` | `ollama`, `gemini` or `groq` |
| `LLM_MODEL` | `qwen2.5:7b-instruct` | Model name for the chosen provider |
| `SECRET_KEY` | — | **Must be changed** outside dev |
