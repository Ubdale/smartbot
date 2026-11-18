# AskData

Chat with your business data. Upload a CSV or Excel file (or connect a database), ask a question in plain English, and get back a table, a chart and a short summary.

**Core rule:** the LLM interprets the question, and code computes the answer. The model only produces a structured query plan. The backend validates that plan and runs it with DuckDB/pandas, so every number shown comes from executed code.

Everything runs on free tools: SQLite, a local Ollama model, and the free tiers of Gemini or Groq.

> Status: **Milestone 2 (auth + workspaces)**. Sign-up, login (JWT) and per-company workspace isolation are in place, on top of the Milestone 1 skeleton.

## Quick start (local)

Requires [uv](https://docs.astral.sh/uv/) (`pip install uv`). uv downloads Python 3.12 automatically.

```bash
cp .env.example .env
uv sync
uv run alembic upgrade head
uv run uvicorn app.main:app --reload
```

Open http://localhost:8000/health or http://localhost:8000/docs.

## API so far

| Method | Path | Auth | What it does |
|---|---|---|---|
| `GET` | `/health` | — | App and database status |
| `POST` | `/auth/signup` | — | Creates a workspace for your company, with you as owner; returns a token |
| `POST` | `/auth/login` | — | Form fields `username` (your email) and `password`; returns a token |
| `GET` | `/auth/me` | Bearer | The signed-in user |
| `GET` | `/workspaces/me` | Bearer | Your workspace |
| `GET` | `/workspaces/me/members` | Bearer | Users in your workspace |
| `GET` | `/workspaces/{id}` | Bearer | 404 unless it's your workspace |

In `/docs`, click **Authorize** and sign in with your email and password to try the protected routes.

**Workspace isolation:** every query for company data goes through `workspace_scoped()` in [app/api/deps.py](app/api/deps.py). Anything outside your workspace returns 404, so you can't even tell whether it exists. [tests/test_isolation.py](tests/test_isolation.py) proves this.

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
