# AskData

Chat with your business data. Upload a CSV or Excel file (or connect a database), ask a question in plain English, and get back a table, a chart and a short summary.

**Core rule:** the LLM interprets the question, and code computes the answer. The model only produces a structured query plan. The backend validates that plan and runs it with DuckDB/pandas, so every number shown comes from executed code.

Everything runs on free tools: SQLite, a local Ollama model, and the free tiers of Gemini or Groq.

> Status: **Milestone 3 (datasets)**. Upload CSV/XLSX files, get each column's type and a summary, and preview the rows. Builds on auth with workspace isolation (M2) and the skeleton (M1).

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
| `POST` | `/datasets` | Bearer | Upload a `.csv`/`.xlsx` (multipart `file`, optional `name`, `sheet`); max 20 MB |
| `GET` | `/datasets` | Bearer | Your workspace's datasets |
| `GET` | `/datasets/{id}` | Bearer | One dataset, with each column's type, kind, samples, null count and min/max |
| `GET` | `/datasets/{id}/preview?limit=50` | Bearer | First rows (up to 200) |
| `DELETE` | `/datasets/{id}` | Bearer | Delete the dataset and its files |

In `/docs`, click **Authorize** and sign in with your email and password to try the protected routes.

**Workspace isolation:** every query for company data goes through `workspace_scoped()` in [app/api/deps.py](app/api/deps.py). Anything outside your workspace returns 404, so you can't even tell whether it exists. [tests/test_isolation.py](tests/test_isolation.py) proves this.

## Datasets

Every upload is converted once to **Parquet**, and every query reads that file with DuckDB. Files are stored per workspace in `data/workspaces/{workspace_id}/datasets/{dataset_id}/`, as `original.csv`/`.xlsx` plus `data.parquet`.

- **CSV:** column types are detected by scanning the whole file. Files saved by Excel on Windows (cp1252) and files starting with a UTF-8 BOM are handled.
- **Excel (`.xlsx`):** the first sheet is used unless you pass `sheet`. Blank rows and blank unnamed columns are dropped, and a column mixing numbers and text is kept as text.
- **Column names:** trimmed; blanks become `column_N`; duplicates get `_2`, `_3`.
- **Limits:** 20 MB (`MAX_UPLOAD_MB`), 500 columns, 2 million rows. Oversized uploads are refused with 413 before the body is read.

Only the column summary (types, 3 sample values, min/max) is ever sent to the LLM, never the rows.

### Sample data

`sample_data/` holds synthetic demo data: `hr_employees.csv` (400 employees) and `sales_orders.csv` (3,000 orders, 2024-01 to 2026-09). It's generated deterministically, so the files are identical every time:

```bash
uv run python scripts/generate_sample_data.py
```

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

CI (GitHub Actions) runs ruff, mypy and pytest on every push, on SQLite **and** on a throwaway Postgres 17 container. It also checks that migrations match the models, then builds the Docker image and smoke-tests `/health`.

To run the tests against your own throwaway Postgres, set `TEST_DATABASE_URL`. **Never point it at real data**: tables are dropped between tests. Tests never use `DATABASE_URL` from `.env`.

## Configuration

Every setting is listed in [.env.example](.env.example). The main ones:

| Variable | Default | Notes |
|---|---|---|
| `DATABASE_URL` | `sqlite:///./data/askdata.db` | Paste a Postgres/Neon URL to switch, then run `uv run alembic upgrade head` |
| `LLM_PROVIDER` | `ollama` | `ollama`, `gemini` or `groq` |
| `LLM_MODEL` | `qwen2.5:7b-instruct` | Model name for the chosen provider |
| `SECRET_KEY` | — | **Must be changed** outside dev |
