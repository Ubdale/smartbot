# AskData

Chat with your business data. Upload a CSV or Excel file (or connect a database), ask a question in plain English, and get back a table, a chart and a short summary.

**Core rule:** the LLM interprets the question, and code computes the answer. The model only produces a structured query plan. The backend validates that plan and runs it with DuckDB/pandas, so every number shown comes from executed code.

Everything runs on free tools: SQLite, a local Ollama model, and the free tiers of Gemini or Groq.

> Status: **Milestone 4 (ask questions)**. Ask in plain English and get a computed table, a chart, a summary and a "how this was calculated" panel, in a Next.js web app. Follow-up questions work. Builds on datasets (M3), auth with workspace isolation (M2) and the skeleton (M1).

## How it fits together

```
Browser ──> Next.js app (github.com/Ubdale/chatbot-fe, Vercel)
              │  /api/* forwarded (same origin: HttpOnly cookie, no CORS)
              v
            FastAPI (app/, Render) ──> Postgres/SQLite: users, datasets, chats, usage
              │
              ├─ LLM (Ollama locally / Gemini free tier): question -> JSON query plan
              │                                           result -> 2-3 sentence summary
              └─ DuckDB over Parquet: validated plan -> parameterised SQL -> the numbers
```

## Quick start (local)

Requires [uv](https://docs.astral.sh/uv/) (`pip install uv`). uv downloads Python 3.12 automatically.

```bash
cp .env.example .env
uv sync
uv run alembic upgrade head
uv run uvicorn app.main:app --reload
```

Open http://localhost:8000/health or http://localhost:8000/docs.

Then start the web app from its own repo, [Ubdale/chatbot-fe](https://github.com/Ubdale/chatbot-fe), in a second terminal:

```bash
git clone https://github.com/Ubdale/chatbot-fe.git && cd chatbot-fe
npm install && npm run dev     # http://localhost:3000, forwards /api to http://localhost:8000
```

### Local LLM (Ollama)

```bash
ollama pull qwen2.5:7b-instruct
```

- **Laptop with a small GPU (2-4 GB)?** Set `OLLAMA_NUM_GPU=0` in `.env`. Otherwise Ollama tries the GPU, runs out of memory and crashes ("CUDA error: out of memory").
- **On a CPU only**, a 7B model answers in roughly 15-90 seconds per question. `qwen2.5:3b-instruct` is about twice as fast but makes more planning mistakes. The hosted Gemini free tier (Milestone 5) answers in a few seconds.

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
| `POST` | `/chat/ask` | Bearer | `{question, dataset_id}` starts a conversation; `{question, conversation_id}` asks a follow-up |
| `GET` | `/chat/conversations` | Bearer | Your conversations (private to you) |
| `GET` | `/chat/conversations/{id}` | Bearer | A conversation with every answer as it was shown |
| `DELETE` | `/chat/conversations/{id}` | Bearer | Delete a conversation |
| `POST` | `/auth/logout` | — | Clears the session cookie |

In `/docs`, click **Authorize** and sign in with your email and password to try the protected routes.

**Browser sessions:** login also sets an HttpOnly `askdata_session` cookie (`SameSite=Lax`, `Secure` in prod). Writes authenticated by that cookie must send `X-AskData-CSRF: 1`, which other sites can't add, so cross-site request forgery is blocked. API clients use the Bearer token and don't need the header.

**Workspace isolation:** every query for company data goes through `workspace_scoped()` in [app/api/deps.py](app/api/deps.py). Anything outside your workspace returns 404, so you can't even tell whether it exists. [tests/test_isolation.py](tests/test_isolation.py) proves this.

## Datasets

Every upload is converted once to **Parquet**, and every query reads that file with DuckDB. Files are stored per workspace in `data/workspaces/{workspace_id}/datasets/{dataset_id}/`, as `original.csv`/`.xlsx` plus `data.parquet`.

- **CSV:** column types are detected by scanning the whole file. Files saved by Excel on Windows (cp1252) and files starting with a UTF-8 BOM are handled.
- **Excel (`.xlsx`):** the first sheet is used unless you pass `sheet`. Blank rows and blank unnamed columns are dropped, and a column mixing numbers and text is kept as text.
- **Column names:** trimmed; blanks become `column_N`; duplicates get `_2`, `_3`.
- **Limits:** 20 MB (`MAX_UPLOAD_MB`), 500 columns, 2 million rows. Oversized uploads are refused with 413 before the body is read.

Only the column summary (types, 3 sample values, min/max) is ever sent to the LLM, never the rows.

## Asking questions: how an answer is made

1. **Plan:** the LLM sees only the column summary and returns a JSON **query plan**: metric, groups, filters, a time phrase like "last quarter", sort, limit and chart. It never writes SQL. Follow-ups include the previous plans so the model can modify them.
2. **Validate:** every column must exist (typos get "did you mean…"), operators must fit the column type, and time phrases are turned into exact dates **in code** ([app/query/dates.py](app/query/dates.py)). If the plan is invalid, the model gets one retry with the exact error; after that the user is asked to rephrase. Ambiguous questions get a clarifying question back.
3. **Compute:** a fixed SQL template is filled in with quoted column names, and every value is passed as a DuckDB parameter. Queries have a 10 s timeout and a row cap, and results are cached by dataset version + plan ([app/query/executor.py](app/query/executor.py)).
4. **Summarise:** the LLM writes 2-3 sentences from the computed rows, plus highest/lowest facts computed by code. **Every number in the summary is checked against the result.** If any number can't be traced back to the result, the summary is replaced with one generated in code.
5. **Explain:** the response includes a plain-English description of what was computed, the date range used, and the SQL with its parameters.

Every LLM call is logged with its token counts in `usage_events`, per workspace and user.

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
| `OLLAMA_NUM_GPU` | unset | `0` = CPU only (fixes GPU out-of-memory crashes on small GPUs) |
| `OLLAMA_TIMEOUT_SECONDS` | `300` | How long to wait for the local model |
| `SECRET_KEY` | — | **Must be changed** outside dev |
