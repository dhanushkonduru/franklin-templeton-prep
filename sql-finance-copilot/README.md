# SQL Copilot for Financial Database

One-line summary
----------------

An AI-powered text-to-SQL copilot that converts natural-language financial questions into safe, executable SQL against a sample financial database (yfinance), with schema-aware prompting, validation, execution, and an automatic repair loop.

Short description
-----------------

This project demonstrates a production-style pipeline for building a domain-specific SQL Copilot for financial analytics. Users ask plain-English questions such as "What was the top-performing tech stock in 2023?" and the system returns both the answer and the audited SQL query used to compute it.

---

## 1. Project Title

SQL Copilot for Financial Database

- Project name: SQL Copilot for Financial Database
- One-line summary: Natural language → safe SQL → query results for financial analytics
- Short description: A demonstrator that combines schema retrieval, LLM-based SQL generation (Groq), SQL validation (sqlglot), safe execution (SQLAlchemy + SQLite or Postgres), and an LLM-based repair loop to produce auditable, reproducible answers to finance questions.

---

## 2. Original Project Prompt

SQL Copilot for Financial Database

What you build:
A text-to-SQL system over a sample financial database (using yfinance dataset). User asks questions like:
"What was the top-performing tech stock in 2023?"
The system writes and executes SQL, then returns the answer along with generated SQL for auditability.

Why this matters:
Internal analyst tooling is heavily used in financial institutions and asset managers.

Architecture Requirements:
- SQLite/Postgres database
- Schema-aware prompting
- Semantic schema retrieval
- SQL validation
- Query execution
- Repair loop for failed SQL

Stretch Goals:
- query repair
- visualization
- semantic retrieval

---

## 3. Features

- Natural language → SQL translation using a Groq LLM adapter
- Schema retrieval (schema grounding for prompts)
- SQL generation with strict system prompts
- SQL validation (SELECT-only, forbid DDL/DML, parse checks)
- SQL execution via SQLAlchemy (SQLite by default)
- Self-healing repair loop that asks the LLM to fix failing queries
- Financial reasoning rules encoded in prompts (returns, yearly aggregation)
- Minimal runtime CLI interaction (`scripts/run_pipeline.py`)
- Auditability: system returns generated SQL alongside results

---

## 4. Tech Stack

- Python — primary implementation language; clear dependency management and scripting.
- Groq — LLM provider used for SQL generation and repair. Chosen for a straightforward API and demonstrator purposes.
- SQLite (default) — lightweight local database for easy seeding and reproducible demos. Postgres is supported via `DATABASE_URL`.
- SQLAlchemy — DB abstraction and execution helper; provides consistent engine handling across SQLite/Postgres.
- sqlglot — SQL parsing and AST inspection for robust validation.
- pandas + yfinance — data collection and transformation for seeding the sample financial dataset.
- python-dotenv — environment variable loading for local development.

Why each technology:
- Python: fast iteration and rich ecosystem for data and dev tooling.
- Groq: used as the LLM in this demo; prompts and adapters are isolated so other providers can be swapped.
- SQLite: zero-config local DB for reviewers and recruiters to run locally.
- SQLAlchemy: reliable engine and execution model that ports easily to Postgres.
- sqlglot: provides grammar-agnostic parsing and the ability to assert the query is a `SELECT`.
- pandas/yfinance: produce realistic market data for demonstrations.

---

## 5. Project Architecture (high level)

User Question → Schema Retrieval → LLM SQL Generation → SQL Validation → SQL Execution → Repair Loop → Final Results

Detailed module responsibilities:
- `retrieval/` — maps a natural language question to the relevant database schema snippet used to ground the LLM prompt. This avoids table/column hallucination.
- `llm/` — contains adapters and prompt templates for Groq; generates SQL and repairs SQL when asked.
- `validation/` — enforces safety rules (SELECT-only, no DDL/DML, comment/unsafe-function blocking) and optionally enforces per-table allowed columns and `LIMIT` caps.
- `execution/` — runs SQL through a shared SQLAlchemy engine and converts results into JSON-friendly structures.
- `repair/` — orchestration wrapper around the LLM repairer that implements retry/backoff and decision logic for accepting repaired SQL.
- `database/` — shared engine loader with a default fallback to a local SQLite file for easy demos.
- `scripts/` — helper scripts: `seed_db.py` (seed SQLite), `scripts/seed_postgres.py` (Postgres seeder), `scripts/run_pipeline.py` (demo CLI pipeline)

This separation keeps the runtime surface small and the LLM-specific logic isolated and testable.

---

## 6. Folder Structure

Top-level (important files/folders):

```
src/sql_finance_copilot/
    retrieval/        # question → schema mapping (runtime wrapper)
    llm/              # Groq adapters, prompts, generator & repairer
    validation/       # SQL validators
    execution/        # SQL execution helpers
    repair/           # repair orchestration
    database/         # shared engine and URL helpers

scripts/
    run_pipeline.py   # preserved end-to-end pipeline (CLI demo)
    seed_postgres.py  # optional Postgres seeder

seed_db.py            # local sqlite seeder (yfinance)
finance.db            # generated demo SQLite DB (ignored in git ideally)
README.md
requirements.txt
pyproject.toml
```

Files of special note:
- `scripts/run_pipeline.py` — minimal end-to-end CLI that demonstrates retrieval → generation → validation → execution → repair.
- `seed_db.py` — seeds `daily_prices` and `stocks` tables using `yfinance` and `pandas`.
- `src/sql_finance_copilot/llm/sql_generator.py` — LLM prompt assembly and extraction helpers.
- `src/sql_finance_copilot/validation/sql_safety.py` — more advanced AST-based validator for production-readiness.

---

## 7. Database Schema

Two primary tables are used for the demo:

1) `daily_prices`

- `ticker` (TEXT) — stock ticker symbol
- `date` (DATE / TEXT) — trading date
- `open` (NUMERIC) — opening price
- `high` (NUMERIC) — highest intra-day price
- `low` (NUMERIC) — lowest intra-day price
- `close` (NUMERIC) — closing price
- `volume` (INTEGER) — traded volume

2) `stocks`

- `ticker` (TEXT) — primary key / join column
- `company_name` (TEXT)
- `sector` (TEXT)
- `industry` (TEXT)
- `market_cap` (NUMERIC)

Relationship: `daily_prices.ticker` JOINs to `stocks.ticker`. Typical analysis aggregates `daily_prices` by `ticker` (and by derived time windows such as YEAR via `STRFTIME` for SQLite or `EXTRACT` for Postgres).

---

## 8. Example Workflow (real example)

Question:

```
What was the top-performing tech stock in 2023?
```

Schema retrieved (excerpt):

```
Table: daily_prices
- ticker, date, open, high, low, close, volume

Table: stocks
- ticker, company_name, sector, industry, market_cap
Known values: sector: Tech
```

Generated SQL (example produced by the system):

```sql
SELECT s.ticker, s.company_name,
    (MAX(d.close) - MIN(d.close)) * 100.0 / MIN(d.close) AS return_pct
FROM daily_prices d
JOIN stocks s ON d.ticker = s.ticker
WHERE s.sector = 'Tech' AND STRFTIME('%Y', d.date) = '2023'
GROUP BY s.ticker, s.company_name
ORDER BY return_pct DESC
LIMIT 1;
```

Final Result (demo output):

- `ticker`: NVDA
- `company_name`: NVIDIA
- `return_pct`: ~253.37%

Why the SQL is financially correct
- We compute percentage return across the year using (MAX(close) - MIN(close)) / MIN(close) * 100 to measure total price appreciation over the period. This avoids noisy intraday calculation like (close - open) which can be misleading for long-term performance.
- The SQL groups by ticker and ranks by the computed percentage return, giving the top-performing stock over the selected time window.

---

## 9. SQL Validation and Security

The project enforces several safety rules before executing model-generated SQL:

- SELECT-only enforcement — queries must parse as a `SELECT` expression (no DDL/DML). Implemented with `sqlglot` parse checks.
- Forbidden keywords — statements that include `DROP`, `DELETE`, `UPDATE`, `INSERT`, `ALTER`, `TRUNCATE`, or other write operations are rejected early.
- Multi-statement rejection — multiple `;` separated statements are rejected to avoid command chaining.
- Comment and unsafe function blocking — comments or calls to server-side functions (e.g., `pg_sleep`) are blocked in the simpler validator.
- Schema grounding — the retrieval step returns a concrete schema snippet that is included in the LLM prompt so the model is anchored to actual table/column names.

These measures together reduce the attack surface for malicious or accidental destructive SQL and make the system safer to run against real data.

---

## 10. Repair Loop

Motivation
- LLMs can output syntactically valid SQL that nonetheless fails at execution due to dialect differences, small typos, or incorrect assumptions about column names. A repair loop asks the model to fix the failing SQL while preserving intent.

How it works
1. Run the generated SQL against the DB. If execution succeeds, return results.
2. If execution fails, capture the DB error message and the original SQL.
3. Send a repair prompt to the LLM that includes the failed SQL, the DB error text, and the schema context.
4. Receive repaired SQL, validate it again, and attempt execution.
5. Repeat with a bounded number of attempts (e.g., 1-3) and fail safe if the LLM cannot produce a useful repair.

Example repair (dialect mismatch):

- Failed SQL (LLM used Postgres `EXTRACT` but DB is SQLite):

```sql
SELECT ticker FROM daily_prices WHERE EXTRACT(YEAR FROM date) = 2023;
```

- Error from SQLite: `near "EXTRACT": syntax error`.

- Repaired SQL produced by the model:

```sql
SELECT ticker FROM daily_prices WHERE STRFTIME('%Y', date) = '2023';
```

The repair loop enforces the same validation rules on the repaired SQL before attempting the execution.

---

## 11. Challenges Faced

This project surfaced practical engineering challenges and the fixes we applied:

- SQLite vs Postgres dialect mismatch — LLMs will sometimes prefer Postgres syntax (e.g., `EXTRACT`) while the demo DB is SQLite. Fix: include dialect-specific rules in prompts and implement repair guidance (e.g., STRFTIME replacement) and use the `DATABASE_URL` environment variable to target Postgres when desired.
- Schema hallucinations — models invent table/column names. Fix: supply a schema snippet from a retrieval step and include a clear rule in the system prompt: "Use only tables and columns listed in the schema context." Also add a schema-aware retriever to surface realistic table/column names.
- Financial semantic correctness — naive percentage calculations are wrong for multi-day/time-window comparisons. Fix: include explicit financial reasoning rules in prompts (use MAX/MIN across the period, group by ticker, rank by percentage return) so the model follows domain best practices.
- yfinance MultiIndex column issues — `yfinance` sometimes returns MultiIndex columns (e.g., `Adj Close`). Fix: flatten column names and normalize with a consistent set (`lowercase` + underscore) during seeding.
- Runtime path / packaging problems — early scripts imported mixed legacy and modern modules. Fix: consolidate runtime exports (`retrieval`, `llm`, `validation`, `execution`, `repair`, `database`) and keep `scripts/run_pipeline.py` as the preserved end-to-end surface.
- Environment variable pitfalls — missing `GROQ_API_KEY` or `DATABASE_URL` can cause runtime failures. Fix: use `python-dotenv` for local `.env` files and provide clear README guidance.
- Schema drift — if the production schema changes, prompts and stored schema snippets can become stale. Fix: add dynamic schema introspection utilities in `db/introspection.py` (future improvement: auto-refresh the index used by the retriever).

---

## 12. Key Learnings

- AI systems engineering requires careful orchestration of retrieval, prompting, validation, and execution; each component reduces risk and increases reliability.
- Grounding prompts with concrete schema is essential to reduce hallucination.
- Prompt engineering is a core safety tool — including explicit negative instructions (e.g., "do not use DDL/DML") reduces dangerous outputs.
- `sqlglot` is an effective tool for structural SQL checks and language-agnostic parsing.
- Repair loops improve robustness but should be bounded and auditable (we store and log attempted repairs and the errors that triggered them).
- Tests benefit from LLM client mocking to keep CI fast and deterministic.

---

## 13. Future Improvements

- Streamlit UI for interactive question entry and charting of results.
- Semantic retrieval with embeddings + FAISS for richer schema selection and content-based retrieval.
- Dynamic schema introspection and automated refresh of the retrieval index.
- Chart generation and export (Plotly) for richer analyst deliverables.
- A more powerful validator that checks column-level access, enforces `LIMIT` policies, and supports role-based restrictions.
- Multi-database support and explicit Postgres-focused code paths for production deployments.
- An evaluation framework: store human labels, track correctness and hallucination rates, and run periodic prompt regression tests.
- Containerization (Docker) and CI integration for reproducible demos.

---

## 14. Installation Instructions

Minimal steps to run locally on macOS / Linux. These commands assume you are at the repository root.

1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies

```bash
pip install -r requirements.txt
# or editable install
pip install -e .
```

3. Create a `.env` file in the repo root with at least:

```
DATABASE_URL=sqlite:///finance.db
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```

4. Seed the local database (sqlite)

```bash
python seed_db.py
```

5. Run the preserved pipeline (demo)

```bash
PYTHONPATH=src python scripts/run_pipeline.py
```

6. Run tests (LLM calls are mocked in tests)

```bash
pytest -q
```

---

## 15. Example Commands

```bash
# Seed sqlite demo db
python seed_db.py

# Quick demo run (uses Groq if GROQ_API_KEY is set)
PYTHONPATH=src python scripts/run_pipeline.py

# Run tests
pytest -q

# Run API server (FastAPI) if you want the HTTP surface
PYTHONPATH=src uvicorn sql_finance_copilot.api.main:app --reload --port 8000

# Run Streamlit frontend
PYTHONPATH=src streamlit run src/sql_finance_copilot/app/streamlit_app.py
```

---

## 16. Example Questions

- "What was the top-performing tech stock in 2023?"
- "Show annual revenue and net income for AAPL for the last 5 years."
- "What was the average daily volume for MSFT in 2023?"
- "List the top 5 stocks by year-over-year return for 2022."

---

## 17. Conclusion

What was achieved
- A compact, auditable pipeline that demonstrates how to combine schema retrieval, prompt engineering, SQL validation, execution, and automatic repair to make LLM-generated SQL safer and more reliable for financial analysis.

Why it matters
- Analysts and decision-makers need tools that translate plain-language questions into correct, auditable queries — this project illustrates one pragmatic approach to achieve that while integrating practical engineering safeguards.

What makes the system interesting technically
- The system blends retrieval-grounded prompting, structured AST-based validation, and a bounded LLM repair loop — together these components form a practical strategy for reducing hallucination and runtime errors while keeping the generated SQL interpretable and auditable.

---

If you want, I can: update the repository README with a demo GIF, add a Dockerfile for reproducible demos, or create a short demo video script — tell me which next step you prefer.
# SQL Finance Copilot

Production-style SQL copilot for finance teams. The system converts natural-language questions into safe PostgreSQL queries, retrieves only the relevant schema context, validates generated SQL, executes read-only queries, retries failures with automatic repair, and optionally recommends charts.

## Architecture

The code is organized as a layered system:

- `schema/` handles PostgreSQL introspection, embeddings, FAISS indexing, and semantic retrieval.
- `llm/` handles prompt construction, Groq calls, SQL generation, and repair.
- `validation/` enforces SQL safety rules before execution.
- `db/` owns engine creation, execution, and safe result materialization.
- `core/` coordinates the end-to-end workflow.
- `api/` exposes a FastAPI service.
- `app/` contains the Streamlit UI.
- `charts/` contains chart inference logic.

## Workflow

1. Introspect the database schema and build or load a semantic index.
2. Retrieve only the most relevant schema tables for the user question.
3. Generate SQL from the question and retrieved schema context.
4. Validate the SQL with `sqlglot` and a strict safety policy.
5. Execute the SQL with a hard row cap.
6. If execution fails, repair the SQL and retry.
7. Return SQL, rows, and optional chart guidance for auditability.

## Run

Create a virtual environment, install dependencies, and copy `.env.example` to `.env`.

```bash
pip install -e .
uvicorn sql_finance_copilot.api.main:app --reload
streamlit run src/sql_finance_copilot/app/streamlit_app.py
```

## Notes

- The application is designed for read-only analytics workflows.
- Start with a narrow database role that has `SELECT` access only.
- Rebuild the schema index whenever your database schema changes materially.
