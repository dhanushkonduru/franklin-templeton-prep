# Franklin Templeton AI/ML Interview Prep

> Twelve production-style AI/ML projects that mirror the systems Franklin Templeton's AI Platform team builds for investment research, compliance, ESG, portfolio analytics, and real-time operations.

## What This Is

This repository is a hands-on portfolio built to prepare for a Digital & AI/ML Engineer role at Franklin Templeton, a global asset manager with roughly $1.66 trillion in assets under management. Each project maps to a real problem the AI Platform organization solves: turning filings and research into grounded answers, scoring language in earnings calls, comparing year-over-year disclosures, running analyst copilots over structured data, and shipping models with tests, containers, and monitoring.

The work spans the full stack an AI platform engineer touches: retrieval-augmented generation (RAG), NLP, multi-agent orchestration with LangGraph, classical ML with proper backtesting, fine-tuning, MLOps, and deployment. Projects are self-contained folders with their own READMEs, requirements, and run instructions. Together they show how to go from a notebook idea to something you could demo to a portfolio manager or an AI platform lead.

Franklin Templeton's AI Platform team (led by Vasundhara Chetluru) focuses on trustworthy AI inside regulated finance: answers must cite sources, numbers must be verifiable, and systems must be observable in production. These projects practice that bar. Where a folder is a "capability slice" of a larger system (for example, Docker deployment lives inside `financial-rag`, and MLOps lives inside `portfolio-ml`), the sections below call that out explicitly so the twelve-project narrative stays aligned with the portfolio document in this repo.

## Projects Overview

| # | Project | What It Does | Key Tech | What I Learned |
|---|---------|--------------|----------|----------------|
| 1 | Financial Document Intelligence RAG | Ask plain-English questions about 10-K filings; get cited answers with confidence levels | LangChain, BGE embeddings, ChromaDB, BM25, Cohere Rerank, Groq, FastAPI, Streamlit | Hybrid retrieval (dense + BM25) matters for tickers and exact figures that pure vector search misses |
| 2 | Earnings Call Sentiment & Surprise Detector | Score transcript tone with FinBERT and compare it to stock reaction around earnings | FinBERT, rule-based hedging, SQLite, yfinance, event study | The interesting signal is divergence: positive language with a negative price move |
| 3 | Multi-Agent Investment Research Assistant | Specialist agents build and critique an investment thesis for a ticker | LangGraph, Pydantic state, Groq, hybrid retrieval, SQLite audit log | A critic loop with routing beats a single-shot "write a report" prompt for quality control |
| 4 | Portfolio Construction with Classical ML | Predict cross-sectional returns and backtest a long-short book with walk-forward validation | XGBoost, PyTorch MLP, yfinance, expanding-window backtest | Random k-fold leaks future information; walk-forward validation and transaction costs change the story |
| 5 | ESG Document Analyzer | Extract metrics from sustainability PDFs, classify disclosures, flag greenwashing, score ESG | PyMuPDF, regex + Groq LLM, Pydantic, Streamlit, Plotly | Regex-first extraction with LLM fallback keeps cost down while still handling messy PDF layouts |
| 6 | Production FastAPI + Docker Deployment | Containerized RAG API behind nginx with persistent vector storage | Docker, docker-compose, nginx, ChromaDB service, multi-stage Dockerfile | Separating app, vector DB, and reverse proxy is the minimum viable production shape for a RAG service |
| 7 | Hallucination Detection & Eval Harness | Measure whether RAG answers are grounded in annual report text | Qdrant, LlamaIndex, NLI verifier, RAGAS, BLEU/ROUGE, Streamlit | Evaluators disagree (~22% agreement on saved runs); NLI catches unsupported numbers heuristics miss |
| 8 | SQL Copilot for Financial Database | Natural language to safe, auditable SQL over market data | Groq, sqlglot, SQLAlchemy, FAISS schema retrieval, FastAPI, Streamlit | Schema grounding plus a bounded repair loop fixes dialect errors without opening DDL/DML risk |
| 9 | Fine-Tune Mistral-7B with QLoRA | Teach concise finance Q&A style on a free T4 GPU | Unsloth, QLoRA, LoRA/PEFT, SFTTrainer, Mistral-7B, Groq (eval) | Fine-tuning shaped behavior (100% concise format) more than raw keyword overlap (~49% across models) |
| 10 | Real-Time News + Filing Alert System | Poll news and SEC feeds, dedupe, classify, match portfolio, alert | FastAPI, asyncio, Redis Streams, PostgreSQL, sentence-transformers, Docker | Embedding dedup at 0.92 similarity stops duplicate alerts when ten outlets republish the same headline |
| 11 | MLOps Pipeline for Portfolio ML | Track experiments, promote models, monitor drift, CI train on push | MLflow, DVC, Evidently, GitHub Actions, FastAPI inference | A model registry stage ("Production") plus drift reports makes the classical ML path operable, not just research |
| 12 | Long-Context Document Comparator | Diff two 10-Ks section-by-section and surface material changes for analysts | PyMuPDF, text diff, Groq semantic analysis, React dashboard | Section-level processing controls token cost and lets you use semantic analysis only where diff alone is blind |

**Also in repo:** `LangGraph/` is a compact three-node LangGraph tutorial (data, sentiment, report) that informed Project 3.

## Projects In Detail

### 1. Financial Document Intelligence RAG

**What we built:** A question-answering system over SEC 10-K style filings. You ask a question in normal English, and the system finds the right passages, sends them to an LLM, and returns an answer with file-and-page citations plus a HIGH/MEDIUM/LOW confidence label. It is built to refuse when the documents do not contain enough evidence.

**Why it matters at Franklin Templeton:** Portfolio managers and research analysts spend hours in filings. A grounded RAG layer lets them query risk factors, segment revenue, or policy changes without reading hundreds of pages manually, as long as every claim traces back to a source.

**How it works (simply):**
- Filings land in `data/raw/` (Apple, Microsoft, Tesla in the demo set).
- PDFs and HTML are parsed into clean text and tables.
- Long documents are cut into chunks (800 tokens, 100 overlap).
- Each chunk is turned into a vector (embedding) and stored in ChromaDB.
- At query time, dense vector search and BM25 keyword search run in parallel, then scores are merged.
- Cohere Rerank reorders the top candidates for relevance.
- The best chunks plus the question go to Groq (`llama-3.3-70b-versatile`) with strict citation rules.
- The API returns answer text, sources, and confidence; Streamlit provides a UI.

**Stack:** Python, LangChain/LCEL, `BAAI/bge-large-en-v1.5`, ChromaDB, BM25, Cohere Rerank, Groq, FastAPI, Streamlit, loguru

**Key thing learned:** Financial retrieval is not "embeddings only." BM25 catches exact strings like ticker symbols and dollar amounts that dense search can rank poorly. Combining both before reranking is the difference between plausible and auditable answers.

**Run:** `cd financial-rag` → `pip install -r requirements.txt` → `python ingest.py` → `uvicorn api.main:app --reload` and `streamlit run ui/app.py`

---

### 2. Earnings Call Sentiment and Surprise Detector

**What we built:** A pipeline that reads earnings call transcripts, splits them into sentences, scores tone and language patterns, stores everything in a database, and compares the text signal to how the stock moved around the earnings date. The output highlights when management sounds upbeat but the market sells off.

**Why it matters at Franklin Templeton:** Earnings season is a core workflow for equity research. Automating transcript scoring surfaces "surprise" cases where language and price action disagree, which is often where human analysts focus first.

**How it works (simply):**
- Transcript text is ingested (scraper or sample data).
- A parser splits remarks vs Q&A and labels sections.
- FinBERT scores each sentence positive, neutral, or negative.
- Rule patterns flag hedging ("we believe", "we may") and forward-looking vs historical phrasing.
- Scores persist in SQLite via SQLAlchemy.
- An event study pulls prices from yfinance and computes abnormal returns around the event window.
- A runner ties transcript aggregates to market reaction and flags alignment vs divergence.

**Stack:** Python, FinBERT (`yiyanghkust/finbert-tone`), regex hedging taxonomy, SQLAlchemy, SQLite, pandas, NumPy, SciPy, yfinance

**Key thing learned:** FinBERT understands finance phrasing ("headwinds", "beat consensus") better than generic sentiment models, but hedging is lexically explicit, so rules outperform a second ML model for that slice.

**Run:** `cd earnings_detector` → install requirements → `python scripts/run_pipeline.py`

---

### 3. Multi-Agent Investment Research Assistant

**What we built:** Give a stock ticker and a small team of AI agents collaborates: one pulls market data and headlines, one runs retrieval over transcript-like context, others analyze fundamentals, sentiment, and risk, then a writer drafts a thesis and a critic can send the draft back for revision. Every step appends to an SQLite audit log.

**Why it matters at Franklin Templeton:** Investment research is multi-step and multi-source. A graph of specialists with shared state mirrors how analysts divide work (data, fundamentals, sentiment, risk, editorial review) instead of one monolithic chat.

**How it works (simply):**
- `ResearchState` (Pydantic) holds ticker, prices, news, retrieval context, analyst notes, draft report, and critic feedback.
- LangGraph nodes run in order: data gatherer → retrieval → fundamental → sentiment → risk → report writer → critic.
- Routing sends weak reports back to the writer instead of ending immediately.
- Tools wrap yfinance, RSS headlines, and the hybrid retrieval stack (chunk, embed, BM25+dense, rerank, compress).
- `main.py` runs a demo on `NVDA` by default.

**Stack:** Python, LangGraph, Pydantic, Groq, SQLite audit store, yfinance, feedparser, langchain retrieval utilities

**Key thing learned:** Multi-agent is really multi-step orchestration with typed state. The critic loop is what makes the system feel like a review process rather than a single prompt dressed up as agents.

**Run:** `cd investment-research` → install deps → `python main.py`

**Precursor:** See `LangGraph/` for the minimal three-node graph (data → sentiment → report) used to learn LangGraph patterns before this full pipeline.

---

### 4. Portfolio Construction with Classical ML

**What we built:** A research pipeline that downloads prices and fundamentals for a basket of large-cap stocks, engineers momentum and quality-style features, trains models to predict the next 21 trading-day return, and simulates a long-short portfolio (top decile long, bottom decile short) under expanding-window walk-forward validation.

**Why it matters at Franklin Templeton:** Quant and portfolio teams constantly test whether simple ML signals add value after costs and regime change. This project practices the research discipline (no look-ahead, walk-forward, Sharpe/drawdown/IR) not just model accuracy on a static split.

**How it works (simply):**
- yfinance pulls adjusted closes for tickers like AAPL, MSFT, NVDA, XOM, etc.
- Features include 1-day return, 3m/12m momentum, volatility, P/E, ROE, debt/equity, revenue growth.
- Target = forward 21-day return.
- XGBoost (sklearn Pipeline with imputation/scaling) and a small PyTorch MLP train on past years only.
- Each test year is predicted using all prior history (expanding window).
- Predictions rank into long/short sleeves; performance metrics include Sharpe, max drawdown, information ratio.
- KMeans adds a lightweight regime label on market conditions.

**Stack:** Python, XGBoost, scikit-learn, PyTorch, yfinance, pandas, NumPy

**Key thing learned:** On a recent full run, XGBoost reached Sharpe ~0.90 vs MLP ~0.73, but max drawdown was still severe (-90% range), which reinforces that validation design and costs matter more than picking the fancier model.

**Run:** `cd portfolio-ml` → `pip install -r requirements.txt` → `python main.py` (artifacts under `output/`)

---

### 5. ESG Document Analyzer

**What we built:** An NLP pipeline for sustainability reports: ingest PDFs, segment Environmental/Social/Governance sections, extract metrics with regex, fall back to an LLM when rules miss fields, classify against SASB/GRI-style keywords, estimate greenwashing risk, and output a comparable ESG scorecard plus a Streamlit dashboard.

**Why it matters at Franklin Templeton:** ESG integration is standard in institutional mandates. Automating extraction and greenwashing checks helps analysts compare issuers at scale instead of manually reading inconsistent report formats.

**How it works (simply):**
- PyMuPDF extracts text; cleaning and chunking follow.
- Segmenter splits E, S, and G blocks.
- Regex pulls common metrics first (cheaper, deterministic).
- Groq LLM fills structured gaps when `USE_LLM_EXTRACTION=true`.
- Taxonomy classifier maps language to disclosure categories.
- Greenwashing module compares bold claims vs measurable disclosures.
- Weighted scoring engine produces final ESG score and JSON exports.

**Stack:** Python 3.11+, PyMuPDF, Pydantic, Groq, pandas, Streamlit, Plotly, pytest

**Key thing learned:** ESG PDFs are layout-noisy; a regex-first pipeline with LLM fallback is the right cost/latency tradeoff for production, not LLM-on-every-page.

**Run:** `cd esg-analyzer` → install requirements → place PDFs in `data/reports/` → `python main.py` → `streamlit run dashboard/dashboard.py`

---

### 6. Production-Grade FastAPI + Docker Deployment

**What we built:** A deployable packaging of the financial RAG API: multi-stage Docker image, docker-compose stack with the FastAPI app, ChromaDB as a separate service, and nginx as a reverse proxy with health checks and restart policies.

**Why it matters at Franklin Templeton:** Research prototypes only matter if they can run reliably behind internal infrastructure: separate processes for app and vector store, health endpoints, and a stable ingress path for other teams to call.

**How it works (simply):**
- Dockerfile builder stage installs dependencies; runtime stage copies only app code.
- `docker-compose.yml` wires `app`, `chromadb`, and `nginx` on an internal network.
- Chroma persists vectors in a named volume.
- nginx proxies port 80 to the app with longer read timeout for LLM latency.
- Environment variables point the app at the Chroma host inside the compose network.

**Stack:** Docker, docker-compose, nginx, FastAPI/uvicorn, ChromaDB official image, Python 3.11

**Key thing learned:** Treat the vector database as its own service with a volume, not a sidecar file inside the app container, or you lose persistence and clean scaling boundaries.

**Run:** `cd financial-rag` → configure `.env` → `docker compose up --build` (API on port 8000, nginx on port 80)

---

### 7. Hallucination Detection and Eval Harness

**What we built:** A financial RAG test bench over processed annual report text (Goldman, Morgan Stanley, BlackRock samples). It ingests chunks into Qdrant, generates answers with Groq, then scores grounding with NLI, RAGAS, heuristics (BLEU/ROUGE), and a dedicated hallucination classifier, with comparison reports and a Streamlit inspector.

**Why it matters at Franklin Templeton:** In asset management, a wrong revenue number is worse than a vague paragraph. This harness makes failure modes visible (intrinsic vs extrinsic hallucinations, evaluator disagreement) before a copilot reaches analysts.

**How it works (simply):**
- Processed `.txt` filings are chunked and embedded (`BAAI/bge-small-en-v1.5`).
- Vectors live in local Qdrant.
- Retriever + cross-encoder reranker fetch context.
- Groq generates the answer from retrieved passages only.
- NLI model labels entailment, neutral, or contradiction vs context.
- `compare_evaluators.py` runs RAGAS, heuristic, and classifier evaluators side by side.
- `analysis_report.py` writes precision/recall/F1 and disagreement examples to `results/`.

**Stack:** Python, LlamaIndex, Qdrant, HuggingFace embeddings, SentenceTransformers reranker, Groq, Transformers NLI, RAGAS, Streamlit

**Key thing learned:** On the saved 90-sample run, NLI precision was 1.0 but recall ~0.36 (F1 ~0.53), and evaluators agreed only ~22% of the time. Trust requires multiple checks, not one metric.

**Run:** `cd hallucination-eval-harness` → `python rag/ingest.py` → `python generate_dataset.py` → `python compare_evaluators.py` → `python analysis_report.py` → `streamlit run app.py`

---

### 8. SQL Copilot for Financial Database

**What we built:** A text-to-SQL copilot over a seeded finance database (daily prices + stock metadata from yfinance). Analysts ask questions in English; the system retrieves relevant schema, generates SQL with Groq, validates it with sqlglot (SELECT-only, no DDL/DML), executes via SQLAlchemy, and retries with an LLM repair loop when execution fails.

**Why it matters at Franklin Templeton:** Internal analysts live in SQL against positions, risk, and market tables. A copilot is useful only if it returns the query text for audit and refuses unsafe statements.

**How it works (simply):**
- `seed_db.py` builds SQLite (or Postgres via `DATABASE_URL`) with `daily_prices` and `stocks`.
- Semantic schema retrieval (embeddings + FAISS) picks tables/columns for the question.
- LLM prompt includes schema snippet and finance rules (e.g., yearly return from MIN/MAX close, not noisy open-close).
- sqlglot validates structure and blocks writes/multi-statements.
- Executor runs with row caps; failures trigger repair prompts with the DB error message.
- FastAPI and Streamlit surfaces return SQL + rows (+ optional chart hints).

**Stack:** Python, Groq, SQLAlchemy, sqlglot, SQLite/Postgres, FAISS schema index, FastAPI, Streamlit, pandas, yfinance

**Key thing learned:** The demo top tech stock in 2023 example returned NVDA with ~253% return only after dialect-aware repair (SQLite `STRFTIME` vs Postgres `EXTRACT`). Grounding schema in the prompt mattered as much as the repair loop.

**Run:** `cd sql-finance-copilot` → `pip install -e .` → set `GROQ_API_KEY` → `python seed_db.py` → `PYTHONPATH=src python scripts/run_pipeline.py`

---

### 9. Fine-Tune Mistral-7B with QLoRA

**What we built:** A Colab-friendly fine-tuning pipeline on 183 finance Q&A pairs (160 train / 23 test). Mistral-7B-Instruct is loaded in 4-bit (QLoRA), LoRA adapters train on ~0.1% of weights, and evaluation compares fine-tuned, base, and Groq cloud answers on keyword match, length, and finance vocabulary density.

**Why it matters at Franklin Templeton:** Not every problem needs RAG. When you need consistent tone, concise answers, or domain phrasing at low latency, adapter fine-tuning on a small open model can be cheaper than calling a large API on every request.

**How it works (simply):**
- Dataset covers valuation, statements, instruments, corporate finance, markets, and risk concepts.
- Training examples use Mistral `[INST]` format for instruction tuning.
- Unsloth + BitsAndBytes shrink VRAM from ~14GB to ~4GB on a T4.
- LoRA rank 16 on attention projections; 3 epochs, effective batch 8.
- Training loss moved from ~3.1 toward ~0.24 in the documented run.
- Evaluation JSON compares three models on held-out pairs.

**Stack:** Mistral-7B-Instruct-v0.3, Unsloth, QLoRA/BitsAndBytes, PEFT/LoRA, TRL SFTTrainer, Google Colab T4, Groq API (baseline), Google Drive for checkpoints

**Key thing learned:** Keyword match was ~49-51% across all models, but the fine-tuned model hit 100% concise format (34 words avg vs 117 for base) and higher finance term density (6.1% vs 5.0%). SFT taught style and behavior, not new facts.

**Run:** Open `finance_lora_project/finance_lora.ipynb` in Google Colab (GPU required); follow cell order in `finance_lora_project/README.md`

---

### 10. Real-Time News + Filing Alert System

**What we built:** An async service that polls NewsAPI, SEC EDGAR RSS, and Alpaca news on a timer, pushes events through Redis Streams, deduplicates similar headlines with embeddings (threshold 0.92), classifies event type (earnings, M&A, regulation, etc.), matches against portfolio holdings, and dispatches webhook/email alerts with Postgres history and latency metrics.

**Why it matters at Franklin Templeton:** Portfolio managers cannot manually watch every headline for every holding. A streaming alert fabric with dedup and portfolio-aware matching is how operations teams surface what matters in minutes, not hours.

**How it works (simply):**
- Worker polls sources every ~30 seconds (configurable).
- Normalized events enter Redis Stream `news_stream`.
- Pipeline: normalize → embed → dedupe → classify → match holdings → persist → alert if matched and not duplicate.
- FastAPI exposes health, metrics, holdings CRUD, and manual ingest for demos.
- Failed deliveries land on a replay stream for retry.
- Docker Compose runs API worker, Postgres, and Redis together.
- 30 automated pytest cases cover dedup, matching, streams, and delivery.

**Stack:** Python 3.12, FastAPI, asyncio/aiohttp, Redis Streams, PostgreSQL, SQLAlchemy async, sentence-transformers, scikit-learn cosine similarity, Docker Compose

**Key thing learned:** End-to-end latency is often dominated by the poll interval (~30s), while in-pipeline processing is sub-second per event. Embedding dedup is what prevents alert fatigue when many outlets copy the same story.

**Run:** `cd news-alert-system` → copy `.env.example` to `.env` → `docker compose up --build` → demo via http://localhost:8000/docs

---

### 11. MLOps Pipeline for Portfolio ML

**What we built:** Operational layers on top of Project 4: MLflow experiment tracking and model registry promotion, DVC for data versioning, Evidently drift reports, a FastAPI inference service loading the Production-stage model, GitHub Actions CI that trains on push, and monitoring scripts wired into the repo workflow.

**Why it matters at Franklin Templeton:** Models that inform portfolio construction need lineage (which data, which code, which metrics), gated promotion, and drift visibility, not just a one-off `main.py` run on a laptop.

**How it works (simply):**
- Training logs Sharpe, drawdown, and information ratio to MLflow experiment `portfolio_ml`.
- `mlops/promote.py` moves the latest registered `PortfolioXGBoost` version to Production.
- `deployment/api.py` serves `/predict` from the MLflow Production model.
- `data.dvc` tracks dataset artifacts; `.dvcignore` keeps cache clean.
- `monitoring/prepare_monitoring.py` and `drift.py` compare baseline vs current feature CSVs with Evidently.
- `.github/workflows/ml_pipeline.yml` installs deps, runs `python main.py`, and verifies `output/` on CI.

**Stack:** MLflow, DVC, Evidently, FastAPI, GitHub Actions, Docker (portfolio-ml Dockerfile), pytest

**Key thing learned:** Registry stages (`Production`) turn a research script into something an API can load predictably; without that, "deploy model" is just copying a pickle path from memory.

**Run:** Train with `cd portfolio-ml && python main.py` → promote via `python mlops/promote.py` → serve with uvicorn on `deployment/api.py` → drift via monitoring scripts (see CI workflow)

---

### 12. Long-Context Document Comparator

**What we built:** A tool that compares two long regulatory documents (e.g., year-over-year 10-Ks), section by section. It finds additions, removals, and edits, classifies materiality (risk escalations, new disclosures, numeric changes), optionally runs Groq semantic analysis on ambiguous diffs, and renders a React analyst dashboard from JSON output.

**Why it matters at Franklin Templeton:** Compliance and research teams need "what changed and why it matters," not a raw redline. Section-aware diff plus materiality scoring prioritizes analyst time on escalations and new risk language.

**How it works (simply):**
- Parser splits each filing into 10-K-style sections.
- Differ aligns sections between years and extracts text-level changes.
- Semantic analyser (Groq) interprets changes where wording shift hides meaning shift.
- Rule-based classifier scores materiality 1-5 using keywords (e.g., "material weakness", "going concern") and change types.
- `run_all.py` writes `output/final_report.json`.
- `index.html` dashboard shows severity counts, risk delta, and expandable finding cards.

**Stack:** Python, PyMuPDF, Groq, JSON reporting, React (single-file dashboard), text diff

**Key thing learned:** Section-by-section processing is both a token cost control and a UX choice: analysts think in MD&A, risk factors, and notes, not one giant blob of text.

**Run:** `cd doc_comparator` → `pip install -r requirements.txt` → `python run_all.py` → open `index.html` and load `output/final_report.json`

---

## Skills Built

**ML/NLP fundamentals**
- Financial sentiment with FinBERT; hedging and forward-looking language rules
- Feature engineering for cross-sectional equity signals (momentum, volatility, fundamentals)
- Event-study style market reaction analysis around corporate events

**RAG architecture and evaluation**
- Hybrid dense + BM25 retrieval, cross-encoder/Cohere reranking, citation-first prompting
- Qdrant and ChromaDB ingestion patterns; chunk size/overlap tradeoffs
- Multi-evaluator hallucination harness (NLI, RAGAS, heuristics) with disagreement analysis
- RAGAS evaluation hook in `financial-rag/eval/`

**Multi-agent orchestration**
- LangGraph state graphs, conditional routing, critic revision loops
- Typed shared state (Pydantic) and SQLite audit trails for traceability

**Classical ML and backtesting**
- Expanding-window walk-forward validation; long-short decile portfolios
- Sharpe, max drawdown, information ratio; transaction cost awareness
- Regime labeling with KMeans as a simple non-stationarity hook

**MLOps and deployment**
- Docker multi-service compose, nginx reverse proxy, health checks
- MLflow tracking/registry, DVC data versioning, Evidently drift monitoring
- CI pipeline training on push; FastAPI model serving

**Finance domain literacy**
- 10-K and earnings transcript workflows; ESG taxonomy and greenwashing signals
- SQL analytics over market tables with auditable generated queries
- Finance Q&A fine-tuning; news/filing alert taxonomy for portfolio relevance

## Setup & Running Projects

Each project folder is independent. General pattern:

1. `cd <project-folder>`
2. Create a virtual environment: `python -m venv .venv && source .venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt` (or `pip install -e .` for `sql-finance-copilot`)
4. Copy or create `.env` with required API keys (most LLM projects need `GROQ_API_KEY`)
5. Follow the project README for ingest/train steps, then start the app (CLI, FastAPI, Streamlit, or Docker)

| Folder | Quick start |
|--------|-------------|
| `financial-rag` | `python ingest.py` then API + Streamlit (or `docker compose up`) |
| `hallucination-eval-harness` | ingest → `compare_evaluators.py` → Streamlit |
| `investment-research` | `python main.py` |
| `LangGraph` | `python main.py` (tutorial) |
| `earnings_detector` | `python scripts/run_pipeline.py` |
| `esg-analyzer` | `python main.py` / Streamlit dashboard |
| `sql-finance-copilot` | `python seed_db.py` then `scripts/run_pipeline.py` |
| `finance_lora_project` | Colab notebook |
| `news-alert-system` | `docker compose up --build` |
| `portfolio-ml` | `python main.py` |
| `doc_comparator` | `python run_all.py` then open `index.html` |

**API keys commonly used:** Groq (LLM), Cohere (rerank in financial-rag), NewsAPI/Alpaca (news-alert-system). Never commit `.env` files; they are gitignored.

**Repository:** https://github.com/dhanushkonduru/franklin-templeton-prep

## Author

**Dhanush Konduru**, AI/ML engineer candidate building production-shaped finance AI systems (RAG, agents, classical ML, MLOps). This portfolio was assembled to demonstrate readiness for Franklin Templeton's Digital & AI/ML Engineer role and the AI Platform team's standards for grounded, auditable, deployable machine learning.
