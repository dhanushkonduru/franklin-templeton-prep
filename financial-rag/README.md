# Financial RAG (Simple Guide)

This project answers one main question:

"Can we ask normal English questions about company 10-K filings and get accurate answers with proof from the documents?"

Short answer: yes. We built a Retrieval-Augmented Generation (RAG) system for financial filings.

## What we did (in simple words)

1. Collected company filings (Apple, Microsoft, Tesla) into `data/raw/`.
2. Read and cleaned those files (HTML/PDF) into text + tables.
3. Split long text into small pieces called chunks.
4. Converted each chunk into vectors (numbers) using a local embedding model.
5. Stored those vectors in ChromaDB.
6. Built a search pipeline to find the best chunks for any question.
7. Sent those chunks to an LLM to generate a final answer with citations and confidence.

## What we used

- LangChain: pipeline, retrievers, prompts, LCEL chain composition.
- BGE embedding model (`BAAI/bge-large-en-v1.5`): turns text into vectors locally.
- ChromaDB: stores vectors and runs dense similarity search.
- BM25: keyword search (good for exact financial terms).
- Cohere Rerank: rescoring top results for better relevance.
- Groq LLM (`llama-3.3-70b-versatile`): writes final answer.
- FastAPI: backend API.
- Streamlit: frontend app.

## How it works

### A. Ingestion (one-time, or when adding new files)

`ingest.py` does this:

1. Download filings (or use existing files in `data/raw/`).
2. Parse files into LangChain `Document` objects with metadata.
3. Chunk text using:
   - `CHUNK_SIZE=800`
   - `CHUNK_OVERLAP=100`
4. Embed chunks with BGE.
5. Save embeddings in `data/chroma_db/`.

### B. Question answering (runtime)

For each user question:

1. Dense search in ChromaDB (semantic match).
2. BM25 search (keyword match).
3. Combine both scores (hybrid retrieval).
4. Cohere reranks top results.
5. Top chunks + question go to LLM through LCEL chain.
6. System returns:
   - answer text,
   - source citations (file + page),
   - confidence (`HIGH`, `MEDIUM`, `LOW`).

## Where each part lives

- `ingestion/downloader.py`: fetches filings.
- `ingestion/parser.py`: parses PDF/HTML into documents.
- `ingestion/chunker.py`: chunk logic.
- `retrieval/vectorstore.py`: embeddings + ChromaDB load/build.
- `retrieval/retriever.py`: BM25 + dense + rerank pipeline.
- `retrieval/chain.py`: LCEL chain + final response format.
- `api/main.py`: API endpoints.
- `ui/app.py`: Streamlit app.

## Run the project

### 1. Install dependencies

```bash
cd financial-rag
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

Edit `.env` with your keys/settings.

### 3. Build the vector database

```bash
python ingest.py
```

### 4. Start API

```bash
uvicorn api.main:app --reload --port 8000
```

### 5. Start UI (new terminal)

```bash
streamlit run ui/app.py
```

## API endpoints

- `GET /health`: service health.
- `GET /stats`: vector DB stats.
- `POST /query`: full answer response.
- `POST /query/stream`: streamed response.

## One-line summary

This repo is a financial-document question-answering system that searches filing chunks (vector + keyword), reranks them, and generates cited answers with confidence.
