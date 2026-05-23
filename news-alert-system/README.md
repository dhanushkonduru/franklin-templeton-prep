# Real-Time News + Filing Alert System

Production-oriented async pipeline:

**News Sources → Redis Stream → Embedding Dedupe → Event Classification → Portfolio Match → Alert Delivery → PostgreSQL**

## Architecture

```mermaid
flowchart LR
    subgraph sources [News Sources]
        N[NewsAPI]
        S[SEC EDGAR RSS]
        A[Alpaca News]
    end

    subgraph ingest [Ingestion Worker]
        IL[AsyncNewsIngestionLayer]
        P[RedisStreamProducer]
    end

    subgraph redis [Redis]
        ST[(news_stream)]
        DLQ[(news_stream:dlq)]
        DF[(notification_delivery_failures)]
    end

    subgraph process [Processing Worker]
        C[RedisStreamConsumer]
        PL[NewsPipeline]
    end

    subgraph services [Services]
        E[EmbeddingService]
        D[DeduplicationService]
        CL[EventClassifier]
        M[UnifiedPortfolioMatcher]
        N[NotificationService]
        R[DeliveryReplayConsumer]
    end

    subgraph pg [PostgreSQL]
        DB[(events, alerts, metrics)]
    end

    N --> S
    S --> A
    A --> IL --> P --> ST
    ST --> C --> PL
    PL --> E
    PL --> D
    PL --> CL
    PL --> M
    PL --> DB
    M --> N
    N --> DB
    N -.failures.-> DF
    DF --> R
    R --> N
```

## Run locally

1. Copy `.env.example` to `.env` and configure API keys (`.env` is gitignored; never commit secrets).
2. Start the stack: `docker compose up --build`
3. API docs: `http://localhost:8000/docs`
4. Health: `GET /health/live`, `GET /health/ready`
5. Metrics: `GET /metrics`

## Services

| Service | Command | Role |
|---------|---------|------|
| `api` | `uvicorn app.main:app` | REST API, health, metrics, portfolio admin |
| `worker` | `python -m app.worker` | Source polling, stream consumption, delivery replay |
| `postgres` | — | Durable events, alerts, latency metrics |
| `redis` | — | Streams, dedup index, embedding cache |

## Tests

```bash
python -m pytest tests/ -q
```

Unit tests cover classification, deduplication, delivery, matching, repositories, and streams. Integration coverage exercises ingest → Redis → pipeline → PostgreSQL.

Schema is created at startup via SQLAlchemy `create_all` (see `app/db.py`).
