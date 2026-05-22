# Multi-Agent Investment Research Assistant

This project is a small multi-agent research system for investment ideas. You give it a stock ticker, and the agents work together to build a simple investment thesis.

## What it does

The pipeline runs in this order:

1. The data gatherer pulls basic stock data and a few news headlines.
2. The retrieval agent gets transcript-like context from the retrieval pipeline.
3. The fundamental analyst checks valuation using simple ratios.
4. The sentiment analyst reads the headlines and transcript context to judge tone.
5. The risk analyst looks for major risks.
6. The report writer combines everything into a final report.
7. The critic agent reviews the report and can send it back for revision.

Each step updates a shared state object and writes an audit entry so you can see what happened during the run.

## Main ideas

- LangGraph is used to control the flow between agents.
- Pydantic is used for the shared research state.
- SQLite is used for the audit log.
- The helper tools fetch prices, news, transcript text, and LLM-generated analysis.

## Project structure

- `main.py` starts a sample run for one ticker.
- `orchestration/graph.py` defines the LangGraph workflow.
- `orchestration/routing.py` decides whether the critic sends the report back for another pass.
- `state/research_state.py` defines the typed state that moves through the graph.
- `agents/` contains the specialist research steps.
- `tools/` contains the data source helpers.
- `retrieval/` contains the transcript chunking, embedding, search, reranking, and compression pieces.
- `persistence/audit_store.py` stores a simple audit trail in SQLite.

## How it works in simple words

The system keeps one shared research record. Every agent reads the current record, adds its own findings, and passes the updated record to the next agent. Because the flow is a graph, it is easier to control than a loose chat between agents. It is also easier to resume, debug, and trace.

The critic agent makes the loop a little smarter. If the report is weak, it can send the workflow back to the report writer instead of ending immediately.

## Tech stack

- Python
- LangGraph
- Pydantic
- Groq LLM API
- SQLite
- yfinance
- feedparser
- langchain retrieval utilities

## Notes

- The news and transcript helpers in this version are lightweight demo implementations, so the project is easy to run and explain.
- The audit log is intentionally simple so the workflow is easy to inspect.

## Run it

Install the requirements, activate the virtual environment, then run:

```bash
python main.py
```

By default, the demo run uses `NVDA` as the ticker.