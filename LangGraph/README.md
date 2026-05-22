# LangGraph Stock Research Demo

This repository is a small LangGraph example that builds a three-step workflow for a stock ticker.

## What it does

1. `data_agent` fetches stock data from Yahoo Finance.
2. `sentiment_agent` classifies the stock as `Positive` or `Neutral` based on the price.
3. `report_agent` assembles a short investment report.

Run the app with:

```bash
python main.py
```

## What we learned about LangGraph

- A LangGraph app is built around a state schema and a graph of nodes.
- Each node handles one focused step in the workflow.
- Edges define the execution order, and `compile()` turns the graph into a runnable app.
- `invoke()` runs the graph with an initial state and returns the final state after all nodes finish.

## Project notes

- The state model lives in `state.py`.
- The graph wiring lives in `graph.py`.
- The node logic lives in `research_agents.py`.
- The market data helper lives in `tools.py`.
