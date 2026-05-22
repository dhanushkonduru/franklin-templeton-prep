#!/usr/bin/env bash
# Lightweight launcher for the Streamlit frontend (development)
# Usage: ./scripts/run_streamlit.sh

set -euo pipefail

# Ensure PYTHONPATH includes src so imports resolve
export PYTHONPATH="src"

# Optional: load .env if present
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

streamlit run src/sql_finance_copilot/app/streamlit_app.py
