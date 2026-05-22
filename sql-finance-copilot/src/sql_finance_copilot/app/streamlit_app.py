from __future__ import annotations

import sys
from pathlib import Path
import os
import logging

from dotenv import load_dotenv

load_dotenv()

# Ensure `src/` is on sys.path so Streamlit can import package modules
# regardless of how Streamlit changes the working directory.
_ROOT = Path(__file__).resolve().parents[3]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import streamlit as st

from sql_finance_copilot.app.frontend_ui import (
    append_history,
    execute_query,
    init_session_state,
    render_chart,
    render_error,
    render_execution_summary,
    render_header,
    render_history_sidebar,
    render_query_controls,
    render_repair_details,
    render_result_table,
    render_sql_panel,
)
from sql_finance_copilot.config import AppSettings
from sql_finance_copilot.core.orchestrator import SqlCopilot


st.set_page_config(page_title="SQL Finance Copilot", layout="wide")

# Debug logging to both Streamlit and console
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("streamlit_app")

# Ensure env is loaded and DATABASE_URL is absolute for sqlite local demos
def _normalize_database_url():
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        return None
    # Only handle sqlite local path normalization here; leave other URLs alone
    if db_url.startswith("sqlite://"):
        # Extract path after sqlite:/// or sqlite:////
        parts = db_url.split("sqlite://")
        path_part = parts[-1]
        # Remove leading slashes to allow relative input like sqlite:///finance.db
        rel = path_part.lstrip("/")
        if not rel:
            return db_url
        abs_path = (_ROOT / rel).resolve()
        abs_url = f"sqlite:///{abs_path}"
        os.environ["DATABASE_URL"] = abs_url
        return abs_url
    return db_url

resolved_db = _normalize_database_url()
cwd = Path.cwd()
logger.info("Streamlit start: cwd=%s, resolved DATABASE_URL=%s", cwd, resolved_db)

settings = AppSettings.from_env()

# Surface debug info in the UI sidebar for transparency
with st.sidebar.expander("Runtime Info", expanded=False):
    st.write({
        "cwd": str(cwd),
        "resolved_database_url": os.getenv("DATABASE_URL"),
        "groq_api_key_present": bool(settings.groq_api_key),
    })


@st.cache_resource(show_spinner=True)
def get_copilot() -> SqlCopilot:
    return SqlCopilot.build(settings)

init_session_state()
render_header()
render_history_sidebar()

question, show_chart, chart_type, run_clicked = render_query_controls()

if run_clicked:
    with st.spinner("Generating and validating SQL..."):
        artifact, elapsed_ms, run_error = execute_query(get_copilot(), question, show_chart)

    if run_error is not None:
        render_error(run_error, elapsed_ms)
    elif artifact is not None:
        append_history(artifact, question, elapsed_ms)
        render_execution_summary(artifact, elapsed_ms)
        render_sql_panel(artifact)

        left, right = st.columns([1, 1])
        with left:
            frame = render_result_table(artifact)
        with right:
            if show_chart:
                render_chart(frame, chart_type=chart_type, title=question)

        render_repair_details(artifact)
