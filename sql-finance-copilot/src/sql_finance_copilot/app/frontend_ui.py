from __future__ import annotations

from datetime import datetime
from time import perf_counter
from typing import Any

import pandas as pd
import streamlit as st

from sql_finance_copilot.app.frontend_models import QueryHistoryItem
from sql_finance_copilot.charts.builder import ChartBuilder
from sql_finance_copilot.charts.renderer import ChartRenderer, ChartType
from sql_finance_copilot.core.models import QueryArtifact
from sql_finance_copilot.core.orchestrator import SqlCopilot


_HISTORY_KEY = "query_history"


def init_session_state() -> None:
    if _HISTORY_KEY not in st.session_state:
        st.session_state[_HISTORY_KEY] = []


def render_header() -> None:
    st.title("Financial SQL Copilot")
    st.caption("Analyst workflow: ask in natural language, inspect SQL, review results, and audit repairs.")


def render_query_controls() -> tuple[str, bool, ChartType, bool]:
    question = st.text_area(
        "Ask a finance question",
        placeholder="Example: Compare quarterly revenue growth by sector over the last 2 years.",
        height=120,
    )
    cols = st.columns([1, 1, 1, 2])
    with cols[0]:
        show_chart = st.checkbox("Chart", value=True)
    with cols[1]:
        chart_type = st.selectbox("Chart Type", ["line", "bar", "scatter", "histogram"], index=0, disabled=not show_chart)
    with cols[2]:
        run_clicked = st.button("Run Query", type="primary", disabled=not question.strip())
    with cols[3]:
        st.caption("Tip: review generated and validated SQL before sharing outputs.")
    return question.strip(), show_chart, chart_type, run_clicked


def execute_query(copilot: SqlCopilot, question: str, show_chart: bool) -> tuple[QueryArtifact | None, float, Exception | None]:
    started = perf_counter()
    try:
        artifact = copilot.answer(question, chart=show_chart)
        elapsed_ms = (perf_counter() - started) * 1000.0
        return artifact, elapsed_ms, None
    except Exception as exc:  # noqa: BLE001 - frontend must surface backend errors without crashing the page
        elapsed_ms = (perf_counter() - started) * 1000.0
        return None, elapsed_ms, exc


def append_history(artifact: QueryArtifact, question: str, elapsed_ms: float) -> None:
    history: list[QueryHistoryItem] = st.session_state[_HISTORY_KEY]
    history.insert(
        0,
        QueryHistoryItem(
            asked_at=datetime.utcnow(),
            question=question,
            elapsed_ms=elapsed_ms,
            row_count=artifact.row_count,
            repaired=artifact.repaired,
            repair_attempts=artifact.repair_attempts,
            error=artifact.error,
            sql=artifact.sql,
            validated_sql=artifact.validated_sql,
            rows_preview=artifact.rows[:5],
        ),
    )
    del history[50:]


def render_execution_summary(artifact: QueryArtifact, elapsed_ms: float) -> None:
    metrics = st.columns(5)
    metrics[0].metric("Execution Time", f"{elapsed_ms:.1f} ms")
    metrics[1].metric("Rows", str(artifact.row_count))
    metrics[2].metric("Repaired", "Yes" if artifact.repaired else "No")
    metrics[3].metric("Repair Attempts", str(artifact.repair_attempts))
    metrics[4].metric("Truncated", "Yes" if artifact.truncated else "No")


def render_sql_panel(artifact: QueryArtifact) -> None:
    st.subheader("SQL Audit")
    col_a, col_b = st.columns(2)
    with col_a:
        st.caption("Generated SQL")
        st.code(artifact.sql, language="sql")
    with col_b:
        st.caption("Validated SQL")
        st.code(artifact.validated_sql, language="sql")


def render_repair_details(artifact: QueryArtifact) -> None:
    st.subheader("Repair Visibility")
    if not artifact.repair_log:
        st.info("No repair attempts were recorded.")
        return
    for item in artifact.repair_log:
        with st.expander(f"Attempt {item.get('attempt', 0)}"):
            st.write({"error": item.get("error")})
            st.caption("Input SQL")
            st.code(item.get("sql", ""), language="sql")
            st.caption("Repaired SQL")
            st.code(item.get("repaired_sql", ""), language="sql")


def render_result_table(artifact: QueryArtifact) -> pd.DataFrame:
    st.subheader("Result Table")
    if not artifact.rows:
        st.info("Query returned no rows.")
        return pd.DataFrame()
    frame = pd.DataFrame(artifact.rows)
    st.dataframe(frame, use_container_width=True)
    return frame


def render_chart(frame: pd.DataFrame, chart_type: ChartType, title: str) -> None:
    st.subheader("Chart")
    if frame.empty:
        st.info("No rows to chart.")
        return
    renderer = ChartRenderer()
    try:
        rendered = renderer.render_streamlit(st, frame, chart_type=chart_type, title=title)
        st.caption(f"Rendered {rendered.chart_type} chart on x={rendered.x!r} and y={rendered.y!r}.")
    except ValueError:
        inferred = ChartBuilder().suggest(frame.to_dict("records"), title=title)
        if inferred.figure is None:
            st.info("Unable to infer a chart for this result set.")
            return
        st.plotly_chart(inferred.figure, use_container_width=True)


def render_error(error: Exception | str, elapsed_ms: float) -> None:
    # Surface full tracebacks for easier debugging in the UI
    if isinstance(error, Exception):
        st.exception(error)
    else:
        st.error(error)
    st.caption(f"Execution failed after {elapsed_ms:.1f} ms")


def render_history_sidebar() -> None:
    st.sidebar.header("Query History")
    history: list[QueryHistoryItem] = st.session_state[_HISTORY_KEY]
    if not history:
        st.sidebar.caption("No queries yet.")
        return

    for idx, item in enumerate(history[:15], start=1):
        status = "error" if item.error else "ok"
        with st.sidebar.expander(f"{idx}. {item.question[:40]} ({status})"):
            st.write(
                {
                    "asked_at": item.asked_at.isoformat(timespec="seconds"),
                    "elapsed_ms": round(item.elapsed_ms, 2),
                    "row_count": item.row_count,
                    "repaired": item.repaired,
                    "repair_attempts": item.repair_attempts,
                    "error": item.error,
                }
            )
            st.caption("Validated SQL")
            st.code(item.validated_sql, language="sql")
            if item.rows_preview:
                st.caption("Preview")
                st.dataframe(pd.DataFrame(item.rows_preview), use_container_width=True)
