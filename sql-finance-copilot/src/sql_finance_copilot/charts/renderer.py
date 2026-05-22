from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


ChartType = Literal["line", "bar", "scatter", "histogram"]


@dataclass(slots=True)
class RenderedChart:
    chart_type: ChartType
    figure: go.Figure
    x: str | None = None
    y: str | None = None
    title: str | None = None


class ChartRenderer:
    """Reusable Plotly chart renderer for pandas DataFrames.

    The renderer accepts a DataFrame and an explicit chart type, then infers
    reasonable default axes when they are not provided. This keeps Streamlit
    integration thin while preserving a single rendering policy.
    """

    def render(
        self,
        frame: pd.DataFrame,
        chart_type: ChartType,
        x: str | None = None,
        y: str | None = None,
        title: str | None = None,
    ) -> RenderedChart:
        if frame.empty:
            raise ValueError("Cannot render a chart from an empty DataFrame")

        prepared = self._prepare_frame(frame)
        x_axis, y_axis = self._resolve_axes(prepared, chart_type, x, y)

        if chart_type == "line":
            figure = px.line(prepared, x=x_axis, y=y_axis, title=title)
        elif chart_type == "bar":
            figure = px.bar(prepared, x=x_axis, y=y_axis, title=title)
        elif chart_type == "scatter":
            figure = px.scatter(prepared, x=x_axis, y=y_axis, title=title)
        elif chart_type == "histogram":
            figure = px.histogram(prepared, x=x_axis, title=title)
        else:
            raise ValueError(f"Unsupported chart type: {chart_type}")

        figure.update_layout(margin=dict(l=20, r=20, t=50, b=20), title=title)
        return RenderedChart(chart_type=chart_type, figure=figure, x=x_axis, y=y_axis, title=title)

    def render_streamlit(
        self,
        st_module: Any,
        frame: pd.DataFrame,
        chart_type: ChartType,
        x: str | None = None,
        y: str | None = None,
        title: str | None = None,
        use_container_width: bool = True,
    ) -> RenderedChart:
        rendered = self.render(frame, chart_type=chart_type, x=x, y=y, title=title)
        st_module.plotly_chart(rendered.figure, use_container_width=use_container_width)
        return rendered

    def _prepare_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        prepared = frame.copy()
        for column in prepared.columns:
            if prepared[column].dtype == "object" and ("date" in column.lower() or "time" in column.lower()):
                parsed = pd.to_datetime(prepared[column], errors="coerce")
                if parsed.notna().sum() >= max(1, len(prepared) // 2):
                    prepared[column] = parsed
        return prepared

    def _resolve_axes(
        self,
        frame: pd.DataFrame,
        chart_type: ChartType,
        x: str | None,
        y: str | None,
    ) -> tuple[str | None, str | None]:
        if chart_type == "histogram":
            return self._resolve_histogram_axis(frame, x)

        numeric_columns = list(frame.select_dtypes(include="number").columns)
        datetime_columns = list(frame.select_dtypes(include=["datetime", "datetimetz"]).columns)
        categorical_columns = [
            column
            for column in frame.columns
            if column not in numeric_columns and column not in datetime_columns
        ]

        resolved_x = x
        resolved_y = y

        if chart_type == "line":
            if resolved_x is None:
                resolved_x = datetime_columns[0] if datetime_columns else (categorical_columns[0] if categorical_columns else frame.columns[0])
            if resolved_y is None:
                resolved_y = numeric_columns[0] if numeric_columns else self._first_non_x_column(frame, resolved_x)
        elif chart_type == "bar":
            if resolved_x is None:
                resolved_x = categorical_columns[0] if categorical_columns else (datetime_columns[0] if datetime_columns else frame.columns[0])
            if resolved_y is None:
                resolved_y = numeric_columns[0] if numeric_columns else self._first_non_x_column(frame, resolved_x)
        elif chart_type == "scatter":
            if resolved_x is None:
                resolved_x = numeric_columns[0] if numeric_columns else frame.columns[0]
            if resolved_y is None:
                resolved_y = numeric_columns[1] if len(numeric_columns) >= 2 else self._first_non_x_column(frame, resolved_x)

        if resolved_x is None:
            raise ValueError(f"Could not infer x-axis for {chart_type}")
        if chart_type != "histogram" and resolved_y is None:
            raise ValueError(f"Could not infer y-axis for {chart_type}")
        return resolved_x, resolved_y

    def _resolve_histogram_axis(self, frame: pd.DataFrame, x: str | None) -> tuple[str, None]:
        if x is not None:
            return x, None
        numeric_columns = list(frame.select_dtypes(include="number").columns)
        if numeric_columns:
            return numeric_columns[0], None
        return frame.columns[0], None

    def _first_non_x_column(self, frame: pd.DataFrame, x_axis: str) -> str:
        for column in frame.columns:
            if column != x_axis:
                return column
        raise ValueError("Could not infer a secondary axis")


__all__ = ["ChartRenderer", "ChartType", "RenderedChart"]
