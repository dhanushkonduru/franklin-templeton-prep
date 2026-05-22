from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from sql_finance_copilot.core.models import ChartSuggestion


@dataclass(slots=True)
class ChartResult:
    suggestion: ChartSuggestion
    figure: go.Figure | None


class ChartBuilder:
    def suggest(self, rows: list[dict[str, Any]], title: str | None = None) -> ChartResult:
        if not rows:
            return ChartResult(suggestion=ChartSuggestion(chart_type=None, title=title), figure=None)

        frame = pd.DataFrame(rows)
        for column in frame.columns:
            if frame[column].dtype == "object" and ("date" in column.lower() or "time" in column.lower()):
                parsed = pd.to_datetime(frame[column], errors="coerce")
                if parsed.notna().sum() >= max(1, len(frame) // 2):
                    frame[column] = parsed

        numeric_columns = list(frame.select_dtypes(include="number").columns)
        datetime_columns = list(frame.select_dtypes(include=["datetime", "datetimetz"]).columns)
        if not datetime_columns:
            datetime_columns = [column for column in frame.columns if "date" in column.lower() or "time" in column.lower()]

        if len(datetime_columns) >= 1 and len(numeric_columns) >= 1:
            x_axis = datetime_columns[0]
            y_axis = numeric_columns[0]
            figure = px.line(frame, x=x_axis, y=y_axis, title=title)
            return ChartResult(
                suggestion=ChartSuggestion(chart_type="line", x=x_axis, y=y_axis, title=title),
                figure=figure,
            )

        categorical_columns = [column for column in frame.columns if column not in numeric_columns and column not in datetime_columns]
        if categorical_columns and numeric_columns:
            x_axis = categorical_columns[0]
            y_axis = numeric_columns[0]
            figure = px.bar(frame, x=x_axis, y=y_axis, title=title)
            return ChartResult(
                suggestion=ChartSuggestion(chart_type="bar", x=x_axis, y=y_axis, title=title),
                figure=figure,
            )

        if len(numeric_columns) >= 2:
            x_axis, y_axis = numeric_columns[:2]
            figure = px.scatter(frame, x=x_axis, y=y_axis, title=title)
            return ChartResult(
                suggestion=ChartSuggestion(chart_type="scatter", x=x_axis, y=y_axis, title=title),
                figure=figure,
            )

        return ChartResult(suggestion=ChartSuggestion(chart_type=None, title=title), figure=None)
