"""Phase 2 chart recommendations with clear, type-safe chart rendering."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from eda_tool.profiling import get_categorical_field_guidance, has_meaningful_numeric_variation


@dataclass(frozen=True)
class ChartRecommendation:
    """The recommended chart plus a plain-language reason for it."""

    chart_type: str
    reason: str


def recommend_chart_type(dataframe: pd.DataFrame, selected_columns: list[str]) -> ChartRecommendation:
    """Recommend the clearest chart for one field or a pair of fields."""
    if not selected_columns:
        return ChartRecommendation("Bar chart", "Choose a field to receive a chart suggestion.")
    if len(selected_columns) == 1:
        column = selected_columns[0]
        if pd.api.types.is_numeric_dtype(dataframe[column]):
            if not has_meaningful_numeric_variation(dataframe, column):
                return ChartRecommendation("No chart", f"{column} is constant or has too little data variation, so a distribution chart would not add useful information.")
            return ChartRecommendation("Histogram", f"{column} contains numbers, so a distribution chart shows its usual range.")
        guidance = get_categorical_field_guidance(dataframe, column)
        if guidance["identifier_like"]:
            return ChartRecommendation("No chart", f"{column} looks like an ID or URL field, so charting individual values would be unreadable.")
        if guidance["high_cardinality"]:
            return ChartRecommendation("Top 10 bar chart", f"{column} has {guidance['unique_count']:,} unique values. Showing only the 10 most common values keeps the chart readable.")
        return ChartRecommendation("Bar chart", f"{column} contains groups or labels, so a count chart is easiest to read.")
    first, second = selected_columns[:2]
    first_is_number = pd.api.types.is_numeric_dtype(dataframe[first])
    second_is_number = pd.api.types.is_numeric_dtype(dataframe[second])
    if first_is_number and second_is_number:
        if not has_meaningful_numeric_variation(dataframe, first) or not has_meaningful_numeric_variation(dataframe, second):
            constant = first if not has_meaningful_numeric_variation(dataframe, first) else second
            return ChartRecommendation("No chart", f"Scatter plot not meaningful because {constant} has no variation or too few usable values.")
        return ChartRecommendation("Scatter plot", "Both selected fields contain numbers, so each point can show how they move together.")
    if first_is_number != second_is_number:
        numeric_column = first if first_is_number else second
        category_column = second if first_is_number else first
        category_guidance = get_categorical_field_guidance(dataframe, category_column)
        if not has_meaningful_numeric_variation(dataframe, numeric_column):
            return ChartRecommendation("No chart", f"{numeric_column} has no variation, so comparing it across groups would not be meaningful.")
        if category_guidance["identifier_like"]:
            return ChartRecommendation("No chart", f"{category_column} looks like an ID or URL field, not a meaningful grouping field.")
        if category_guidance["high_cardinality"]:
            return ChartRecommendation("No chart", f"{category_column} has too many groups ({category_guidance['unique_count']:,}) for a readable comparison chart.")
        return ChartRecommendation("Box plot", "One field is a number and one is a group, so this chart compares typical ranges across groups.")
    first_guidance = get_categorical_field_guidance(dataframe, first)
    second_guidance = get_categorical_field_guidance(dataframe, second)
    if first_guidance["identifier_like"] or second_guidance["identifier_like"]:
        return ChartRecommendation("No chart", "One selected field looks like an ID or URL, so category combinations would not be meaningful.")
    if first_guidance["high_cardinality"] or second_guidance["high_cardinality"]:
        return ChartRecommendation("Top 10 count heatmap", "At least one field has many categories. The chart will limit each field to its 10 most common values.")
    return ChartRecommendation("Count heatmap", "Both selected fields are groups, so this chart counts which combinations occur most often.")


def get_compatible_chart_types(dataframe: pd.DataFrame, selected_columns: list[str]) -> list[str]:
    """Only offer overrides that can correctly represent the selected fields."""
    if not selected_columns:
        return []
    if len(selected_columns) == 1:
        recommendation = recommend_chart_type(dataframe, selected_columns)
        return [recommendation.chart_type]
    first, second = selected_columns[:2]
    first_is_number = pd.api.types.is_numeric_dtype(dataframe[first])
    second_is_number = pd.api.types.is_numeric_dtype(dataframe[second])
    if first_is_number and second_is_number:
        return [recommend_chart_type(dataframe, selected_columns).chart_type]
    if first_is_number != second_is_number:
        if recommend_chart_type(dataframe, selected_columns).chart_type == "No chart":
            return ["No chart"]
        return ["Box plot", "Violin plot"]
    recommendation = recommend_chart_type(dataframe, selected_columns)
    if recommendation.chart_type == "No chart":
        return ["No chart"]
    if recommendation.chart_type == "Top 10 count heatmap":
        return ["Top 10 count heatmap", "Stacked bar chart"]
    return ["Count heatmap", "Stacked bar chart"]


def generate_recommended_chart(dataframe: pd.DataFrame, columns: list[str], chart_type: str, show_trend: bool = True) -> go.Figure | None:
    """Render a readable chart using the selected one or two fields only."""
    if not columns:
        return None
    if chart_type == "No chart":
        return None
    first = columns[0]
    second = columns[1] if len(columns) > 1 else None
    if chart_type == "Histogram":
        return _histogram(dataframe, first)
    if chart_type in {"Bar chart", "Top 10 bar chart"}:
        return _bar_chart(dataframe, first)
    if chart_type == "Scatter plot" and second:
        return _scatter_chart(dataframe, first, second, show_trend)
    if chart_type in {"Box plot", "Violin plot"} and second:
        return _group_comparison(dataframe, first, second, chart_type)
    if chart_type in {"Count heatmap", "Top 10 count heatmap", "Stacked bar chart"} and second:
        return _category_comparison(dataframe, first, second, chart_type)
    return _histogram(dataframe, first) if pd.api.types.is_numeric_dtype(dataframe[first]) else _bar_chart(dataframe, first)


def _histogram(dataframe: pd.DataFrame, column: str) -> go.Figure:
    values = dataframe[column].replace([np.inf, -np.inf], np.nan).dropna()
    figure = px.histogram(values, x=column, nbins=min(30, max(8, int(np.sqrt(max(len(values), 1))))), color_discrete_sequence=["#3B82F6"])
    figure.update_traces(hovertemplate=f"<b>{column}</b>: %{{x}}<br><b>Records</b>: %{{y:,}}<extra></extra>")
    if not values.empty:
        figure.add_vline(x=values.median(), line_dash="dash", line_color="#FBBF24", annotation_text="Middle value", annotation_position="top right")
    figure.update_xaxes(title=column)
    figure.update_yaxes(title="Number of records")
    return _style_chart(figure, f"Distribution of {column}")


def _bar_chart(dataframe: pd.DataFrame, column: str) -> go.Figure:
    counts = dataframe[column].fillna("(missing)").astype(str).value_counts().head(12).sort_values()
    labels = [_shorten(value) for value in counts.index]
    figure = go.Figure(go.Bar(x=counts.values, y=labels, orientation="h", marker_color="#3B82F6", customdata=counts.index,
        text=[f"{value:,}" for value in counts.values], textposition="auto",
        hovertemplate="<b>%{customdata}</b><br><b>Records</b>: %{x:,}<extra></extra>"))
    figure.update_xaxes(title="Number of records")
    figure.update_yaxes(title=None)
    return _style_chart(figure, f"Most common values in {column}", height=max(340, 150 + len(counts) * 36))


def _scatter_chart(dataframe: pd.DataFrame, x_column: str, y_column: str, show_trend: bool) -> go.Figure:
    clean = dataframe[[x_column, y_column]].replace([np.inf, -np.inf], np.nan).dropna()
    figure = go.Figure(go.Scatter(x=clean[x_column], y=clean[y_column], mode="markers", name="Records",
        marker={"color": "#60A5FA", "size": 8, "opacity": 0.72, "line": {"color": "#0F172A", "width": 0.4}},
        hovertemplate=f"<b>{x_column}</b>: %{{x}}<br><b>{y_column}</b>: %{{y}}<extra></extra>"))
    if show_trend and len(clean) >= 2 and clean[x_column].nunique() > 1:
        slope, intercept = np.polyfit(clean[x_column], clean[y_column], 1)
        x_values = np.linspace(clean[x_column].min(), clean[x_column].max(), 100)
        relationship = clean[x_column].corr(clean[y_column])
        figure.add_scatter(x=x_values, y=slope * x_values + intercept, mode="lines", name="Overall trend", line={"color": "#FBBF24", "width": 3}, hovertemplate="Overall trend<extra></extra>")
        figure.add_annotation(x=0.01, y=0.99, xref="paper", yref="paper", showarrow=False, xanchor="left", yanchor="top", text=f"Relationship strength: {relationship:.2f}", bgcolor="#0F172A", bordercolor="#475569", font={"color": "#F8FAFC"})
    figure.update_xaxes(title=x_column)
    figure.update_yaxes(title=y_column)
    return _style_chart(figure, f"Does {y_column} change with {x_column}?")


def _group_comparison(dataframe: pd.DataFrame, first: str, second: str, chart_type: str) -> go.Figure:
    number, group = (first, second) if pd.api.types.is_numeric_dtype(dataframe[first]) else (second, first)
    frame = dataframe[[number, group]].dropna()
    groups = frame[group].value_counts().head(12).index
    frame = frame[frame[group].isin(groups)]
    if chart_type == "Violin plot":
        figure = px.violin(frame, x=group, y=number, color=group, box=True, points=False, color_discrete_sequence=px.colors.qualitative.Safe)
    else:
        figure = px.box(frame, x=group, y=number, color=group, points="outliers", color_discrete_sequence=px.colors.qualitative.Safe)
    figure.update_layout(showlegend=False)
    figure.update_xaxes(title=group, tickangle=-25)
    figure.update_yaxes(title=number)
    return _style_chart(figure, f"How {number} differs across {group}")


def _category_comparison(dataframe: pd.DataFrame, first: str, second: str, chart_type: str) -> go.Figure:
    frame = dataframe[[first, second]].fillna("(missing)").astype(str)
    first_values = frame[first].value_counts().head(10).index
    second_values = frame[second].value_counts().head(10).index
    counts = pd.crosstab(frame.loc[frame[first].isin(first_values), first], frame.loc[frame[second].isin(second_values), second])
    if chart_type == "Stacked bar chart":
        figure = px.bar(counts, barmode="stack", labels={"value": "Number of records", "index": first}, color_discrete_sequence=px.colors.qualitative.Safe)
        figure.update_yaxes(title="Number of records")
        return _style_chart(figure, f"How {second} is distributed within {first}")
    figure = px.imshow(counts, text_auto=True, aspect="auto", color_continuous_scale="Blues", labels={"x": second, "y": first, "color": "Records"})
    figure.update_xaxes(tickangle=-25)
    return _style_chart(figure, f"Which {first} and {second} combinations are most common?", height=max(380, 150 + len(counts.index) * 38))


def _shorten(value: object, maximum_length: int = 28) -> str:
    text = str(value)
    return text if len(text) <= maximum_length else f"{text[:maximum_length - 1]}..."


def _style_chart(figure: go.Figure, title: str, height: int = 400) -> go.Figure:
    """Apply plain labels and high-contrast hover cards to smart charts."""
    figure.update_layout(template="plotly_dark", title={"text": title, "x": 0}, height=height, paper_bgcolor="#111C33", plot_bgcolor="#111C33", font={"color": "#E2E8F0"}, margin={"l": 18, "r": 18, "t": 60, "b": 55}, hoverlabel={"bgcolor": "#0F172A", "bordercolor": "#60A5FA", "font": {"color": "#F8FAFC", "size": 13}}, legend={"orientation": "h", "y": -0.2})
    figure.update_xaxes(automargin=True, showgrid=False, color="#CBD5E1")
    figure.update_yaxes(automargin=True, gridcolor="#263752", color="#CBD5E1")
    return figure
