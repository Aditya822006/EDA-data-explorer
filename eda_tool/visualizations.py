"""Clean, high-contrast visualizations for the data exploration report."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from scipy.stats import gaussian_kde


CHART_TEMPLATE = "plotly_dark"
ACCENT_COLOR = "#3B82F6"
PALETTE = ["#3B82F6", "#14B8A6", "#F59E0B", "#A78BFA", "#FB7185", "#94A3B8"]


def _short_label(value: object, maximum_length: int = 30) -> str:
    """Prevent long category labels from colliding on compact charts."""
    label = str(value)
    return label if len(label) <= maximum_length else f"{label[:maximum_length - 1].rstrip()}..."


def _wrapped_label(value: object, line_length: int = 14) -> str:
    """Break long heatmap headings over multiple lines without losing words."""
    words = str(value).replace("_", " ").split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > line_length:
            lines.append(current)
            current = word
        else:
            current = candidate
    return "<br>".join(lines + [current])


def _apply_chart_style(figure: go.Figure, title: str, height: int = 390) -> go.Figure:
    """Use one readable dark theme and explicit tooltip contrast everywhere."""
    figure.update_layout(
        template=CHART_TEMPLATE,
        title={"text": title, "x": 0, "xanchor": "left", "font": {"size": 18, "color": "#F8FAFC"}},
        height=height,
        margin={"l": 18, "r": 18, "t": 66, "b": 40},
        paper_bgcolor="#111C33",
        plot_bgcolor="#111C33",
        font={"color": "#E2E8F0"},
        hoverlabel={"bgcolor": "#0F172A", "bordercolor": "#60A5FA", "font": {"color": "#F8FAFC", "size": 13}},
        legend={"orientation": "h", "y": -0.2, "font": {"color": "#CBD5E1"}},
    )
    figure.update_xaxes(showgrid=False, zeroline=False, color="#CBD5E1", automargin=True)
    figure.update_yaxes(gridcolor="#263752", zeroline=False, color="#CBD5E1", automargin=True)
    return figure


def generate_missing_values_chart(dataframe: pd.DataFrame) -> go.Figure:
    missing = dataframe.isna().sum().sort_values(ascending=False)
    values = missing[missing > 0]
    percentages = values.div(len(dataframe)).mul(100)
    labels = [_short_label(column) for column in values.index]
    figure = go.Figure(go.Bar(
        x=values.values, y=labels, orientation="h", marker_color="#F87171",
        customdata=np.column_stack([values.index.astype(str), percentages.round(2)]),
        text=[f"{value:,}" for value in values.values], textposition="auto", textfont={"color": "#F8FAFC"},
        hovertemplate="<b>%{customdata[0]}</b><br>Missing entries: %{x:,}<br>Share: %{customdata[1]:.2f}%<extra></extra>",
    ))
    figure.update_yaxes(autorange="reversed", title=None)
    figure.update_xaxes(title="Missing entries")
    return _apply_chart_style(figure, "Where data is incomplete", max(320, min(680, 170 + len(labels) * 42)))


def generate_numeric_distribution_chart(dataframe: pd.DataFrame, column: str) -> go.Figure:
    values = dataframe[column].replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        figure = go.Figure()
        return _apply_chart_style(figure, f"No values available for {column}")
    bin_count = min(35, max(8, int(np.sqrt(len(values)))))
    figure = go.Figure(go.Histogram(
        x=values, nbinsx=bin_count, name="Records", marker={"color": ACCENT_COLOR, "line": {"color": "#93C5FD", "width": 0.5}},
        hovertemplate="<b>Value range</b>: %{x}<br><b>Records</b>: %{y:,}<extra></extra>",
    ))
    if len(values) >= 3 and values.nunique() > 1:
        sample = values.sample(n=min(5_000, len(values)), random_state=42)
        x_values = np.linspace(sample.min(), sample.max(), 200)
        try:
            bin_width = (sample.max() - sample.min()) / bin_count
            if bin_width > 0:
                curve = gaussian_kde(sample)(x_values) * len(values) * bin_width
                figure.add_scatter(x=x_values, y=curve, mode="lines", name="Overall pattern", line={"color": "#FBBF24", "width": 3},
                                   hovertemplate="<b>Value</b>: %{x:.2f}<br><b>Expected records</b>: %{y:.1f}<extra></extra>")
        except (np.linalg.LinAlgError, ValueError):
            pass
    mean_value, median_value = values.mean(), values.median()
    if pd.notna(mean_value) and pd.notna(median_value):
        figure.add_vline(x=mean_value, line_color="#FB923C", line_width=2, line_dash="dash")
        figure.add_vline(x=median_value, line_color="#4ADE80", line_width=2, line_dash="dot")
        figure.add_annotation(x=0.99, y=0.98, xref="paper", yref="paper", xanchor="right", yanchor="top", showarrow=False,
                              text=f"Average: {mean_value:,.2f}<br>Middle value: {median_value:,.2f}", align="right",
                              bgcolor="#0F172A", bordercolor="#334155", font={"color": "#E2E8F0", "size": 12})
    figure.update_xaxes(title=column)
    figure.update_yaxes(title="Number of records")
    return _apply_chart_style(figure, f"How {column} is distributed")


def generate_numeric_box_plot(dataframe: pd.DataFrame, column: str) -> go.Figure:
    values = dataframe[column].replace([np.inf, -np.inf], np.nan).dropna()
    figure = go.Figure(go.Box(
        x=values, orientation="h", name=column, boxmean=True, boxpoints="outliers",
        marker={"color": "#FBBF24", "size": 7}, line={"color": "#14B8A6", "width": 2}, fillcolor="rgba(20, 184, 166, 0.28)",
        hovertemplate="<b>Value</b>: %{x}<extra></extra>",
    ))
    figure.update_xaxes(title=column)
    figure.update_yaxes(showticklabels=False)
    return _apply_chart_style(figure, f"Typical range and unusual values: {column}", 270)


def generate_categorical_bar_chart(dataframe: pd.DataFrame, column: str, top_n: int = 15) -> go.Figure:
    counts = dataframe[column].fillna("(missing)").astype(str).value_counts().head(top_n)
    labels = [_short_label(value) for value in counts.index]
    figure = go.Figure(go.Bar(
        x=counts.values, y=labels, orientation="h", marker_color=ACCENT_COLOR, customdata=counts.index.astype(str),
        text=[f"{value:,}" for value in counts.values], textposition="auto", textfont={"color": "#F8FAFC"},
        hovertemplate="<b>%{customdata}</b><br><b>Records</b>: %{x:,}<extra></extra>",
    ))
    figure.update_yaxes(autorange="reversed", title=None)
    figure.update_xaxes(title="Number of records")
    return _apply_chart_style(figure, f"Most common values in {column}", max(340, min(720, 160 + len(labels) * 38)))


def generate_scatter_plot(dataframe: pd.DataFrame, x_column: str, y_column: str, color_column: str | None = None) -> go.Figure:
    settings = {"data_frame": dataframe, "x": x_column, "y": y_column, "color_discrete_sequence": PALETTE}
    if color_column:
        settings["color"] = color_column
    figure = px.scatter(**settings, opacity=0.7)
    figure.update_traces(marker={"size": 8, "line": {"width": 0.4, "color": "#0F172A"}},
                         hovertemplate=f"<b>{x_column}</b>: %{{x}}<br><b>{y_column}</b>: %{{y}}<extra></extra>")
    clean = dataframe[[x_column, y_column]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) >= 2 and clean[x_column].nunique() > 1:
        try:
            slope, intercept = np.polyfit(clean[x_column], clean[y_column], 1)
            x_values = np.linspace(clean[x_column].min(), clean[x_column].max(), 100)
            strength = clean[x_column].corr(clean[y_column])
            figure.add_scatter(x=x_values, y=slope * x_values + intercept, mode="lines", name="Overall trend", line={"color": "#FBBF24", "width": 3}, hovertemplate="Overall trend<extra></extra>")
            figure.add_annotation(x=0.01, y=0.99, xref="paper", yref="paper", showarrow=False, xanchor="left", yanchor="top",
                                  text=f"Relationship strength: {strength:.2f}", bgcolor="#0F172A", bordercolor="#334155", font={"color": "#E2E8F0"})
        except (np.linalg.LinAlgError, ValueError, TypeError):
            pass
    return _apply_chart_style(figure, f"How {x_column} and {y_column} relate")


def generate_correlation_heatmap(correlation: pd.DataFrame, title: str) -> go.Figure:
    """Show a legible triangular relationship map; the table retains all fields."""
    if correlation.empty:
        figure = go.Figure()
        return _apply_chart_style(figure, title)
    constant_fields = set(correlation.attrs.get("constant_fields", []))
    display = correlation.iloc[:14, :14]
    full_labels = display.columns.astype(str).tolist()
    labels = [_wrapped_label(label) for label in full_labels]
    matrix = display.to_numpy(dtype=float, copy=True)
    matrix[np.triu_indices_from(matrix, k=0)] = np.nan
    text = np.full(matrix.shape, "", dtype=object)
    hover_text = np.full(matrix.shape, "", dtype=object)
    for row_index, row in enumerate(full_labels):
        for column_index, column in enumerate(full_labels):
            if row_index <= column_index:
                continue
            if row in constant_fields or column in constant_fields:
                text[row_index, column_index] = "N/A"
                hover_text[row_index, column_index] = f"<b>{row}</b> and <b>{column}</b><br>Relationship: N/A (one field has no variation)"
            elif not np.isnan(matrix[row_index, column_index]):
                value = matrix[row_index, column_index]
                if abs(value) >= 0.30:
                    text[row_index, column_index] = f"{value:.2f}"
                hover_text[row_index, column_index] = f"<b>{row}</b> and <b>{column}</b><br>Relationship: {value:.2f}"
    figure = go.Figure(go.Heatmap(
        z=matrix, x=labels, y=labels, zmin=-1, zmax=1, xgap=2, ygap=2,
        colorscale=[(0, "#DC2626"), (0.5, "#172033"), (1, "#2563EB")],
        colorbar={"title": "Strength", "tickvals": [-1, 0, 1], "ticktext": ["Opposite", "None", "Together"]},
        text=text, texttemplate="%{text}", textfont={"size": 11, "color": "#F8FAFC"}, customdata=hover_text,
        hovertemplate="%{customdata}<extra></extra>",
    ))
    figure.update_yaxes(autorange="reversed")
    figure.update_xaxes(tickangle=-35)
    visible_title = title if len(correlation.columns) <= 14 else f"{title} (first 14 fields shown)"
    if constant_fields:
        visible_title += " - N/A marks fields with no variation"
    return _apply_chart_style(figure, visible_title, max(430, 130 + len(labels) * 42))


def generate_grouped_distribution_chart(dataframe: pd.DataFrame, numeric_column: str, category_column: str) -> go.Figure:
    frame = dataframe[[numeric_column, category_column]].dropna()
    top_groups = frame[category_column].value_counts().head(12).index
    frame = frame[frame[category_column].isin(top_groups)]
    figure = px.box(frame, x=category_column, y=numeric_column, color=category_column, color_discrete_sequence=PALETTE, points="outliers")
    figure.update_layout(showlegend=False)
    figure.update_xaxes(tickangle=-25, tickfont={"size": 11})
    return _apply_chart_style(figure, f"How {numeric_column} varies across {category_column}")


def generate_pair_plot_figure(dataframe: pd.DataFrame, columns: list[str]):
    """Create a compact, high-contrast static relationship view for up to three fields."""
    sample = dataframe[columns].dropna().sample(n=min(500, len(dataframe[columns].dropna())), random_state=42)
    sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#111C33", "figure.facecolor": "#111C33", "text.color": "#E2E8F0", "axes.labelcolor": "#E2E8F0", "xtick.color": "#CBD5E1", "ytick.color": "#CBD5E1", "grid.color": "#263752"})
    grid = sns.pairplot(sample, corner=True, diag_kind="hist", height=2.45, plot_kws={"alpha": 0.6, "s": 22, "color": "#60A5FA", "edgecolor": "none"})
    grid.fig.suptitle("Selected field relationships", y=1.02, fontsize=14, fontweight="bold", color="#F8FAFC")
    grid.fig.tight_layout()
    return grid.fig
