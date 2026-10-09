"""Phase 2 stakeholder-ready PDF report generation."""
from __future__ import annotations

import base64
from io import BytesIO
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def generate_pdf_report(dataframe: pd.DataFrame, health_score: Any, rule_results: list[Any], iqr_summary: pd.DataFrame, correlation: pd.DataFrame) -> bytes:
    """Create a styled executive PDF report using WeasyPrint or a clean 3-page Matplotlib fallback."""
    try:
        from jinja2 import Template
        from weasyprint import HTML

        from eda_tool.analysis import generate_numeric_statistical_summary

        stats_summary = generate_numeric_statistical_summary(dataframe)
        stats_records = stats_summary.to_dict("records") if not stats_summary.empty else []

        context = {
            "rows": f"{len(dataframe):,}",
            "columns": len(dataframe.columns),
            "memory_mb": f"{dataframe.memory_usage(deep=True).sum() / 1024**2:.2f}",
            "health": health_score,
            "rules": rule_results,
            "profile": _profile_rows(dataframe),
            "stats": stats_records,
            "outliers": iqr_summary.to_dict("records") if not iqr_summary.empty else [],
            "health_chart": _health_chart(health_score),
            "correlation_chart": _correlation_chart(correlation),
            "distribution_chart": _distribution_chart(dataframe),
            "summary": health_score.summary,
        }
        html = Template(_REPORT_TEMPLATE).render(**context)
        return HTML(string=html).write_pdf()
    except Exception:
        return _generate_fallback_pdf(dataframe, health_score, rule_results, iqr_summary, correlation)


def _generate_fallback_pdf(dataframe: pd.DataFrame, health_score: Any, rule_results: list[Any], iqr_summary: pd.DataFrame, correlation: pd.DataFrame) -> bytes:
    """Generate a clean, professional 3-page PDF report via Matplotlib backend."""
    from matplotlib.backends.backend_pdf import PdfPages
    from eda_tool.analysis import generate_numeric_statistical_summary

    buffer = BytesIO()
    with PdfPages(buffer) as report:
        # Page 1: Executive Overview & Data Health
        figure = plt.figure(figsize=(8.27, 11.69), facecolor="#FFFFFF")
        figure.text(0.08, 0.94, "Datalens - An Intelligent Exploratory Data Analysis Platform", fontsize=16, fontweight="bold", color="#0F172A")
        figure.text(0.08, 0.915, f"Total Records: {len(dataframe):,}   |   Fields: {len(dataframe.columns):,}   |   Memory Footprint: {dataframe.memory_usage(deep=True).sum() / 1024**2:.2f} MB", fontsize=9.5, color="#334155")
        
        # Health Box
        figure.text(0.08, 0.85, f"Data Health Score: {health_score.overall}/100", fontsize=22, fontweight="bold", color="#1D4ED8")
        figure.text(0.08, 0.82, health_score.summary, fontsize=10.5, color="#0F172A", wrap=True)
        
        # Health Breakdown Chart
        axis = figure.add_axes([0.10, 0.58, 0.80, 0.20])
        _apply_print_axis_style(axis)
        labels = ["Complete data", "No duplicates", "Typical values", "Rule checks"]
        scores = [health_score.missing_data, health_score.duplicate_records, health_score.unusual_values, health_score.rule_compliance]
        bars = axis.barh(labels, scores, color="#2563EB", height=0.55)
        axis.set_xlim(0, 100); axis.set_xlabel("Component Score (out of 100)", fontsize=9); axis.grid(axis="x", alpha=0.3, linestyle="--")
        for bar in bars:
            width = bar.get_width()
            axis.text(width + 1.5, bar.get_y() + bar.get_height()/2, f"{width:.0f}", va="center", fontsize=9, fontweight="bold", color="#1E3A8A")
        axis.set_title("Data Health Component Breakdown", fontsize=11, fontweight="bold", pad=8)
        
        # Quality Checks Section
        figure.text(0.08, 0.50, "Data Quality Checks Summary", fontsize=14, fontweight="bold", color="#0F172A")
        rule_lines = []
        for result in rule_results[:8]:
            status = "PASS" if result.passed else "REVIEW"
            rule_lines.append(f"[{status}]  {result.rule_name}\n        -> {result.message}")
        checks_text = "\n".join(rule_lines) if rule_lines else "No specific quality checks were enabled."
        figure.text(0.08, 0.46, checks_text, fontsize=8.5, va="top", color="#0F172A")

        figure.text(0.08, 0.04, "Page 1 of 3  |  Datalens Executive Report", fontsize=8, color="#94A3B8")
        report.savefig(figure, bbox_inches="tight"); plt.close(figure)

        # Page 2: Field Profiling & Key Numerical Statistics
        figure = plt.figure(figsize=(8.27, 11.69), facecolor="#FFFFFF")
        figure.text(0.08, 0.94, "Field Overview & Key Statistics", fontsize=18, fontweight="bold", color="#0F172A")
        
        # Field profile summary text table
        figure.text(0.08, 0.90, "Field Summary (First 15 Fields)", fontsize=12, fontweight="bold", color="#0F172A")
        profile_headers = f"{'Field Name':<28} | {'Type':<12} | {'Filled':<8} | {'Blank':<8} | {'Unique':<8}"
        figure.text(0.08, 0.88, profile_headers, fontsize=8.5, fontweight="bold", family="monospace", color="#0F172A")
        figure.text(0.08, 0.873, "-" * 82, fontsize=8.5, family="monospace", color="#64748B")
        
        y_pos = 0.855
        for row in _profile_rows(dataframe)[:14]:
            line = f"{row['name'][:26]:<28} | {row['type']:<12} | {row['filled']:<8} | {row['blank']:<8} | {row['unique']:<8}"
            figure.text(0.08, y_pos, line, fontsize=8, family="monospace", color="#0F172A")
            y_pos -= 0.019

        # Numerical statistics summary table
        stats_df = generate_numeric_statistical_summary(dataframe)
        if not stats_df.empty:
            y_pos -= 0.03
            figure.text(0.08, y_pos, "Key Numerical Measure Summary", fontsize=12, fontweight="bold", color="#0F172A")
            y_pos -= 0.02
            stats_headers = f"{'Field':<22} | {'Average':<10} | {'Middle (Med)':<12} | {'Spread':<10} | {'Distribution':<18}"
            figure.text(0.08, y_pos, stats_headers, fontsize=8.5, fontweight="bold", family="monospace", color="#0F172A")
            y_pos -= 0.007
            figure.text(0.08, y_pos, "-" * 82, fontsize=8.5, family="monospace", color="#64748B")
            y_pos -= 0.018
            for _, row in stats_df.head(10).iterrows():
                line = f"{str(row['Field'])[:20]:<22} | {float(row['Average']):<10.2f} | {float(row['Middle Value']):<12.2f} | {float(row['Spread']):<10.2f} | {str(row['Value Distribution']):<18}"
                figure.text(0.08, y_pos, line, fontsize=8, family="monospace", color="#0F172A")
                y_pos -= 0.019

        figure.text(0.08, 0.04, "Page 2 of 3  |  Datalens Executive Report", fontsize=8, color="#94A3B8")
        report.savefig(figure, bbox_inches="tight"); plt.close(figure)

        # Page 3: Visual Relationships & Unusual Values
        if not correlation.empty or not iqr_summary.empty:
            figure, axes = plt.subplots(2, 1, figsize=(8.27, 11.69), facecolor="#FFFFFF", gridspec_kw={"height_ratios": [3, 2]})
            _apply_print_axis_style(axes[0])
            _apply_print_axis_style(axes[1])
            if not correlation.empty:
                sns.heatmap(correlation.iloc[:12, :12], cmap="RdBu_r", vmin=-1, vmax=1, center=0, ax=axes[0], cbar_kws={"label": "Relationship Strength"})
                axes[0].set_title("Field Relationships (Correlation Heatmap)", fontsize=12, fontweight="bold", pad=10)
            else:
                axes[0].axis("off")
            if not iqr_summary.empty:
                top = iqr_summary.sort_values("outlier %", ascending=False).head(10)
                axes[1].barh(top["column"], top["outlier %"], color="#F59E0B")
                axes[1].invert_yaxis()
                axes[1].set_title("Unusual Values Share by Field (%)", fontsize=12, fontweight="bold", pad=10)
                axes[1].set_xlabel("Share of records (%)")
            else:
                axes[1].axis("off")
            figure.tight_layout()
            figure.text(0.08, 0.02, "Page 3 of 3  |  Datalens Executive Report", fontsize=8, color="#94A3B8")
            report.savefig(figure); plt.close(figure)

    return buffer.getvalue()


def _apply_print_axis_style(axis) -> None:
    """Reset axes after Streamlit's dark charts modify global Matplotlib defaults."""
    axis.set_facecolor("#FFFFFF")
    axis.tick_params(axis="both", colors="#0F172A", labelsize=8.5)
    axis.xaxis.label.set_color("#0F172A")
    axis.yaxis.label.set_color("#0F172A")
    axis.title.set_color("#0F172A")
    for spine in axis.spines.values():
        spine.set_color("#94A3B8")


def _profile_rows(dataframe: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for column in dataframe.columns[:25]:
        series = dataframe[column]
        dtype_str = str(series.dtype)
        if "int" in dtype_str or "float" in dtype_str:
            user_type = "Number"
        elif "date" in dtype_str:
            user_type = "Date / Time"
        elif "bool" in dtype_str:
            user_type = "Yes / No"
        else:
            user_type = "Text"
        rows.append({
            "name": str(column),
            "type": user_type,
            "filled": f"{series.notna().sum():,}",
            "blank": f"{series.isna().sum():,}",
            "unique": f"{series.nunique(dropna=True):,}",
            "sample": str(series.dropna().iloc[0])[:40] if not series.dropna().empty else "—"
        })
    return rows


def _health_chart(health: Any) -> str:
    labels = ["Complete data", "No duplicates", "Typical values", "Rule checks"]
    values = [health.missing_data, health.duplicate_records, health.unusual_values, health.rule_compliance]
    figure, axis = plt.subplots(figsize=(6, 2.6), facecolor="#FFFFFF")
    _apply_print_axis_style(axis)
    bars = axis.barh(labels, values, color="#2563EB")
    axis.set_xlim(0, 100); axis.set_xlabel("Score out of 100", fontsize=9); axis.grid(axis="x", alpha=0.25, linestyle="--")
    for bar in bars:
        w = bar.get_width()
        axis.text(w + 1.5, bar.get_y() + bar.get_height()/2, f"{w:.0f}", va="center", fontsize=9, fontweight="bold", color="#1E3A8A")
    axis.set_title("Data Health Components", fontsize=11, fontweight="bold", color="#0F172A")
    return _figure_as_data_uri(figure)


def _correlation_chart(correlation: pd.DataFrame) -> str | None:
    if correlation.empty:
        return None
    figure, axis = plt.subplots(figsize=(7, 4.5), facecolor="#FFFFFF")
    _apply_print_axis_style(axis)
    sns.heatmap(correlation.iloc[:12, :12], cmap="RdBu_r", vmin=-1, vmax=1, center=0, ax=axis)
    axis.set_title("Field Relationships (Correlation Heatmap)", fontsize=11, fontweight="bold", color="#0F172A")
    axis.tick_params(axis="both", colors="#0F172A", labelsize=8)
    figure.tight_layout()
    return _figure_as_data_uri(figure)


def _distribution_chart(dataframe: pd.DataFrame) -> str | None:
    numeric = dataframe.select_dtypes(include="number")
    if numeric.empty:
        return None
    column = numeric.columns[0]
    figure, axis = plt.subplots(figsize=(7, 3.2), facecolor="#FFFFFF")
    _apply_print_axis_style(axis)
    axis.hist(numeric[column].dropna(), bins=25, color="#2563EB", edgecolor="white")
    axis.set_title(f"Distribution Pattern of {column}", fontsize=11, fontweight="bold", color="#0F172A")
    axis.set_xlabel(column); axis.set_ylabel("Records")
    figure.tight_layout()
    return _figure_as_data_uri(figure)


def _figure_as_data_uri(figure) -> str:
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(figure)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


_REPORT_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
@page { size: A4; margin: 16mm; }
body { font-family: Arial, sans-serif; background: #FFFFFF; color: #0F172A; font-size: 10pt; line-height: 1.5; }
.header { border-bottom: 2px solid #2563EB; padding-bottom: 12px; margin-bottom: 20px; }
h1 { color: #0F172A; font-size: 24pt; margin: 0 0 4px 0; letter-spacing: -0.5px; }
.subtitle { color: #334155; font-size: 10pt; margin: 0; }
.cards { display: flex; gap: 14px; margin: 16px 0 24px 0; }
.card { background: #F0F6FF; border: 1px solid #BFDBFE; border-radius: 8px; padding: 12px 16px; flex: 1; }
.card-label { font-size: 8pt; color: #0F172A; font-weight: bold; text-transform: uppercase; letter-spacing: 0.5px; }
.card-value { font-size: 18pt; font-weight: bold; color: #1D4ED8; margin-top: 4px; }
.health-section { background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 18px; margin-bottom: 24px; page-break-inside: avoid; }
.health-score { font-size: 32pt; font-weight: bold; color: #2563EB; }
h2 { color: #0F172A; font-size: 14pt; border-bottom: 2px solid #2563EB; padding-bottom: 6px; margin-top: 24px; margin-bottom: 12px; }
table { width: 100%; border-collapse: collapse; margin-top: 8px; margin-bottom: 16px; font-size: 9pt; }
th { background: #1E3A8A; color: white; padding: 8px 10px; text-align: left; font-weight: 600; }
td { padding: 7px 10px; color: #0F172A; border-bottom: 1px solid #CBD5E1; }
tr:nth-child(even) { background: #F8FAFC; }
.badge-pass { background: #DCFCE7; color: #15803D; padding: 3px 8px; border-radius: 4px; font-weight: bold; font-size: 8pt; }
.badge-review { background: #FEE2E2; color: #B91C1C; padding: 3px 8px; border-radius: 4px; font-weight: bold; font-size: 8pt; }
.chart-img { width: 100%; max-width: 650px; display: block; margin: 12px auto; }
.page-break { page-break-before: always; }
</style>
</head>
<body>
<div class="header">
  <h1>Datalens - An Intelligent Exploratory Data Analysis Platform</h1>
  <p class="subtitle">Comprehensive dataset exploration, data health analysis, and statistical summary</p>
</div>

<div class="cards">
  <div class="card"><div class="card-label">Total Records</div><div class="card-value">{{ rows }}</div></div>
  <div class="card"><div class="card-label">Total Fields</div><div class="card-value">{{ columns }}</div></div>
  <div class="card"><div class="card-label">File Size in Memory</div><div class="card-value">{{ memory_mb }} MB</div></div>
</div>

<div class="health-section">
  <div class="card-label">Data Health Score</div>
  <div class="health-score">{{ health.overall }}/100</div>
  <p><strong>Status:</strong> {{ summary }}</p>
  <img src="{{ health_chart }}" class="chart-img">
</div>

<h2>Data Quality Checks</h2>
<table>
  <tr><th>Check Description</th><th>Status</th><th>Details</th></tr>
  {% for rule in rules %}
  <tr>
    <td>{{ rule.rule_name }}</td>
    <td><span class="{{ 'badge-pass' if rule.passed else 'badge-review' }}">{{ 'PASS' if rule.passed else 'REVIEW NEEDED' }}</span></td>
    <td>{{ rule.message }}</td>
  </tr>
  {% endfor %}
</table>

<div class="page-break"></div>

<h2>Field Overview & Profile</h2>
<table>
  <tr><th>Field Name</th><th>Type</th><th>Filled</th><th>Blank</th><th>Unique Values</th><th>Sample Value</th></tr>
  {% for field in profile %}
  <tr>
    <td><strong>{{ field.name }}</strong></td>
    <td>{{ field.type }}</td>
    <td>{{ field.filled }}</td>
    <td>{{ field.blank }}</td>
    <td>{{ field.unique }}</td>
    <td><code>{{ field.sample }}</code></td>
  </tr>
  {% endfor %}
</table>

{% if stats %}
<h2>Key Numerical Field Statistics</h2>
<table>
  <tr><th>Field</th><th>Records</th><th>Average</th><th>Middle Value</th><th>Spread</th><th>Minimum</th><th>Maximum</th><th>Distribution</th></tr>
  {% for item in stats %}
  <tr>
    <td><strong>{{ item.Field }}</strong></td>
    <td>{{ item.Records }}</td>
    <td>{{ item.Average }}</td>
    <td>{{ item['Middle Value'] }}</td>
    <td>{{ item.Spread }}</td>
    <td>{{ item.Minimum }}</td>
    <td>{{ item.Maximum }}</td>
    <td>{{ item['Value Distribution'] }}</td>
  </tr>
  {% endfor %}
</table>
{% endif %}

{% if outliers %}
<h2>Unusual Values Summary</h2>
<table>
  <tr><th>Field Name</th><th>Lower Expected Guide</th><th>Upper Expected Guide</th><th>Unusual Values Count</th><th>Share of Records (%)</th></tr>
  {% for row in outliers %}
  <tr>
    <td><strong>{{ row.column }}</strong></td>
    <td>{{ row['lower bound'] }}</td>
    <td>{{ row['upper bound'] }}</td>
    <td>{{ row.outliers }}</td>
    <td>{{ row['outlier %'] }}%</td>
  </tr>
  {% endfor %}
</table>
{% endif %}

{% if distribution_chart %}
<h2>Example Distribution Pattern</h2>
<img src="{{ distribution_chart }}" class="chart-img">
{% endif %}

{% if correlation_chart %}
<h2>Field Relationships (Correlation Heatmap)</h2>
<img src="{{ correlation_chart }}" class="chart-img">
{% endif %}

</body>
</html>
"""
