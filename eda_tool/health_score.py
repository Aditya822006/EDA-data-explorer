"""Phase 2 composite data-health scoring and its dashboard visualization."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import plotly.graph_objects as go


@dataclass(frozen=True)
class HealthScore:
    """A transparent, weighted health score with component-level detail."""

    overall: int
    missing_data: float
    duplicate_records: float
    unusual_values: float
    rule_compliance: float
    summary: str


def calculate_data_health_score(
    quality_summary: dict[str, float | int],
    iqr_summary: pd.DataFrame,
    rule_results: list[object],
) -> HealthScore:
    """Calculate a 0-100 score from Phase 1 quality results and Phase 2 checks.

    The score is intentionally transparent: missing data carries 35%, duplicate
    records 20%, unusual values 20%, and enabled data-quality checks 25%. A
    component score of 100 means no issue was observed for that component.
    """
    missing_score = max(0.0, 100.0 - float(quality_summary.get("missing_percent", 0)))
    duplicate_score = max(0.0, 100.0 - float(quality_summary.get("duplicate_percent", 0)))
    outlier_percent = float(iqr_summary["outlier %"].mean()) if not iqr_summary.empty else 0.0
    outlier_score = max(0.0, 100.0 - outlier_percent)
    enabled_results = [result for result in rule_results if getattr(result, "enabled", True)]
    passed_rules = sum(bool(getattr(result, "passed", False)) for result in enabled_results)
    rules_score = 100.0 if not enabled_results else passed_rules / len(enabled_results) * 100
    overall = round(
        missing_score * 0.35 + duplicate_score * 0.20 + outlier_score * 0.20 + rules_score * 0.25
    )
    return HealthScore(
        overall=int(overall),
        missing_data=round(missing_score, 1),
        duplicate_records=round(duplicate_score, 1),
        unusual_values=round(outlier_score, 1),
        rule_compliance=round(rules_score, 1),
        summary=_health_summary(int(overall)),
    )


def generate_health_breakdown_radar(health_score: HealthScore) -> go.Figure:
    """Create a readable radar chart showing every health-score contribution."""
    labels = ["Complete data", "No duplicates", "Typical values", "Rule checks"]
    values = [health_score.missing_data, health_score.duplicate_records, health_score.unusual_values, health_score.rule_compliance]
    figure = go.Figure(go.Scatterpolar(
        r=values + [values[0]], theta=labels + [labels[0]], fill="toself", name="Health score",
        line={"color": "#60A5FA", "width": 3}, fillcolor="rgba(96, 165, 250, 0.25)",
        hovertemplate="<b>%{theta}</b><br>Score: %{r:.1f}/100<extra></extra>",
    ))
    figure.update_layout(
        template="plotly_dark", height=370, paper_bgcolor="#111C33", plot_bgcolor="#111C33",
        title={"text": "What makes up the health score", "x": 0, "font": {"color": "#F8FAFC", "size": 18}},
        font={"color": "#E2E8F0"}, showlegend=False,
        polar={"bgcolor": "#111C33", "radialaxis": {"visible": True, "range": [0, 100], "gridcolor": "#334155", "tickfont": {"color": "#94A3B8"}}, "angularaxis": {"gridcolor": "#334155"}},
        hoverlabel={"bgcolor": "#0F172A", "bordercolor": "#60A5FA", "font": {"color": "#F8FAFC"}},
    )
    return figure


def _health_summary(score: int) -> str:
    if score >= 90:
        return "Excellent: the dataset is ready for confident analysis."
    if score >= 75:
        return "Good: review the lower-scoring areas before making important decisions."
    if score >= 55:
        return "Needs attention: data issues could affect analysis results."
    return "At risk: address the reported quality issues before relying on this dataset."
