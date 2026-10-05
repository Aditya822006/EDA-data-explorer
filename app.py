"""Run with: ``streamlit run app.py``.

Upload a CSV, Excel, or JSON file to explore it interactively. Uploaded data is
processed in memory and is never written to disk by the application.
"""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import streamlit as st

from eda_tool.analysis import (
    calculate_correlation_matrix,
    detect_outliers_isolation_forest,
    detect_univariate_iqr_outliers,
    generate_numeric_statistical_summary,
    identify_strong_correlations,
)
from eda_tool.loaders import load_uploaded_dataset
from eda_tool.health_score import calculate_data_health_score, generate_health_breakdown_radar
from eda_tool.pdf_report import generate_pdf_report
from eda_tool.profiling import (
    build_column_profile,
    build_quality_summary,
    get_all_categorical_columns,
    get_categorical_field_guidance,
    get_categorical_columns,
    get_numeric_columns,
    has_meaningful_numeric_variation,
    infer_and_correct_data_types,
)
from eda_tool.quality_rules import get_builtin_rule_options, run_data_quality_rules
from eda_tool.recommendations import (
    generate_recommended_chart,
    get_compatible_chart_types,
    recommend_chart_type,
)
from eda_tool.visualizations import (
    generate_categorical_bar_chart,
    generate_correlation_heatmap,
    generate_grouped_distribution_chart,
    generate_missing_values_chart,
    generate_numeric_box_plot,
    generate_numeric_distribution_chart,
    generate_pair_plot_figure,
    generate_scatter_plot,
)


st.set_page_config(page_title="Data Explorer", page_icon=":material/insights:", layout="wide")


def apply_application_style() -> None:
    """Apply a modern, restrained visual theme across the application."""
    st.markdown(
        """
        <style>
        .stApp { background: #0B1120; color: #E5E7EB; }
        .block-container { max-width: 1400px; padding-top: 2.2rem; padding-bottom: 3rem; }
        h1 { color: #F8FAFC; font-weight: 750; letter-spacing: -0.03em; }
        h2, h3, h4 { color: #F1F5F9; letter-spacing: -0.015em; }
        p, label, .stCaption { color: #CBD5E1; }
        [data-testid="stSidebar"] { background: #0F172A; border-right: 1px solid #1E293B; }
        [data-testid="stSidebar"] * { color: #E2E8F0; }
        [data-testid="stMetric"] {
            background: #111C33; border: 1px solid #263752; border-radius: 10px;
            padding: 12px 14px; box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15);
        }
        [data-testid="stMetricLabel"] { color: #94A3B8; font-size: 0.82rem; font-weight: 600; }
        [data-testid="stMetricValue"] { color: #F8FAFC; font-weight: 700; }
        .stTabs [data-baseweb="tab-list"] { gap: 0.3rem; border-bottom: 1px solid #263752; }
        .stTabs [data-baseweb="tab"] { color: #94A3B8; font-weight: 600; padding: 0.65rem 0.85rem; }
        .stTabs [aria-selected="true"] { color: #60A5FA; border-bottom-color: #60A5FA; }
        .report-intro { background: linear-gradient(105deg, #111C33, #16274A); border: 1px solid #28436E;
            border-left: 4px solid #60A5FA; border-radius: 10px; padding: 1rem 1.2rem; margin: 0.2rem 0 1.2rem; }
        .report-intro .eyebrow { color: #93C5FD; font-size: 0.72rem; font-weight: 750; letter-spacing: 0.08em; }
        .report-intro h2 { margin: 0.15rem 0 0.3rem; font-size: 1.25rem; }
        .report-intro p { color: #CBD5E1; margin: 0; font-size: 0.92rem; }
        .control-help { background: #111C33; border: 1px solid #263752; border-radius: 8px; color: #CBD5E1;
            font-size: 0.86rem; line-height: 1.45; padding: 0.55rem 0.7rem; margin: 0.2rem 0 0.5rem; }
        .control-help strong { color: #93C5FD; }
        [data-testid="stFileUploader"], [data-testid="stDataFrame"], [data-testid="stMarkdownContainer"] { color: #E2E8F0; }
        [data-testid="stFileUploaderDropzone"] { background: #111C33; border-color: #3B82F6; }
        .stButton > button, .stDownloadButton > button { background: #1D4ED8; border-radius: 7px; border-color: #3B82F6; color: #FFFFFF; font-weight: 600; }
        [data-testid="stVerticalBlockBorderWrapper"] { background: #111C33; border-color: #263752; }
        .landing-hero { background: linear-gradient(120deg, #102A5C 0%, #1E4E9B 58%, #2563EB 100%);
            border-radius: 16px; box-shadow: 0 12px 30px rgba(30, 78, 155, 0.2); color: white;
            padding: 3rem 3.2rem; margin: 0.5rem 0 1.4rem; }
        .landing-hero .brand { color: #BFDBFE; font-size: 0.8rem; font-weight: 750; letter-spacing: 0.12em; }
        .landing-hero h1 { color: #FFFFFF; font-size: clamp(2rem, 3.8vw, 3.2rem); line-height: 1.1;
            max-width: 740px; margin: 0.4rem 0 0.7rem; }
        .landing-hero p { color: #DBEAFE; font-size: 1.05rem; line-height: 1.6; max-width: 640px; margin: 0; }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def prepare_dataset(file_name: str, file_bytes: bytes) -> tuple[pd.DataFrame, str]:
    """Load, standardize headings, and infer safe types for an uploaded file."""
    uploaded_bytes = SimpleNamespace(name=file_name, getvalue=lambda: file_bytes)
    frame, source = load_uploaded_dataset(uploaded_bytes)
    if frame.empty:
        raise ValueError("The file was read successfully but contains no rows.")
    frame.columns = _make_column_names_unique(frame.columns)
    return infer_and_correct_data_types(frame), source


def _make_column_names_unique(columns: pd.Index) -> list[str]:
    """Ensure column headings are unique and unambiguous."""
    seen: dict[str, int] = {}
    names: list[str] = []
    for index, column in enumerate(columns):
        name = str(column).strip() or f"unnamed_{index}"
        occurrence = seen.get(name, 0)
        names.append(name if occurrence == 0 else f"{name}_{occurrence}")
        seen[name] = occurrence + 1
    return names


def show_tab_intro(title: str, purpose: str, example: str) -> None:
    """Display a concise section banner with clear purpose and example."""
    st.markdown(
        f"<div class='report-intro'><div class='eyebrow'>REPORT SECTION</div><h2>{title}</h2>"
        f"<p>{purpose} <strong>Example:</strong> {example}</p></div>",
        unsafe_allow_html=True,
    )


def show_control_help(choice: str, example: str) -> None:
    """Display brief control guidance."""
    st.markdown(
        f"<div class='control-help'><strong>Select:</strong> {choice}<br>"
        f"<strong>Example:</strong> {example}</div>",
        unsafe_allow_html=True,
    )


def show_dataset_header(dataframe: pd.DataFrame, file_name: str, source: str) -> None:
    """Display key summary metrics for the loaded dataset."""
    quality = build_quality_summary(dataframe)
    st.success(f"Dataset ready: {file_name} ({source})")
    values = [
        ("Data health", f"{st.session_state.get('phase2_health_score', 0)}/100"),
        ("Records", f"{quality['rows']:,}"),
        ("Fields", quality["columns"]),
        ("Memory size", f"{quality['memory_mb']} MB"),
        ("Missing cells", f"{quality['missing_percent']}%"),
        ("Repeated rows", f"{quality['duplicate_rows']:,}"),
        ("Numeric fields", len(get_numeric_columns(dataframe))),
    ]
    for container, (label, value) in zip(st.columns(len(values)), values):
        container.metric(label, value)


def render_landing_header() -> None:
    """Render the application landing screen header."""
    st.markdown(
        """
        <section class="landing-hero">
            <div class="brand">DATA EXPLORER</div>
            <h1>Turn your data into clear business insights.</h1>
            <p>Upload your file to evaluate data quality, analyze statistical patterns, and generate executive PDF reports.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def get_phase2_rule_inputs(dataframe: pd.DataFrame) -> tuple[dict[str, bool], dict | None]:
    """Retrieve enabled built-in and custom quality rules."""
    available = get_builtin_rule_options(dataframe)
    enabled = {
        rule_name: bool(st.session_state.get(f"phase2_builtin_{index}", is_available))
        for index, (rule_name, is_available) in enumerate(available.items())
        if is_available
    }
    if not st.session_state.get("phase2_custom_enabled", False) or dataframe.empty:
        return enabled, None
    column = st.session_state.get("phase2_custom_column", dataframe.columns[0])
    condition = st.session_state.get("phase2_custom_condition", "Must not be blank")
    rule = {"column": column, "condition": condition}
    if condition in {"Must be at least", "Must be between"}:
        rule["minimum"] = st.session_state.get("phase2_custom_minimum", 0.0)
    if condition == "Must be between":
        rule["maximum"] = st.session_state.get("phase2_custom_maximum", 100.0)
    return enabled, rule


def render_health_tab(health_score, quality_summary: dict, iqr_summary: pd.DataFrame, rule_results: list) -> None:
    """Render overall data health score and component breakdown."""
    show_tab_intro("Data health score", "Single summary rating of overall dataset quality and readiness for analysis.", "Scores above 85 indicate high-quality, analysis-ready data.")
    left, right = st.columns([1, 2], gap="large")
    with left:
        st.metric("Overall health score", f"{health_score.overall}/100")
        st.write(health_score.summary)
        st.caption("Scoring weight: Complete data (35%), Unique records (20%), Typical values (20%), Quality rules (25%).")
    with right:
        st.plotly_chart(generate_health_breakdown_radar(health_score), use_container_width=True)
    breakdown = pd.DataFrame({
        "Component": ["Complete data", "Unique records", "Typical values", "Quality rules"],
        "Score (0–100)": [health_score.missing_data, health_score.duplicate_records, health_score.unusual_values, health_score.rule_compliance],
        "Summary details": [
            f"{quality_summary['missing_percent']}% of all cells are blank",
            f"{quality_summary['duplicate_percent']}% of records are exact duplicates",
            f"{iqr_summary['outlier %'].mean() if not iqr_summary.empty else 0:.2f}% average unusual value share",
            f"{sum(result.passed for result in rule_results)} of {len(rule_results)} active quality rules passed" if rule_results else "No quality rules enabled",
        ],
    })
    st.dataframe(breakdown, use_container_width=True, hide_index=True)


def render_quality_checks_tab(dataframe: pd.DataFrame, rule_results: list) -> None:
    """Render toggleable quality check rules and custom rule configuration."""
    show_tab_intro("Quality checks", "Run automated validation rules to verify your data satisfies expected logic.", "Verify that 'Price' is positive or 'ID' fields are unique.")
    options = get_builtin_rule_options(dataframe)
    visible_options = [(name, available) for name, available in options.items() if available]
    if visible_options:
        st.markdown("#### Built-in quality rules")
        st.caption("Select rules relevant to your dataset. Rules auto-detect matching fields.")
        for index, (rule_name, _) in enumerate(visible_options):
            st.checkbox(rule_name, value=True, key=f"phase2_builtin_{list(options).index(rule_name)}")
    else:
        st.info("No built-in rules matched the dataset column names. Configure a custom rule below.")

    st.markdown("#### Custom quality rule")
    show_control_help("a field and rule requirement.", "Select 'Age' and 'Must be at least' with a value of 0.")
    custom_enabled = st.checkbox("Enable custom rule", key="phase2_custom_enabled")
    if custom_enabled:
        column = st.selectbox("Field to validate", dataframe.columns.tolist(), key="phase2_custom_column")
        conditions = ["Must not be blank", "Must be unique"]
        if pd.api.types.is_numeric_dtype(dataframe[column]):
            conditions += ["Must be at least", "Must be between"]
        condition = st.selectbox("Rule requirement", conditions, key="phase2_custom_condition")
        if condition in {"Must be at least", "Must be between"}:
            st.number_input("Minimum value", value=0.0, key="phase2_custom_minimum")
        if condition == "Must be between":
            st.number_input("Maximum value", value=100.0, key="phase2_custom_maximum")

    st.markdown("#### Rule validation results")
    if not rule_results:
        st.info("Enable built-in or custom rules to view validation results.")
        return
    results = pd.DataFrame([{
        "Rule name": result.rule_name,
        "Status": "Pass" if result.passed else "Review needed",
        "Validation note": result.message,
    } for result in rule_results])
    st.dataframe(results, use_container_width=True, hide_index=True)


def render_smart_chart_picker(dataframe: pd.DataFrame) -> None:
    """Guide users through a valid one-field or two-field chart selection."""
    st.markdown("#### Smart chart recommendation")
    show_control_help("whether you want to understand one field or compare two fields.", "Choose ‘Compare two fields’ for Sales and Profit, or ‘Understand one field’ for Region.")
    selection_mode = st.radio("What do you want to do?", ["Understand one field", "Compare two fields"], horizontal=True, key="phase2_chart_mode")
    first_column = st.selectbox("First field", dataframe.columns.tolist(), key="phase2_chart_first")
    selected = [first_column]
    if selection_mode == "Compare two fields":
        remaining_columns = [column for column in dataframe.columns if column != first_column]
        if not remaining_columns:
            st.info("Upload a dataset with at least two fields to compare them.")
            return
        second_column = st.selectbox("Second field", remaining_columns, key="phase2_chart_second")
        selected.append(second_column)
    recommendation = recommend_chart_type(dataframe, selected)
    if recommendation.chart_type == "No chart":
        st.warning(recommendation.reason)
        return
    st.success(f"Recommended chart: {recommendation.chart_type}")
    st.caption(recommendation.reason)
    chart_options = get_compatible_chart_types(dataframe, selected)
    selected_chart = st.selectbox("Chart type", chart_options, index=chart_options.index(recommendation.chart_type), key="phase2_chart_override")
    show_trend = st.checkbox("Show overall trend line", value=True, key="phase2_show_trend") if selected_chart == "Scatter plot" else False
    figure = generate_recommended_chart(dataframe, selected, selected_chart, show_trend)
    if figure is not None:
        st.plotly_chart(figure, use_container_width=True)
    if selected_chart == "Histogram":
        st.caption("Read it left to right: taller bars mean more records fall within that value range. The dashed line marks the middle value.")
    elif selected_chart == "Scatter plot":
        st.caption("Each dot is one record. An upward line means both fields usually rise together; a downward line means one usually falls as the other rises.")
    elif selected_chart in {"Box plot", "Violin plot"}:
        st.caption("Compare the height and position of each group. A higher middle area means that group usually has larger values.")
    else:
        st.caption("Darker or larger areas show combinations that occur more often. Hover over a value to see the exact record count.")


def render_pdf_export(dataframe: pd.DataFrame, health_score, rule_results: list, iqr_summary: pd.DataFrame, correlation: pd.DataFrame) -> None:
    """Render PDF export trigger and download button."""
    st.markdown("#### Executive PDF Report")
    st.caption("Generate a 3-page summary report with data health, quality checks, statistical tables, and relationship charts.")
    if st.button("Generate Executive PDF Report", key="phase2_prepare_pdf"):
        try:
            st.session_state["phase2_pdf_bytes"] = generate_pdf_report(dataframe, health_score, rule_results, iqr_summary, correlation)
        except RuntimeError as error:
            st.info(str(error))
    pdf_bytes = st.session_state.get("phase2_pdf_bytes")
    if pdf_bytes:
        st.download_button("Download Executive PDF Report", pdf_bytes, "data_executive_report.pdf", "application/pdf")


def render_overview_tab(dataframe: pd.DataFrame) -> None:
    show_tab_intro("Dataset overview", "Inspect record rows, field data types, and missing entry counts.", "Confirm key fields like 'Date' or 'Sales' were loaded correctly.")
    st.markdown("#### First 100 records")
    st.dataframe(dataframe.head(100), use_container_width=True, height=320)
    st.markdown("#### Field profile")
    st.dataframe(build_column_profile(dataframe), use_container_width=True, hide_index=True)
    st.download_button(
        "Download prepared data (CSV)",
        dataframe.to_csv(index=False).encode("utf-8"),
        "prepared_data.csv",
        "text/csv",
        help="Downloads dataset with inferred number and date formats.",
    )


def render_quality_tab(dataframe: pd.DataFrame) -> None:
    quality = build_quality_summary(dataframe)
    show_tab_intro("Data completeness & duplicates", "Review missing values and repeated records across your dataset.", "Check if missing values affect key analysis fields.")
    st.info(f"Dataset contains {quality['missing_cells']:,} missing cells ({quality['missing_percent']}%) and {quality['duplicate_rows']:,} repeated records ({quality['duplicate_percent']}%).")
    if dataframe.isna().any().any():
        st.plotly_chart(generate_missing_values_chart(dataframe), use_container_width=True)
    else:
        st.success("All fields are complete; no missing values found.")
    if quality["duplicate_rows"]:
        st.markdown("#### Repeated records")
        st.caption("Identical rows in your dataset. Review before removing duplicates.")
        st.dataframe(dataframe[dataframe.duplicated(keep=False)].head(100), use_container_width=True)


def render_statistics_tab(dataframe: pd.DataFrame) -> None:
    show_tab_intro("Statistical summary", "Review averages, medians, spread, and value distributions for numeric fields.", "Use average and median to gauge typical order sizes.")
    summary = generate_numeric_statistical_summary(dataframe)
    if summary.empty:
        st.info("No numeric fields are available for statistical summary.")
        return
    st.caption("Distribution notes indicate shape (skewness or outliers), not a strict statistical pass/fail verdict.")
    st.dataframe(summary, use_container_width=True, hide_index=True)


def render_outliers_tab(dataframe: pd.DataFrame) -> None:
    show_tab_intro("Unusual values", "Identify records with extreme or unexpected values.", "Flag transaction amounts far above typical values.")
    numeric_columns = get_numeric_columns(dataframe)
    if not numeric_columns:
        st.info("No numeric fields are available for unusual value analysis.")
        return
    st.markdown("#### Per-field unusual values")
    st.caption("Values outside the expected range guide for each numeric field.")
    iqr_summary = st.session_state.get("phase2_iqr_summary")
    if iqr_summary is None:
        iqr_summary, _ = detect_univariate_iqr_outliers(dataframe)
    iqr_summary.columns = ["Field", "Lower Expected Guide", "Upper Expected Guide", "Unusual values", "Share of records"]
    st.dataframe(iqr_summary, use_container_width=True, hide_index=True)

    st.markdown("#### Multi-field unusual records")
    st.caption("Records that stand out when inspecting multiple fields together.")
    show_control_help("sensitivity level (percentage of records to flag).", "Choose 5% to flag approximately 5 out of 100 records.")
    expected_share = st.slider("Sensitivity level", 0.01, 0.20, 0.05, 0.01)
    unusual_records = detect_outliers_isolation_forest(dataframe, expected_share)
    count = int(unusual_records["multivariate_outlier"].sum())
    st.metric("Unusual records flagged", count, f"{count / len(dataframe):.2%} of total records")
    if count:
        flagged = dataframe.loc[unusual_records["multivariate_outlier"]].copy()
        flagged["Unusualness score"] = unusual_records.loc[flagged.index, "outlier_score"]
        st.dataframe(flagged.sort_values("Unusualness score", ascending=False).head(100), use_container_width=True)
    elif len(numeric_columns) < 2 or len(dataframe) < 10:
        st.info("Multi-field check requires at least two numeric fields and ten records.")


def render_visualizations_tab(dataframe: pd.DataFrame) -> None:
    show_tab_intro("Visual exploration", "Explore field distributions and compare trends through interactive charts.", "Compare 'Sales' across different 'Regions'.")
    render_smart_chart_picker(dataframe)
    st.divider()
    st.markdown("#### Detailed field exploration")
    numeric_columns = get_numeric_columns(dataframe)
    # Keep high-cardinality fields selectable so the app can explain why a chart
    # is unsuitable instead of silently presenting an unreadable visual.
    categorical_columns = get_all_categorical_columns(dataframe)
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("#### Numeric field distribution")
        show_control_help("a numeric field to explore.", "Select 'Revenue', 'Age', or 'Delivery time'.")
        if numeric_columns:
            selected_number = st.selectbox("Numeric field", numeric_columns, key="number_distribution")
            missing_count = int(dataframe[selected_number].isna().sum())
            missing_percent = dataframe[selected_number].isna().mean() * 100
            st.caption(f"Missing values are excluded from this chart: {missing_count:,} ({missing_percent:.1f}%).")
            if not has_meaningful_numeric_variation(dataframe, selected_number):
                st.info(f"{selected_number} is constant or has too little variation. There is no distribution to analyze.")
            else:
                st.plotly_chart(generate_numeric_distribution_chart(dataframe, selected_number), use_container_width=True)
                st.caption("Taller bars mean more records have values in that range. The orange guide is the average; the green guide is the middle value.")
                st.plotly_chart(generate_numeric_box_plot(dataframe, selected_number), use_container_width=True)
                st.caption("The box shows the middle half of values. Individual dots beyond it are values that stand out from the usual range.")
        else:
            st.info("No numeric fields detected.")
    with right:
        st.markdown("#### Category breakdown")
        show_control_help("a category field to view frequency distributions.", "Select 'Region', 'Category', or 'Status'.")
        if categorical_columns:
            selected_category = st.selectbox("Category field", categorical_columns, key="category_distribution")
            category_guidance = get_categorical_field_guidance(dataframe, selected_category)
            if category_guidance["identifier_like"]:
                st.warning(f"{selected_category} looks like an ID or URL field. A category chart would show mostly one-off values, so choose another field.")
            else:
                if category_guidance["high_cardinality"]:
                    st.warning(f"{selected_category} has {category_guidance['unique_count']:,} unique values. Showing the top 10 only; use another field for a complete category comparison.")
                st.plotly_chart(generate_categorical_bar_chart(dataframe, selected_category, top_n=10), use_container_width=True)
                st.caption("Each bar is a category. Longer bars mean that value appears in more records. Full labels are available when you hover.")
        else:
            st.info("No category fields with manageable unique values detected.")

    if len(numeric_columns) >= 2:
        st.markdown("#### Compare two numeric measures")
        show_control_help("two numeric fields to plot on a scatter chart.", "Compare 'Advertising spend' with 'Sales', grouped by 'Region'.")
        first, second, third = st.columns(3)
        horizontal = first.selectbox("First field (X-axis)", numeric_columns, key="scatter_horizontal")
        remaining_numeric = [column for column in numeric_columns if column != horizontal]
        vertical = second.selectbox("Second field (Y-axis)", remaining_numeric, key="scatter_vertical") if remaining_numeric else horizontal
        grouping = third.selectbox("Optional grouping", ["No grouping"] + categorical_columns, key="scatter_group")
        if not has_meaningful_numeric_variation(dataframe, horizontal):
            st.warning(f"Scatter plot not meaningful because {horizontal} has no variation.")
        elif not has_meaningful_numeric_variation(dataframe, vertical):
            st.warning(f"Scatter plot not meaningful because {vertical} has no variation.")
        else:
            st.plotly_chart(generate_scatter_plot(dataframe, horizontal, vertical, None if grouping == "No grouping" else grouping), use_container_width=True)
            st.caption("Each dot is one record. The yellow line shows the overall direction; dots close to it follow the pattern more closely.")

    if numeric_columns and categorical_columns:
        st.markdown("#### Measure distribution across groups")
        show_control_help("one numeric field and one category field.", "Compare 'Order value' by 'Customer segment'.")
        first, second = st.columns(2)
        measure = first.selectbox("Numeric measure", numeric_columns, key="group_measure")
        group = second.selectbox("Grouping category", categorical_columns, key="distribution_group")
        group_guidance = get_categorical_field_guidance(dataframe, group)
        if not has_meaningful_numeric_variation(dataframe, measure):
            st.warning(f"{measure} has no variation, so comparing it across groups would not be meaningful.")
        elif group_guidance["identifier_like"]:
            st.warning(f"{group} looks like an ID or URL field, so it is not suitable for grouping.")
        elif group_guidance["high_cardinality"]:
            st.warning(f"{group} has {group_guidance['unique_count']:,} unique values. Choose a field with fewer groups for a readable comparison.")
        else:
            st.plotly_chart(generate_grouped_distribution_chart(dataframe, measure, group), use_container_width=True)
            st.caption("Compare the middle lines inside the boxes. A higher middle line means that group usually has a larger value.")

    if len(numeric_columns) >= 2:
        st.markdown("#### Multi-field relationship map")
        show_control_help("two or three numeric fields for pairwise visual comparison.", "Select 'Sales', 'Profit', and 'Quantity'.")
        selected = st.multiselect("Select fields", numeric_columns, default=numeric_columns[:min(3, len(numeric_columns))], max_selections=3)
        if len(selected) >= 2:
            constant_selected = [column for column in selected if not has_meaningful_numeric_variation(dataframe, column)]
            if constant_selected:
                st.warning(f"Relationship map not shown because {', '.join(constant_selected)} has no variation.")
            elif dataframe[selected].dropna().empty:
                st.info("No complete records available across selected fields.")
            else:
                st.pyplot(generate_pair_plot_figure(dataframe, selected), clear_figure=True)


def render_correlation_tab(dataframe: pd.DataFrame) -> None:
    show_tab_intro("Field relationships", "Discover numerical fields that rise or fall together.", "Identify metrics with strong correlation.")
    if len(get_numeric_columns(dataframe)) < 2:
        st.info("At least two numeric fields are required to calculate field relationships.")
        return
    show_control_help("relationship view mode. 'Linear' scans straight-line patterns; 'Ranked' scans monotonic trends.", "Select 'Linear' for revenue metrics.")
    method_label = st.radio("Relationship view", ["Linear", "Ranked"], horizontal=True)
    method = "pearson" if method_label == "Linear" else "spearman"
    correlation = calculate_correlation_matrix(dataframe, method)
    st.caption("Blue indicates positive movement; red indicates inverse movement. Showing top 14 fields in heatmap for clarity.")
    st.plotly_chart(generate_correlation_heatmap(correlation, "Numeric field relationship heatmap"), use_container_width=True)
    show_control_help("minimum correlation threshold for highlighting relationships.", "Set 0.70 to highlight strong relationships.")
    threshold = st.slider("Minimum strength threshold", 0.50, 0.95, 0.70, 0.05)
    strong = identify_strong_correlations(correlation, threshold)
    st.markdown("#### Strong relationships summary")
    if strong.empty:
        st.info("No field pairs meet the selected correlation strength threshold.")
    else:
        st.dataframe(strong, use_container_width=True, hide_index=True)


def main() -> None:
    apply_application_style()
    with st.sidebar:
        st.markdown("## Data Explorer")
        st.markdown("### Quick start")
        st.write("Upload a CSV, Excel, or JSON file. Data is processed in memory only.")
        st.markdown("### Report navigation")
        st.write("Start with Overview, check Data quality, then explore interactive charts.")
    render_landing_header()
    with st.container(border=True):
        st.markdown("#### Upload your dataset")
        st.caption("Supported formats: CSV, Excel (.xlsx, .xls), and JSON.")
        upload = st.file_uploader("Choose a dataset file", type=["csv", "xlsx", "xls", "json"], label_visibility="visible")
    if upload is None:
        st.info("Upload a file above to generate your exploration report.")
        return
    try:
        dataframe, source = prepare_dataset(upload.name, upload.getvalue())
    except ValueError as error:
        st.error(str(error))
        return
    except Exception:
        st.error("The file could not be processed. Please verify it is a valid tabular file and try again.")
        return

    upload_token = (upload.name, len(upload.getvalue()))
    if st.session_state.get("phase2_upload_token") != upload_token:
        st.session_state["phase2_upload_token"] = upload_token
        st.session_state.pop("phase2_pdf_bytes", None)

    quality_summary = build_quality_summary(dataframe)
    iqr_summary, _ = detect_univariate_iqr_outliers(dataframe)
    enabled_rules, custom_rule = get_phase2_rule_inputs(dataframe)
    rule_results = run_data_quality_rules(dataframe, enabled_rules, custom_rule)
    health_score = calculate_data_health_score(quality_summary, iqr_summary, rule_results)
    correlation_for_report = calculate_correlation_matrix(dataframe)
    st.session_state["phase2_iqr_summary"] = iqr_summary
    st.session_state["phase2_health_score"] = health_score.overall

    show_dataset_header(dataframe, upload.name, source)
    tabs = st.tabs(["Overview", "Data health", "Data quality", "Quality checks", "Statistics", "Unusual values", "Visual exploration", "Field relationships"])
    with tabs[0]:
        render_overview_tab(dataframe)
    with tabs[1]:
        render_health_tab(health_score, quality_summary, iqr_summary, rule_results)
        render_pdf_export(dataframe, health_score, rule_results, iqr_summary, correlation_for_report)
    with tabs[2]:
        render_quality_tab(dataframe)
    with tabs[3]:
        render_quality_checks_tab(dataframe, rule_results)
    with tabs[4]:
        render_statistics_tab(dataframe)
    with tabs[5]:
        render_outliers_tab(dataframe)
    with tabs[6]:
        render_visualizations_tab(dataframe)
    with tabs[7]:
        render_correlation_tab(dataframe)


if __name__ == "__main__":
    main()
