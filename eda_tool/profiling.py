"""Dataset profiling and data-quality calculations."""
from __future__ import annotations

import numpy as np
import pandas as pd


HIGH_CARDINALITY_LIMIT = 20


def infer_and_correct_data_types(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with safe numeric and date-like object columns converted.

    Conversion only happens when a substantial majority of non-empty values agree,
    preventing IDs and free-text fields from becoming misleading numeric/datetime
    fields. Numeric conversion is considered before dates because strings such as
    ``20240101`` are often codes rather than dates.
    """
    frame = dataframe.copy()
    for column in frame.select_dtypes(include=["object", "string"]).columns:
        series = frame[column].replace(r"^\s*$", np.nan, regex=True)
        non_null = series.dropna()
        if non_null.empty:
            frame[column] = series
            continue
        numeric = pd.to_numeric(non_null.astype(str).str.replace(",", "", regex=False), errors="coerce")
        if numeric.notna().mean() >= 0.9:
            frame[column] = pd.to_numeric(series.astype(str).str.replace(",", "", regex=False), errors="coerce")
            continue
        # Dates need recognizable date separators/names to avoid converting IDs.
        date_signals = non_null.astype(str).str.contains(r"[-/]|[A-Za-z]{3}", regex=True).mean()
        if date_signals >= 0.5:
            parsed_dates = pd.to_datetime(non_null, errors="coerce")
            if parsed_dates.notna().mean() >= 0.9:
                frame[column] = pd.to_datetime(series, errors="coerce")
            else:
                frame[column] = series
        else:
            frame[column] = series
    return frame


def build_column_profile(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Create a compact per-column profile suitable for display and export."""
    rows = []
    for column in dataframe.columns:
        series = dataframe[column]
        dtype_str = str(series.dtype)
        if "int" in dtype_str or "float" in dtype_str:
            user_type = "Number"
        elif "date" in dtype_str:
            user_type = "Date / Time"
        elif "bool" in dtype_str:
            user_type = "Yes / No"
        else:
            user_type = "Text / Category"
        rows.append({
            "Field Name": str(column),
            "Type": user_type,
            "Filled Cells": int(series.notna().sum()),
            "Blank Cells": int(series.isna().sum()),
            "Missing %": round(float(series.isna().mean() * 100), 2),
            "Unique Values": int(series.nunique(dropna=True)),
            "Sample Value": _format_example(series),
        })
    return pd.DataFrame(rows)


def build_quality_summary(dataframe: pd.DataFrame) -> dict[str, float | int]:
    """Calculate top-level quality metrics once for the dashboard."""
    total_cells = dataframe.shape[0] * dataframe.shape[1]
    missing_cells = int(dataframe.isna().sum().sum())
    duplicate_rows = int(dataframe.duplicated().sum())
    memory_bytes = int(dataframe.memory_usage(deep=True).sum())
    return {
        "rows": int(dataframe.shape[0]),
        "columns": int(dataframe.shape[1]),
        "memory_mb": round(memory_bytes / (1024**2), 2),
        "missing_cells": missing_cells,
        "missing_percent": round((missing_cells / total_cells * 100) if total_cells else 0, 2),
        "duplicate_rows": duplicate_rows,
        "duplicate_percent": round((duplicate_rows / len(dataframe) * 100) if len(dataframe) else 0, 2),
    }


def get_numeric_columns(dataframe: pd.DataFrame) -> list[str]:
    return dataframe.select_dtypes(include=np.number).columns.tolist()


def get_categorical_columns(dataframe: pd.DataFrame, max_unique: int = 50) -> list[str]:
    """Return suitable grouping/chart fields, excluding high-cardinality text."""
    candidates = dataframe.select_dtypes(include=["object", "string", "category", "bool"]).columns
    return [column for column in candidates if dataframe[column].nunique(dropna=True) <= max_unique]


def get_all_categorical_columns(dataframe: pd.DataFrame) -> list[str]:
    """Return all text/category fields so the UI can warn about unsuitable ones."""
    return dataframe.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()


def has_meaningful_numeric_variation(dataframe: pd.DataFrame, column: str) -> bool:
    """Return whether a numeric field has enough non-null variation to visualize."""
    if column not in dataframe or not pd.api.types.is_numeric_dtype(dataframe[column]):
        return False
    values = dataframe[column].replace([np.inf, -np.inf], np.nan).dropna()
    return len(values) >= 2 and values.nunique() > 1 and not np.isclose(values.std(ddof=0), 0)


def get_categorical_field_guidance(dataframe: pd.DataFrame, column: str) -> dict[str, object]:
    """Classify a category field before presenting it in a chart or grouping control."""
    values = dataframe[column].dropna().astype(str)
    count = len(values)
    unique_count = int(values.nunique())
    unique_ratio = unique_count / count if count else 0.0
    name = str(column).lower()
    url_ratio = values.str.match(r"^(https?|ftp)://", case=False, na=False).mean() if count else 0.0
    id_named = name == "id" or name.endswith("_id") or name.endswith("id")
    identifier_like = bool(id_named or url_ratio >= 0.5 or (unique_ratio >= 0.9 and unique_count > HIGH_CARDINALITY_LIMIT))
    high_cardinality = bool(unique_count > HIGH_CARDINALITY_LIMIT or unique_ratio > 0.5)
    return {
        "unique_count": unique_count,
        "unique_ratio": unique_ratio,
        "high_cardinality": high_cardinality,
        "identifier_like": identifier_like,
        "url_like": bool(url_ratio >= 0.5),
    }


def _format_example(series: pd.Series) -> str:
    values = series.dropna()
    if values.empty:
        return "—"
    value = str(values.iloc[0])
    return value[:80] + ("…" if len(value) > 80 else "")
