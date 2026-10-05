"""Statistical, correlation, and outlier analysis functions."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def generate_numeric_statistical_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return clear descriptive statistics for each numeric field."""
    numeric = dataframe.select_dtypes(include=np.number)
    if numeric.empty:
        return pd.DataFrame()
    summary = numeric.describe().T
    summary["median"] = numeric.median()
    summary["skewness"] = numeric.skew()
    summary["kurtosis"] = numeric.kurtosis()
    summary["normality p-value"] = [
        _normality_p_value(numeric[column]) for column in numeric.columns
    ]
    summary["missing_values"] = numeric.isna().sum()
    summary["missing_percent"] = numeric.isna().mean() * 100
    summary["distribution"] = [
        _interpret_distribution(numeric[column], summary.loc[column, "normality p-value"])
        for column in numeric.columns
    ]
    summary = summary.reset_index(names="Field")
    summary = summary.rename(columns={
        "count": "Records",
        "mean": "Average",
        "std": "Spread",
        "min": "Minimum",
        "25%": "25th Percentile",
        "50%": "Middle Value",
        "median": "Median",
        "75%": "75th Percentile",
        "max": "Maximum",
        "missing_values": "Missing Values",
        "missing_percent": "Missing %",
        "distribution": "Value Distribution",
    })
    cols = ["Field", "Records", "Missing Values", "Missing %", "Average", "Middle Value", "Spread", "Minimum", "Maximum", "Value Distribution"]
    return summary[cols].round(2)


def detect_univariate_iqr_outliers(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flag values beyond usual ranges for each number field."""
    numeric = dataframe.select_dtypes(include=np.number)
    flags = pd.DataFrame(False, index=dataframe.index, columns=numeric.columns)
    results = []
    for column in numeric.columns:
        series = numeric[column].dropna()
        if series.empty:
            lower = upper = np.nan
        else:
            q1, q3 = series.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            flags[column] = (numeric[column] < lower) | (numeric[column] > upper)
        count = int(flags[column].sum())
        results.append({"column": column, "lower bound": lower, "upper bound": upper,
                        "outliers": count,
                        "outlier %": round(count / len(dataframe) * 100 if len(dataframe) else 0, 2)})
    return pd.DataFrame(results).round(2), flags


def detect_outliers_isolation_forest(dataframe: pd.DataFrame, expected_share: float = 0.05, contamination: float | None = None) -> pd.DataFrame:
    """Detect unusual records across multiple fields."""
    numeric = dataframe.select_dtypes(include=np.number)
    result = pd.DataFrame(index=dataframe.index, data={"multivariate_outlier": False, "outlier_score": np.nan})
    if numeric.shape[1] < 2 or len(numeric) < 10:
        return result
    try:
        from pyod.models.iforest import IForest
    except ImportError:
        return result
    share = contamination if contamination is not None else expected_share
    features = numeric.replace([np.inf, -np.inf], np.nan).copy()
    features = features.fillna(features.median()).fillna(0)
    model = IForest(contamination=share, random_state=42, n_estimators=200)
    labels = model.fit_predict(features)
    result["multivariate_outlier"] = labels.astype(bool)
    result["outlier_score"] = model.decision_scores_
    return result


def calculate_correlation_matrix(dataframe: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    numeric = dataframe.select_dtypes(include=np.number)
    if numeric.shape[1] < 2:
        return pd.DataFrame()
    correlation = numeric.corr(method=method)
    constant_fields = [
        column for column in numeric.columns
        if numeric[column].dropna().nunique() <= 1 or np.isclose(numeric[column].dropna().std(ddof=0), 0)
    ]
    # A correlation needs variation in both fields. Explicit NaNs make this
    # business rule clear, including the constant field's diagonal cell.
    if constant_fields:
        correlation.loc[constant_fields, :] = np.nan
        correlation.loc[:, constant_fields] = np.nan
    correlation.attrs["constant_fields"] = constant_fields
    return correlation


def identify_strong_correlations(correlation: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
    """Return each high-strength relationship once, excluding self-comparisons."""
    if correlation.empty:
        return pd.DataFrame(columns=["Fields compared", "Relationship", "Direction"])
    upper = correlation.where(np.triu(np.ones(correlation.shape), k=1).astype(bool))
    pairs = upper.stack().reset_index()
    pairs.columns = ["first_field", "second_field", "relationship"]
    pairs = pairs[pairs["relationship"].abs() >= threshold].copy()
    pairs["Fields compared"] = pairs["first_field"].astype(str) + "  ↔  " + pairs["second_field"].astype(str)
    pairs["Direction"] = pairs["relationship"].map(lambda value: "Move together" if value > 0 else "Move oppositely")
    pairs = pairs.rename(columns={"relationship": "Relationship"})
    return pairs[["Fields compared", "Relationship", "Direction"]].sort_values(
        "Relationship", key=lambda series: series.abs(), ascending=False
    ).round(2)


def _normality_p_value(series: pd.Series) -> float:
    """Use D'Agostino's test; it requires at least eight finite observations."""
    values = series.replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) < 8 or values.nunique() < 2:
        return np.nan
    # The test is informative rather than definitive, particularly for very large n.
    return float(stats.normaltest(values).pvalue)


def _interpret_distribution(series: pd.Series, normality_p_value: float) -> str:
    """Describe distribution shape without making claims for sparse/constant data."""
    values = series.replace([np.inf, -np.inf], np.nan).dropna()
    if values.nunique() <= 1 or (len(values) >= 2 and np.isclose(values.std(ddof=0), 0)):
        return "Constant / No variation"
    if len(values) < 8 or values.nunique() < 3 or pd.isna(normality_p_value):
        return "Insufficient data for distribution analysis"
    return "Balanced / Bell-shaped" if normality_p_value >= 0.05 else "Skewed / Long tail"
