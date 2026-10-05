"""Phase 2 built-in and custom data-quality rules, backed by Great Expectations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RuleResult:
    rule_name: str
    passed: bool
    message: str
    enabled: bool = True


def get_builtin_rule_options(dataframe: pd.DataFrame) -> dict[str, bool]:
    """Expose only rules that match likely fields in the uploaded dataset."""
    names = [str(column).lower() for column in dataframe.columns]
    return {
        "Check non-negative ages and amounts": any(any(word in name for word in ("age", "price", "amount", "cost", "quantity")) for name in names),
        "Check percentages stay between 0 and 100": any(any(word in name for word in ("percent", "percentage", "pct")) for name in names),
        "Check end dates are not before start dates": any("start" in name for name in names) and any("end" in name for name in names),
        "Check ID fields are unique": any(name == "id" or name.endswith("_id") for name in names),
    }


def run_data_quality_rules(dataframe: pd.DataFrame, enabled_rules: dict[str, bool], custom_rule: dict[str, Any] | None = None) -> list[RuleResult]:
    """Run enabled built-ins plus one user-defined rule and return a clear checklist.

    Great Expectations is used for each column expectation when available. Native
    checks remain as a dependable fallback if an older/newer GE API is installed.
    """
    results: list[RuleResult] = []
    if enabled_rules.get("Check non-negative ages and amounts"):
        results.extend(_non_negative_results(dataframe))
    if enabled_rules.get("Check percentages stay between 0 and 100"):
        results.extend(_percentage_results(dataframe))
    if enabled_rules.get("Check end dates are not before start dates"):
        results.append(_date_order_result(dataframe))
    if enabled_rules.get("Check ID fields are unique"):
        results.extend(_unique_id_results(dataframe))
    if custom_rule:
        results.append(_custom_rule_result(dataframe, custom_rule))
    return results


def _non_negative_results(dataframe: pd.DataFrame) -> list[RuleResult]:
    results = []
    for column in dataframe.select_dtypes(include=np.number).columns:
        if any(word in column.lower() for word in ("age", "price", "amount", "cost", "quantity")):
            requires_positive = "price" in column.lower()
            passed = _expectation_or_fallback(dataframe, "expect_column_values_to_be_between", column, min_value=0, strict_min=requires_positive)
            invalid = int((dataframe[column] <= 0).sum()) if requires_positive else int((dataframe[column] < 0).sum())
            wording = "Positive values" if requires_positive else "Non-negative values"
            results.append(RuleResult(f"{column}: {wording}", passed, "All values satisfy this rule." if passed else f"{invalid:,} invalid values found."))
    return results


def _percentage_results(dataframe: pd.DataFrame) -> list[RuleResult]:
    results = []
    for column in dataframe.select_dtypes(include=np.number).columns:
        if any(word in column.lower() for word in ("percent", "percentage", "pct")):
            passed = _expectation_or_fallback(dataframe, "expect_column_values_to_be_between", column, min_value=0, max_value=100)
            invalid = int(((dataframe[column] < 0) | (dataframe[column] > 100)).sum())
            results.append(RuleResult(f"{column}: Percentage range (0–100)", passed, "All values stay between 0 and 100." if passed else f"{invalid:,} values fall outside 0 to 100."))
    return results


def _date_order_result(dataframe: pd.DataFrame) -> RuleResult:
    starts = [column for column in dataframe.columns if "start" in column.lower()]
    ends = [column for column in dataframe.columns if "end" in column.lower()]
    if not starts or not ends:
        return RuleResult("Date order check", True, "No matching start and end date fields were found.")
    start, end = starts[0], ends[0]
    invalid = int((pd.to_datetime(dataframe[end], errors="coerce") < pd.to_datetime(dataframe[start], errors="coerce")).sum())
    return RuleResult(f"{end} after {start}", invalid == 0, "All start dates occur before or on end dates." if invalid == 0 else f"{invalid:,} records have end dates before start dates.")


def _unique_id_results(dataframe: pd.DataFrame) -> list[RuleResult]:
    results = []
    for column in dataframe.columns:
        if column.lower() == "id" or column.lower().endswith("_id"):
            passed = _expectation_or_fallback(dataframe, "expect_column_values_to_be_unique", column)
            duplicates = int(dataframe[column].duplicated().sum())
            results.append(RuleResult(f"{column}: Unique identifiers", passed, "All ID values are unique." if passed else f"{duplicates:,} duplicate ID values found."))
    return results


def _custom_rule_result(dataframe: pd.DataFrame, rule: dict[str, Any]) -> RuleResult:
    column, condition = rule["column"], rule["condition"]
    series = dataframe[column]
    if condition == "Must not be blank":
        passed = _expectation_or_fallback(dataframe, "expect_column_values_to_not_be_null", column)
        count = int(series.isna().sum())
    elif condition == "Must be unique":
        passed = _expectation_or_fallback(dataframe, "expect_column_values_to_be_unique", column)
        count = int(series.duplicated().sum())
    elif condition == "Must be at least":
        value = float(rule["minimum"])
        if not pd.api.types.is_numeric_dtype(series):
            return RuleResult(f"Custom check: {column} - {condition}", False, "This rule requires a number field.")
        passed = _expectation_or_fallback(dataframe, "expect_column_values_to_be_between", column, min_value=value)
        count = int((series < value).sum())
    else:
        low, high = float(rule["minimum"]), float(rule["maximum"])
        if not pd.api.types.is_numeric_dtype(series):
            return RuleResult(f"Custom check: {column} - {condition}", False, "This rule requires a number field.")
        passed = _expectation_or_fallback(dataframe, "expect_column_values_to_be_between", column, min_value=low, max_value=high)
        count = int(((series < low) | (series > high)).sum())
    name = f"Custom check: {column} ({condition})"
    return RuleResult(name, passed, "All records satisfy this rule." if passed else f"{count:,} records do not meet this rule.")


def _expectation_or_fallback(dataframe: pd.DataFrame, expectation: str, column: str, **kwargs: Any) -> bool:
    """Evaluate through Great Expectations where its compatible pandas API exists."""
    fallback = _native_expectation(dataframe[column], expectation, **kwargs)
    try:
        import great_expectations as gx  # type: ignore

        if hasattr(gx, "from_pandas"):
            result = getattr(gx.from_pandas(dataframe), expectation)(column, **kwargs)
            return bool(result["success"])
        dataset_class = getattr(getattr(gx, "dataset", None), "PandasDataset", None)
        if dataset_class:
            result = getattr(dataset_class(dataframe), expectation)(column, **kwargs)
            return bool(result["success"])
    except Exception:
        pass
    return fallback


def _native_expectation(series: pd.Series, expectation: str, **kwargs: Any) -> bool:
    if expectation == "expect_column_values_to_not_be_null":
        return bool(series.notna().all())
    if expectation == "expect_column_values_to_be_unique":
        return bool(series.is_unique)
    lower, upper = kwargs.get("min_value", -np.inf), kwargs.get("max_value", np.inf)
    values = series.dropna()
    lower_ok = values > lower if kwargs.get("strict_min") else values >= lower
    upper_ok = values < upper if kwargs.get("strict_max") else values <= upper
    return bool((lower_ok & upper_ok).all())
