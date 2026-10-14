from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.profiling import ColumnProfile, DatasetProfile
from src.utils_stats import (
    EXTREME_IQR_MULTIPLIER,
    NUMERIC_CONVERSION_THRESHOLD,
    OUTLIER_IQR_MULTIPLIER,
    iqr_bounds,
)


@dataclass
class ColumnQuality:
    column_name: str
    role: str

    missing_count: int
    missing_percentage: float

    type_mismatch_count: int
    is_type_mismatch: bool
    conversion_rate: float | None

    # Statistical outliers detected with the standard IQR rule.
    n_outliers: int
    outlier_percentage: float

    unique_percentage: float

    top_values: list[tuple[str, int]] = field(default_factory=list)

    # Technically invalid values.
    n_invalid_values: int = 0
    invalid_percentage: float = 0.0

    # More conservative extreme values intended as cleaning candidates.
    n_extreme_values: int = 0
    extreme_percentage: float = 0.0


@dataclass
class QualityReport:
    n_duplicates: int
    duplicate_percentage: float
    columns: list[ColumnQuality]


def _check_type_mismatch(
    series: pd.Series,
    profile: ColumnProfile,
) -> tuple[int, bool, float | None]:
    """
    Check whether a column profiled as numeric is stored as text.

    Returns:
        type_mismatch_count:
            Number of non-convertible non-missing values.

        is_type_mismatch:
            True when at least 90% of the values can be interpreted
            as numeric.

        conversion_rate:
            Fraction of non-missing values that can be converted to
            numeric. None when the check is not applicable.
    """
    if profile.role != "numeric":
        return 0, False, None

    if not (
        pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)
    ):
        return 0, False, None

    non_missing = series.dropna()

    if non_missing.empty:
        return 0, False, None

    converted = pd.to_numeric(
        non_missing,
        errors="coerce",
    )

    conversion_rate = float(converted.notna().mean())
    type_mismatch_count = int(converted.isna().sum())

    is_type_mismatch = conversion_rate >= NUMERIC_CONVERSION_THRESHOLD

    return (
        type_mismatch_count,
        is_type_mismatch,
        conversion_rate,
    )


def _check_outliers(
    series: pd.Series,
) -> tuple[int, float, int, float]:
    """
    Detect statistical outliers and more extreme values using IQR.

    Two thresholds are calculated:

    - 1.5 × IQR:
        Standard statistical outlier threshold.

    - 3.0 × IQR:
        More conservative threshold for values that are
        sufficiently extreme to be considered cleaning candidates.

    Outliers are statistical extremes, not necessarily invalid data.

    Returns:
        (
            n_outliers,
            outlier_percentage,
            n_extreme_values,
            extreme_percentage,
        )
    """
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    numeric = numeric[np.isfinite(numeric)].dropna()

    if numeric.empty:
        return 0, 0.0, 0, 0.0

    outlier_bounds = iqr_bounds(numeric, OUTLIER_IQR_MULTIPLIER)

    if outlier_bounds is None:
        return 0, 0.0, 0, 0.0

    outlier_lower_bound, outlier_upper_bound = outlier_bounds

    outlier_mask = (numeric < outlier_lower_bound) | (numeric > outlier_upper_bound)

    n_outliers = int(outlier_mask.sum())

    outlier_percentage = round(
        n_outliers / len(numeric) * 100,
        2,
    )

    # Conservative cleaning threshold.
    extreme_bounds = iqr_bounds(numeric, EXTREME_IQR_MULTIPLIER)

    if extreme_bounds is None:
        return n_outliers, outlier_percentage, 0, 0.0

    extreme_lower_bound, extreme_upper_bound = extreme_bounds

    extreme_mask = (numeric < extreme_lower_bound) | (numeric > extreme_upper_bound)

    n_extreme_values = int(extreme_mask.sum())

    extreme_percentage = round(
        n_extreme_values / len(numeric) * 100,
        2,
    )

    return (
        n_outliers,
        outlier_percentage,
        n_extreme_values,
        extreme_percentage,
    )


def _check_invalid_values(
    series: pd.Series,
    profile: ColumnProfile,
) -> tuple[int, float]:
    """
    Detect technically invalid numeric values.

    At this stage the check is intentionally conservative.
    We only consider non-finite numeric values such as +inf and -inf
    invalid.
    """
    if profile.role != "numeric":
        return 0, 0.0

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    invalid_mask = numeric.notna() & ~np.isfinite(numeric)

    n_invalid = int(invalid_mask.sum())

    non_missing = int(numeric.notna().sum())

    percentage = round(n_invalid / non_missing * 100, 2) if non_missing else 0.0

    return n_invalid, percentage


def _top_values(
    series: pd.Series,
    profile: ColumnProfile,
    n: int = 5,
) -> list[tuple[str, int]]:
    """
    Return the most frequent values for categorical-like columns.
    """
    if profile.role not in {"categorical", "boolean"}:
        return []

    values = series.dropna().astype("string").value_counts().head(n)

    return [(str(value), int(count)) for value, count in values.items()]


def check_column_quality(
    series: pd.Series,
    profile: ColumnProfile,
) -> ColumnQuality:
    """
    Analyze the quality of a single column.

    This function does not modify the original data.
    """
    n_rows = len(series)

    missing_count = int(series.isna().sum())

    missing_percentage = missing_count / n_rows * 100 if n_rows else 0.0

    (
        type_mismatch_count,
        is_type_mismatch,
        conversion_rate,
    ) = _check_type_mismatch(
        series,
        profile,
    )

    (
        n_outliers,
        outlier_percentage,
        n_extreme_values,
        extreme_percentage,
    ) = _check_outliers(series) if profile.role == "numeric" else (0, 0.0, 0, 0.0)

    n_invalid_values, invalid_percentage = _check_invalid_values(
        series,
        profile,
    )

    top_values = _top_values(
        series,
        profile,
    )

    return ColumnQuality(
        column_name=profile.name,
        role=profile.role,
        missing_count=missing_count,
        missing_percentage=missing_percentage,
        type_mismatch_count=type_mismatch_count,
        is_type_mismatch=is_type_mismatch,
        conversion_rate=conversion_rate,
        n_outliers=n_outliers,
        outlier_percentage=outlier_percentage,
        unique_percentage=profile.pct_unique,
        top_values=top_values,
        n_invalid_values=n_invalid_values,
        invalid_percentage=invalid_percentage,
        n_extreme_values=n_extreme_values,
        extreme_percentage=extreme_percentage,
    )


def build_quality_report(
    df: pd.DataFrame,
    profile: DatasetProfile,
) -> QualityReport:
    """
    Build a quality report for the entire dataset.

    This function only detects quality issues.
    It never modifies the dataframe.
    """
    n_rows = len(df)

    n_duplicates = int(df.duplicated().sum())

    duplicate_percentage = n_duplicates / n_rows * 100 if n_rows else 0.0

    columns = [
        check_column_quality(
            df[column_profile.name],
            column_profile,
        )
        for column_profile in profile.columns
    ]

    return QualityReport(
        n_duplicates=n_duplicates,
        duplicate_percentage=duplicate_percentage,
        columns=columns,
    )


def quality_dataset(
    df: pd.DataFrame,
    profile: DatasetProfile,
) -> QualityReport:
    """
    Public entry point for dataset-level quality analysis.
    """
    return build_quality_report(
        df=df,
        profile=profile,
    )
