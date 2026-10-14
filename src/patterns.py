"""Detect descriptive statistical patterns in unknown datasets.

Responsibilities:
- Detect descriptive differences between groups.
- Detect group means that are distorted by extreme values.
- Detect numeric outliers.
- Return quantitative evidence for later interpretation.

This module does not remove data, perform hypothesis tests, establish
causality, or generate natural-language insights.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.utils_stats import (
    OUTLIER_IQR_MULTIPLIER,
    finite_numeric,
    iqr_bounds,
)


@dataclass
class GroupDifferencePattern:
    group_column: str
    numeric_column: str
    group_value: str
    group_size: int

    group_mean: float
    overall_mean: float
    difference: float
    standardized_difference: float

    group_median: float
    overall_median: float
    median_difference: float
    robust_standardized_difference: float


@dataclass
class GroupMeanDistortionPattern:
    group_column: str
    numeric_column: str
    group_value: str
    group_size: int

    outlier_count: int
    group_mean: float
    mean_without_outliers: float
    group_median: float

    mean_shift: float
    mean_shift_relative_to_median: float | None

    outlier_values: list[float] = field(default_factory=list)


@dataclass
class NumericOutlierPattern:
    column: str
    n_observations: int
    outlier_count: int
    lower_bound: float
    upper_bound: float
    outlier_positions: list[int] = field(default_factory=list)
    outlier_values: list[float] = field(default_factory=list)


def _robust_scale(values: pd.Series) -> float:
    """Estimate scale from IQR, falling back to standard deviation."""
    q1 = float(values.quantile(0.25))
    q3 = float(values.quantile(0.75))

    scale = (q3 - q1) / 1.349

    if np.isfinite(scale) and scale > 0:
        return scale

    std = float(values.std(ddof=0))

    if np.isfinite(std) and std > 0:
        return std

    return 0.0


def detect_group_differences(
    df: pd.DataFrame,
    group_column: str,
    numeric_column: str,
    *,
    min_group_size: int = 5,
    min_standardized_difference: float = 1.0,
    max_groups: int = 20,
) -> list[GroupDifferencePattern]:
    """Detect groups whose means or medians differ from the overall data.

    Mean differences are standardized using pooled within-group standard
    deviation. Median differences use a robust within-group scale, with an
    overall robust scale as a fallback.

    Thresholds are descriptive heuristics, not significance tests.
    """
    if min_group_size < 1:
        raise ValueError("min_group_size must be at least 1.")

    if not np.isfinite(min_standardized_difference) or min_standardized_difference < 0:
        raise ValueError("min_standardized_difference must be finite and non-negative.")

    if max_groups < 2:
        raise ValueError("max_groups must be at least 2.")

    if group_column not in df.columns or numeric_column not in df.columns:
        return []

    data = pd.DataFrame(
        {
            "group": df[group_column],
            "value": finite_numeric(df[numeric_column]),
        }
    ).dropna()

    if len(data) < 2:
        return []

    grouped = data.groupby("group", observed=True, sort=False)["value"]

    if not 2 <= grouped.ngroups <= max_groups:
        return []

    overall_mean = float(data["value"].mean())
    overall_median = float(data["value"].median())

    # Estimate variation within groups, excluding between-group differences.
    pooled_sum_squares = 0.0
    pooled_degrees_of_freedom = 0

    for _, values in grouped:
        if len(values) > 1:
            pooled_sum_squares += float(((values - values.mean()) ** 2).sum())
            pooled_degrees_of_freedom += len(values) - 1

    pooled_std = (
        float(np.sqrt(pooled_sum_squares / pooled_degrees_of_freedom))
        if pooled_degrees_of_freedom > 0
        else 0.0
    )

    within_scales = [_robust_scale(values) for _, values in grouped if len(values) >= 2]

    positive_scales = [scale for scale in within_scales if scale > 0]

    robust_within_scale = float(np.median(positive_scales)) if positive_scales else 0.0
    # Used only if within-group variation is exactly zero.
    overall_robust_scale = _robust_scale(data["value"])

    patterns: list[GroupDifferencePattern] = []

    for group_value, values in grouped:
        group_size = len(values)

        if group_size < min_group_size:
            continue

        group_mean = float(values.mean())
        group_median = float(values.median())

        difference = group_mean - overall_mean
        median_difference = group_median - overall_median

        mean_scale = pooled_std or overall_robust_scale

        standardized_difference = difference / mean_scale if mean_scale > 0 else 0.0

        robust_standardized_difference = (
            median_difference / robust_within_scale if robust_within_scale > 0 else 0.0
        )

        mean_signal = abs(standardized_difference) >= min_standardized_difference

        median_signal = abs(
            robust_standardized_difference
        ) >= min_standardized_difference or (
            robust_within_scale == 0 and not np.isclose(median_difference, 0.0)
        )

        zero_within_variation = pooled_std == 0

        if zero_within_variation:
            mean_signal = mean_signal or not np.isclose(difference, 0.0)
            median_signal = median_signal or not np.isclose(median_difference, 0.0)

        if not mean_signal and not median_signal:
            continue
        patterns.append(
            GroupDifferencePattern(
                group_column=group_column,
                numeric_column=numeric_column,
                group_value=str(group_value),
                group_size=group_size,
                group_mean=group_mean,
                overall_mean=overall_mean,
                difference=difference,
                standardized_difference=standardized_difference,
                group_median=group_median,
                overall_median=overall_median,
                median_difference=median_difference,
                robust_standardized_difference=(robust_standardized_difference),
            )
        )

    return sorted(
        patterns,
        key=lambda pattern: max(
            abs(pattern.standardized_difference),
            abs(pattern.robust_standardized_difference),
        ),
        reverse=True,
    )


def detect_group_mean_distortions(
    df: pd.DataFrame,
    group_column: str,
    numeric_column: str,
    *,
    min_group_size: int = 5,
    iqr_multiplier: float = OUTLIER_IQR_MULTIPLIER,
    min_relative_mean_shift: float = 0.25,
    max_groups: int = 20,
) -> list[GroupMeanDistortionPattern]:
    """Detect groups whose means change substantially after excluding IQR outliers.

    Outliers are excluded only for this diagnostic calculation. The input
    DataFrame is never modified.

    A pattern is returned when at least one IQR outlier exists and excluding
    those values changes the mean by at least min_relative_mean_shift times
    the absolute group median. When the median is zero, any nonzero change
    in the mean qualifies.
    """
    if min_group_size < 4:
        raise ValueError("min_group_size must be at least 4.")

    if not np.isfinite(iqr_multiplier) or iqr_multiplier < 0:
        raise ValueError("iqr_multiplier must be finite and non-negative.")

    if not np.isfinite(min_relative_mean_shift) or min_relative_mean_shift < 0:
        raise ValueError("min_relative_mean_shift must be finite and non-negative.")

    if max_groups < 2:
        raise ValueError("max_groups must be at least 2.")

    if group_column not in df.columns or numeric_column not in df.columns:
        return []

    data = pd.DataFrame(
        {
            "group": df[group_column],
            "value": finite_numeric(df[numeric_column]),
        }
    ).dropna()

    grouped = data.groupby("group", observed=True, sort=False)

    if not 2 <= grouped.ngroups <= max_groups:
        return []

    patterns: list[GroupMeanDistortionPattern] = []

    for group_value, group_data in grouped:
        values = group_data["value"]

        if len(values) < min_group_size:
            continue

        bounds = iqr_bounds(
            values,
            multiplier=iqr_multiplier,
        )

        if bounds is None:
            continue

        lower_bound, upper_bound = bounds

        outlier_mask = (values < lower_bound) | (values > upper_bound)
        outliers = values.loc[outlier_mask]

        if outliers.empty:
            continue

        remaining = values.loc[~outlier_mask]

        if remaining.empty:
            continue

        group_mean = float(values.mean())
        mean_without_outliers = float(remaining.mean())
        group_median = float(values.median())

        mean_shift = group_mean - mean_without_outliers

        relative_shift = (
            abs(mean_shift) / abs(group_median) if group_median != 0 else None
        )

        if group_median == 0:
            significant_shift = mean_shift != 0
        else:
            significant_shift = (
                relative_shift is not None and relative_shift >= min_relative_mean_shift
            )

        if not significant_shift:
            continue

        patterns.append(
            GroupMeanDistortionPattern(
                group_column=group_column,
                numeric_column=numeric_column,
                group_value=str(group_value),
                group_size=len(values),
                outlier_count=len(outliers),
                group_mean=group_mean,
                mean_without_outliers=mean_without_outliers,
                group_median=group_median,
                mean_shift=mean_shift,
                mean_shift_relative_to_median=relative_shift,
                outlier_values=outliers.astype(float).tolist(),
            )
        )

    return sorted(
        patterns,
        key=lambda pattern: abs(pattern.mean_shift),
        reverse=True,
    )


def detect_numeric_outliers(
    df: pd.DataFrame,
    numeric_columns: list[str],
    *,
    min_observations: int = 4,
    iqr_multiplier: float = OUTLIER_IQR_MULTIPLIER,
) -> list[NumericOutlierPattern]:
    """Detect IQR outliers in explicitly selected numeric columns.

    outlier_positions contains zero-based row positions, not DataFrame index
    labels. Numeric identifiers should not be passed unless they are intended
    for statistical analysis.
    """
    if min_observations < 4:
        raise ValueError("min_observations must be at least 4.")

    if not np.isfinite(iqr_multiplier) or iqr_multiplier < 0:
        raise ValueError("iqr_multiplier must be finite and non-negative.")

    patterns: list[NumericOutlierPattern] = []

    for column in dict.fromkeys(numeric_columns):
        if column not in df.columns:
            continue

        numeric = finite_numeric(df[column])
        valid_mask = numeric.notna()
        valid_values = numeric.loc[valid_mask]

        if len(valid_values) < min_observations:
            continue

        bounds = iqr_bounds(
            valid_values,
            multiplier=iqr_multiplier,
        )

        if bounds is None:
            continue

        lower_bound, upper_bound = bounds

        outlier_mask = valid_mask & ((numeric < lower_bound) | (numeric > upper_bound))

        positions = np.flatnonzero(outlier_mask.to_numpy())

        if len(positions) == 0:
            continue

        patterns.append(
            NumericOutlierPattern(
                column=column,
                n_observations=len(valid_values),
                outlier_count=len(positions),
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                outlier_positions=positions.tolist(),
                outlier_values=numeric.iloc[positions].astype(float).tolist(),
            )
        )

    return patterns
