from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.profiling import DatasetProfile
from src.quality import QualityReport
from src.utils_stats import (
    NUMERIC_CONVERSION_THRESHOLD,
    OUTLIER_IQR_MULTIPLIER,
    iqr_outlier_mask,
)

MISSING_STRATEGIES = {"keep", "remove", "mean", "median", "mode"}
TYPE_CONVERSION_STRATEGIES = {"keep", "convert"}
OUTLIER_STRATEGIES = {"keep", "remove"}


@dataclass
class CleaningAction:
    action: str
    column_name: str | None
    affected_rows: int
    description: str
    original_dtype: str | None = None
    resulting_dtype: str | None = None


@dataclass
class CleaningConfig:
    remove_duplicates: bool = True

    missing_numeric_strategy: str = "median"
    missing_categorical_strategy: str = "mode"

    replace_invalid_values: bool = True
    convert_type_mismatches: bool = False

    outlier_strategy: str = "keep"
    outlier_columns: set[str] = field(default_factory=set)

    numeric_ranges: dict[
        str,
        tuple[float | None, float | None],
    ] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.missing_numeric_strategy not in MISSING_STRATEGIES:
            raise ValueError(
                f"Invalid numeric missing strategy: {self.missing_numeric_strategy}"
            )

        if self.missing_categorical_strategy not in MISSING_STRATEGIES:
            raise ValueError(
                f"Invalid categorical missing strategy: "
                f"{self.missing_categorical_strategy}"
            )

        if self.outlier_strategy not in OUTLIER_STRATEGIES:
            raise ValueError(f"Invalid outlier strategy: {self.outlier_strategy}")

        for column_name, bounds in self.numeric_ranges.items():
            if len(bounds) != 2:
                raise ValueError(
                    f"Numeric range for '{column_name}' must contain "
                    "exactly two values: (min, max)."
                )

            lower, upper = bounds

            if lower is not None and upper is not None and lower > upper:
                raise ValueError(
                    f"Invalid numeric range for '{column_name}': "
                    f"minimum ({lower}) is greater than maximum ({upper})."
                )


@dataclass
class CleaningReport:
    original_rows: int
    final_rows: int

    duplicates_removed: int = 0
    missing_rows_removed: int = 0
    missing_values_filled: int = 0
    invalid_values_replaced: int = 0
    type_conversions: int = 0

    outlier_rows_removed: int = 0
    range_rows_removed: int = 0

    actions: list[CleaningAction] = field(default_factory=list)


def _replace_invalid_values(
    df: pd.DataFrame,
    profile: DatasetProfile,
    report: CleaningReport,
) -> None:
    for column in profile.columns:
        if column.role != "numeric":
            continue

        series = df[column.name]

        if not pd.api.types.is_numeric_dtype(series):
            continue

        invalid_mask = np.isinf(series)
        invalid_count = int(invalid_mask.sum())

        if invalid_count == 0:
            continue

        df.loc[invalid_mask, column.name] = np.nan
        report.invalid_values_replaced += invalid_count

        report.actions.append(
            CleaningAction(
                action="replace_invalid_values",
                column_name=column.name,
                affected_rows=invalid_count,
                description=(
                    f"Replaced {invalid_count} invalid numeric values "
                    f"in '{column.name}' with missing values."
                ),
                original_dtype=str(series.dtype),
                resulting_dtype=str(df[column.name].dtype),
            )
        )


def _remove_duplicates(
    df: pd.DataFrame,
    report: CleaningReport,
) -> pd.DataFrame:
    duplicate_mask = df.duplicated(keep="first")
    duplicate_count = int(duplicate_mask.sum())

    if duplicate_count == 0:
        return df

    cleaned_df = df.loc[~duplicate_mask].copy()

    report.duplicates_removed = duplicate_count

    report.actions.append(
        CleaningAction(
            action="remove_duplicates",
            column_name=None,
            affected_rows=duplicate_count,
            description=(f"Removed {duplicate_count} duplicate rows."),
        )
    )

    return cleaned_df


def _fill_numeric_missing(
    df: pd.DataFrame,
    profile: DatasetProfile,
    strategy: str,
    report: CleaningReport,
) -> pd.DataFrame:
    if strategy == "keep":
        return df

    cleaned_df = df.copy()

    for column in profile.columns:
        if column.role != "numeric":
            continue

        series = cleaned_df[column.name]

        if not pd.api.types.is_numeric_dtype(series):
            continue

        missing_mask = series.isna()
        missing_count = int(missing_mask.sum())

        if missing_count == 0:
            continue

        if strategy == "remove":
            cleaned_df = cleaned_df.loc[~missing_mask].copy()

            report.missing_rows_removed += missing_count

            report.actions.append(
                CleaningAction(
                    action="remove_missing_rows",
                    column_name=column.name,
                    affected_rows=missing_count,
                    description=(
                        f"Removed {missing_count} rows with missing "
                        f"values in '{column.name}'."
                    ),
                )
            )

            continue

        if strategy == "mode":
            raise ValueError("Mode strategy is not supported for numeric columns.")

        if strategy == "mean":
            replacement = series.mean()
        elif strategy == "median":
            replacement = series.median()
        else:
            raise ValueError(f"Unsupported numeric missing strategy: {strategy}")

        if pd.isna(replacement):
            continue

        cleaned_df.loc[missing_mask, column.name] = replacement

        report.missing_values_filled += missing_count

        report.actions.append(
            CleaningAction(
                action="fill_missing_values",
                column_name=column.name,
                affected_rows=missing_count,
                description=(
                    f"Filled {missing_count} missing values in "
                    f"'{column.name}' using the {strategy}."
                ),
            )
        )

    return cleaned_df


def _fill_categorical_missing(
    df: pd.DataFrame,
    profile: DatasetProfile,
    strategy: str,
    report: CleaningReport,
) -> pd.DataFrame:
    if strategy == "keep":
        return df

    cleaned_df = df.copy()

    for column in profile.columns:
        if column.role not in {"categorical", "boolean"}:
            continue

        series = cleaned_df[column.name]

        missing_mask = series.isna()
        missing_count = int(missing_mask.sum())

        if missing_count == 0:
            continue

        if strategy == "remove":
            cleaned_df = cleaned_df.loc[~missing_mask].copy()

            report.missing_rows_removed += missing_count

            report.actions.append(
                CleaningAction(
                    action="remove_missing_rows",
                    column_name=column.name,
                    affected_rows=missing_count,
                    description=(
                        f"Removed {missing_count} rows with missing "
                        f"values in '{column.name}'."
                    ),
                )
            )

            continue

        if strategy != "mode":
            raise ValueError(f"Unsupported categorical missing strategy: {strategy}")

        mode = series.mode(dropna=True)

        if mode.empty:
            continue

        replacement = mode.iloc[0]

        cleaned_df.loc[missing_mask, column.name] = replacement

        report.missing_values_filled += missing_count

        report.actions.append(
            CleaningAction(
                action="fill_missing_values",
                column_name=column.name,
                affected_rows=missing_count,
                description=(
                    f"Filled {missing_count} missing values in "
                    f"'{column.name}' using the mode."
                ),
            )
        )

    return cleaned_df


def _handle_missing_values(
    df: pd.DataFrame,
    profile: DatasetProfile,
    config: CleaningConfig,
    report: CleaningReport,
) -> pd.DataFrame:
    cleaned_df = _fill_numeric_missing(
        df,
        profile,
        config.missing_numeric_strategy,
        report,
    )

    cleaned_df = _fill_categorical_missing(
        cleaned_df,
        profile,
        config.missing_categorical_strategy,
        report,
    )

    return cleaned_df


def _convert_type_mismatches(
    df: pd.DataFrame,
    profile: DatasetProfile,
    quality: QualityReport,
    report: CleaningReport,
) -> pd.DataFrame:
    cleaned_df = df.copy()

    quality_by_column = {column.column_name: column for column in quality.columns}

    for column in profile.columns:
        if column.role != "numeric":
            continue

        quality_column = quality_by_column.get(column.name)

        if quality_column is None:
            continue

        conversion_rate = quality_column.conversion_rate

        if conversion_rate is None:
            continue

        if conversion_rate < NUMERIC_CONVERSION_THRESHOLD:
            continue

        series = cleaned_df[column.name]

        if pd.api.types.is_numeric_dtype(series):
            continue

        converted = pd.to_numeric(series, errors="coerce")

        original_non_missing = series.notna()
        converted_non_missing = converted.notna()

        if not converted_non_missing[original_non_missing].all():
            continue

        cleaned_df[column.name] = converted

        report.type_conversions += 1

        report.actions.append(
            CleaningAction(
                action="convert_type",
                column_name=column.name,
                affected_rows=int(original_non_missing.sum()),
                description=(
                    f"Converted '{column.name}' from {series.dtype} to numeric."
                ),
                original_dtype=str(series.dtype),
                resulting_dtype=str(converted.dtype),
            )
        )

    return cleaned_df


def _find_outlier_rows(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.Series:
    """
    Return a boolean mask indicating rows that are statistical
    outliers in at least one selected numeric column.

    Outliers are defined using the standard 1.5 × IQR rule.
    """

    outlier_mask = pd.Series(False, index=df.index)

    for column_name in columns:
        outlier_mask |= iqr_outlier_mask(
            df[column_name],
            OUTLIER_IQR_MULTIPLIER,
        )

    return outlier_mask


def _remove_outliers(
    df: pd.DataFrame,
    profile: DatasetProfile,
    config: CleaningConfig,
    report: CleaningReport,
) -> pd.DataFrame:
    if config.outlier_strategy == "keep":
        return df

    if not config.outlier_columns:
        return df

    numeric_columns = {
        column.name for column in profile.columns if column.role == "numeric"
    }

    unknown_columns = config.outlier_columns - numeric_columns

    if unknown_columns:
        raise ValueError(
            "Outlier removal was requested for non-numeric or unknown "
            f"columns: {sorted(unknown_columns)}"
        )

    selected_columns = [
        column for column in config.outlier_columns if column in df.columns
    ]

    if not selected_columns:
        return df

    outlier_mask = _find_outlier_rows(
        df,
        selected_columns,
    )

    outlier_count = int(outlier_mask.sum())

    if outlier_count == 0:
        return df

    cleaned_df = df.loc[~outlier_mask].copy()

    report.outlier_rows_removed = outlier_count

    report.actions.append(
        CleaningAction(
            action="remove_outliers",
            column_name=None,
            affected_rows=outlier_count,
            description=(
                f"Removed {outlier_count} rows containing "
                "statistical outliers in the selected numeric columns "
                "using the 1.5 × IQR rule."
            ),
        )
    )

    return cleaned_df


def _apply_numeric_ranges(
    df: pd.DataFrame,
    profile: DatasetProfile,
    numeric_ranges: dict[
        str,
        tuple[float | None, float | None],
    ],
    report: CleaningReport,
) -> pd.DataFrame:
    """
    Remove rows outside user-defined numeric ranges.

    Missing values are not considered outside the range. They remain
    available for the configured missing-value strategy.
    """

    if not numeric_ranges:
        return df

    cleaned_df = df.copy()

    numeric_profile_columns = {
        column.name for column in profile.columns if column.role == "numeric"
    }

    unknown_columns = set(numeric_ranges) - numeric_profile_columns

    if unknown_columns:
        raise ValueError(
            "Numeric ranges were provided for non-numeric or unknown "
            f"columns: {sorted(unknown_columns)}"
        )

    for column_name, (lower_bound, upper_bound) in numeric_ranges.items():
        if column_name not in cleaned_df.columns:
            raise ValueError(f"Numeric range references unknown column: {column_name}")

        series = pd.to_numeric(
            cleaned_df[column_name],
            errors="coerce",
        )

        outside_mask = pd.Series(False, index=cleaned_df.index)

        if lower_bound is not None:
            outside_mask |= series < lower_bound

        if upper_bound is not None:
            outside_mask |= series > upper_bound

        outside_mask &= series.notna()

        affected_rows = int(outside_mask.sum())

        if affected_rows == 0:
            continue

        cleaned_df = cleaned_df.loc[~outside_mask].copy()

        report.range_rows_removed += affected_rows

        if lower_bound is None:
            range_description = f"maximum {upper_bound}"
        elif upper_bound is None:
            range_description = f"minimum {lower_bound}"
        else:
            range_description = f"{lower_bound} to {upper_bound}"

        report.actions.append(
            CleaningAction(
                action="apply_numeric_range",
                column_name=column_name,
                affected_rows=affected_rows,
                description=(
                    f"Removed {affected_rows} rows outside the "
                    f"allowed range ({range_description}) for "
                    f"'{column_name}'."
                ),
            )
        )

    return cleaned_df


def clean_dataset(
    df: pd.DataFrame,
    profile: DatasetProfile,
    quality: QualityReport,
    config: CleaningConfig,
) -> tuple[pd.DataFrame, CleaningReport]:
    """
    Apply explicit cleaning decisions to a dataset.

    Detection and cleaning are deliberately separated:
    - profiling/quality detect possible problems
    - CleaningConfig describes what the user wants changed
    - this function applies those decisions
    """

    cleaned_df = df.copy()

    report = CleaningReport(
        original_rows=len(cleaned_df),
        final_rows=len(cleaned_df),
    )

    if config.replace_invalid_values:
        _replace_invalid_values(
            cleaned_df,
            profile,
            report,
        )

    if config.remove_duplicates:
        cleaned_df = _remove_duplicates(
            cleaned_df,
            report,
        )

    if config.convert_type_mismatches:
        cleaned_df = _convert_type_mismatches(
            cleaned_df,
            profile,
            quality,
            report,
        )

    cleaned_df = _handle_missing_values(
        cleaned_df,
        profile,
        config,
        report,
    )

    cleaned_df = _remove_outliers(
        cleaned_df,
        profile,
        config,
        report,
    )

    cleaned_df = _apply_numeric_ranges(
        cleaned_df,
        profile,
        config.numeric_ranges,
        report,
    )

    report.final_rows = len(cleaned_df)

    return cleaned_df, report
