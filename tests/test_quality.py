import pandas as pd

from src.profiling import (
    ColumnProfile,
    profile_dataset,
)
from src.quality import (
    QualityReport,
    check_column_quality,
    quality_dataset,
    _check_outliers,
)


def test_numeric_column_has_no_type_mismatch():
    series = pd.Series([1.0, 2.5, 3.3, 4.1, 5.0])

    profile = ColumnProfile(
        name="price",
        dtype="float64",
        role="numeric",
        n_missing=0,
        pct_missing=0.0,
        n_unique=5,
        pct_unique=100.0,
        sample_values=[1.0, 2.5, 3.3, 4.1, 5.0],
    )

    quality = check_column_quality(series, profile)

    assert quality.type_mismatch_count == 0
    assert quality.conversion_rate is None
    assert quality.is_type_mismatch is False


def test_string_numeric_column_is_detected_as_type_mismatch():
    series = pd.Series(["1", "2", "3", "4", "5"])

    profile = ColumnProfile(
        name="price",
        dtype="object",
        role="numeric",
        n_missing=0,
        pct_missing=0.0,
        n_unique=5,
        pct_unique=100.0,
        sample_values=["1", "2", "3", "4", "5"],
    )

    quality = check_column_quality(series, profile)

    assert quality.type_mismatch_count == 0
    assert quality.conversion_rate == 1.0
    assert quality.is_type_mismatch is True


def test_string_column_with_invalid_numeric_values_is_not_confirmed_as_type_mismatch():
    series = pd.Series(["1", "2", "three", "4", "five"])

    profile = ColumnProfile(
        name="price",
        dtype="object",
        role="numeric",
        n_missing=0,
        pct_missing=0.0,
        n_unique=5,
        pct_unique=100.0,
        sample_values=["1", "2", "three", "4", "five"],
    )

    quality = check_column_quality(series, profile)

    assert quality.type_mismatch_count == 2
    assert quality.conversion_rate == 0.6
    assert quality.is_type_mismatch is False


def test_missing_values_are_counted():
    series = pd.Series([1, 2, 3, None, 5, None, 7, 8, 9, 10])

    profile = ColumnProfile(
        name="value",
        dtype="float64",
        role="numeric",
        n_missing=2,
        pct_missing=20.0,
        n_unique=8,
        pct_unique=80.0,
        sample_values=[1, 2, 3, 5, 7],
    )

    quality = check_column_quality(series, profile)

    assert quality.missing_count == 2
    assert quality.missing_percentage == 20.0


def test_top_values_are_returned():
    series = pd.Series(["A", "A", "A", "B", "B", "C"])

    profile = ColumnProfile(
        name="category",
        dtype="object",
        role="categorical",
        n_missing=0,
        pct_missing=0.0,
        n_unique=3,
        pct_unique=50.0,
        sample_values=["A", "B", "C"],
    )

    quality = check_column_quality(series, profile)

    assert quality.top_values[0] == ("A", 3)
    assert quality.top_values[1] == ("B", 2)
    assert quality.top_values[2] == ("C", 1)


def test_dataset_quality_counts_duplicate_rows():
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 3, 4, 5, 6, 7, 8, 9],
            "value": [10, 20, 30, 30, 40, 50, 60, 70, 80, 90],
        }
    )

    profile = profile_dataset(df)

    report = quality_dataset(df, profile)

    assert isinstance(report, QualityReport)
    assert report.n_duplicates == 1
    assert report.duplicate_percentage == 10.0


def test_dataset_quality_with_two_duplicate_rows():
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 3, 4, 5, 6, 7, 8, 8],
            "value": [10, 20, 30, 30, 40, 50, 60, 70, 80, 80],
        }
    )

    profile = profile_dataset(df)

    report = quality_dataset(df, profile)

    assert report.n_duplicates == 2
    assert report.duplicate_percentage == 20.0


def test_empty_dataset_has_zero_duplicate_percentage():
    df = pd.DataFrame(columns=["id", "value"])

    profile = profile_dataset(df)

    report = quality_dataset(df, profile)

    assert report.n_duplicates == 0
    assert report.duplicate_percentage == 0.0


def test_quality_dataset_returns_quality_for_each_column():
    df = pd.DataFrame(
        {
            "price": [10, 20, 30],
            "category": ["A", "B", "A"],
        }
    )

    profile = profile_dataset(df)

    report = quality_dataset(df, profile)

    assert len(report.columns) == 2
    assert report.columns[0].column_name == "price"
    assert report.columns[1].column_name == "category"


def test_numeric_series_has_no_outliers():
    series = pd.Series([10, 11, 12, 13, 14])

    n_outliers, percentage = _check_outliers(series)

    assert n_outliers == 0
    assert percentage == 0.0


def test_numeric_series_detects_outliers():
    series = pd.Series([10, 11, 12, 13, 14, 100])

    n_outliers, percentage = _check_outliers(series)

    assert n_outliers == 1
    assert percentage == round(1 / 6 * 100, 2)


def test_outlier_detection_ignores_missing_values():
    series = pd.Series([10, 11, 12, 13, 14, 100, None])

    n_outliers, percentage = _check_outliers(series)

    assert n_outliers == 1
    assert percentage == round(1 / 6 * 100, 2)


def test_empty_series_has_no_outliers():
    series = pd.Series([], dtype="float64")

    n_outliers, percentage = _check_outliers(series)

    assert n_outliers == 0
    assert percentage == 0.0


def test_numeric_column_reports_outliers():
    series = pd.Series([10, 11, 12, 13, 14, 100])

    profile = ColumnProfile(
        name="price",
        dtype="int64",
        role="numeric",
        n_missing=0,
        pct_missing=0.0,
        n_unique=6,
        pct_unique=100.0,
        sample_values=[10, 11, 12, 13, 14],
    )

    quality = check_column_quality(series, profile)

    assert quality.n_outliers == 1
    assert quality.outlier_percentage == round(1 / 6 * 100, 2)


def test_categorical_column_has_no_outliers():
    series = pd.Series(["A", "A", "B", "B", "C", "C"])

    profile = ColumnProfile(
        name="category",
        dtype="object",
        role="categorical",
        n_missing=0,
        pct_missing=0.0,
        n_unique=3,
        pct_unique=50.0,
        sample_values=["A", "B", "C"],
    )

    quality = check_column_quality(series, profile)

    assert quality.n_outliers == 0
    assert quality.outlier_percentage == 0.0
