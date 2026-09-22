import pandas as pd

from src.preparation import (
    prepare_categorical_series,
    prepare_numeric_series,
)
from src.profiling import ColumnProfile
from src.quality import ColumnQuality


def make_column_profile(
    name: str,
    role: str,
) -> ColumnProfile:
    return ColumnProfile(
        name=name,
        dtype="object",
        role=role,
        n_missing=0,
        pct_missing=0.0,
        n_unique=3,
        pct_unique=30.0,
        sample_values=[],
    )


def make_column_quality(
    name: str,
    conversion_rate: float | None,
) -> ColumnQuality:
    return ColumnQuality(
        column_name=name,
        role="numeric",
        missing_count=0,
        missing_percentage=0.0,
        type_mismatch_count=0,
        is_type_mismatch=False,
        conversion_rate=conversion_rate,
        n_outliers=0,
        outlier_percentage=0.0,
        unique_percentage=30.0,
        top_values=[],
    )


def test_prepare_categorical_series_normalizes_values():
    series = pd.Series([" Laptop ", "LAPTOP", "laptop", "Mouse"])

    profile = make_column_profile(
        "product",
        "categorical",
    )

    prepared = prepare_categorical_series(
        series,
        profile,
    )

    assert prepared is not None
    assert prepared.tolist() == [
        "laptop",
        "laptop",
        "laptop",
        "mouse",
    ]

    assert str(prepared.dtype) == "category"


def test_prepare_categorical_series_preserves_missing_values():
    series = pd.Series(["Laptop", None, "Mouse"])

    profile = make_column_profile(
        "product",
        "categorical",
    )

    prepared = prepare_categorical_series(
        series,
        profile,
    )

    assert prepared is not None
    assert prepared.isna().tolist() == [
        False,
        True,
        False,
    ]


def test_prepare_categorical_series_returns_none_for_non_categorical():
    series = pd.Series([1, 2, 3])

    profile = make_column_profile(
        "quantity",
        "numeric",
    )

    prepared = prepare_categorical_series(
        series,
        profile,
    )

    assert prepared is None


def test_prepare_categorical_series_does_not_modify_original():
    series = pd.Series([" Laptop ", "LAPTOP"])

    original = series.copy()

    profile = make_column_profile(
        "product",
        "categorical",
    )

    prepare_categorical_series(
        series,
        profile,
    )

    pd.testing.assert_series_equal(
        series,
        original,
    )


def test_prepare_numeric_series_returns_numeric_copy():
    series = pd.Series([10, 20, 30])

    profile = make_column_profile(
        "quantity",
        "numeric",
    )

    quality = make_column_quality(
        "quantity",
        None,
    )

    prepared = prepare_numeric_series(
        series,
        profile,
        quality,
    )

    assert prepared is not None
    assert prepared.tolist() == [10, 20, 30]
    assert pd.api.types.is_numeric_dtype(prepared)


def test_prepare_numeric_series_converts_numeric_strings():
    series = pd.Series(["10", "20", "30", "40"])

    profile = make_column_profile(
        "quantity",
        "numeric",
    )

    quality = make_column_quality(
        "quantity",
        1.0,
    )

    prepared = prepare_numeric_series(
        series,
        profile,
        quality,
    )

    assert prepared is not None
    assert prepared.tolist() == [
        10,
        20,
        30,
        40,
    ]


def test_prepare_numeric_series_converts_invalid_values_to_missing():
    series = pd.Series(["10", "20", "invalid", "40"])

    profile = make_column_profile(
        "quantity",
        "numeric",
    )

    quality = make_column_quality(
        "quantity",
        0.75,
    )

    prepared = prepare_numeric_series(
        series,
        profile,
        quality,
    )

    assert prepared is None


def test_prepare_numeric_series_returns_none_when_conversion_rate_is_too_low():
    series = pd.Series(["10", "invalid", "invalid", "40"])

    profile = make_column_profile(
        "quantity",
        "numeric",
    )

    quality = make_column_quality(
        "quantity",
        0.50,
    )

    prepared = prepare_numeric_series(
        series,
        profile,
        quality,
    )

    assert prepared is None


def test_prepare_numeric_series_returns_none_for_non_numeric_role():
    series = pd.Series(["10", "20", "30"])

    profile = make_column_profile(
        "product",
        "categorical",
    )

    quality = make_column_quality(
        "product",
        None,
    )

    prepared = prepare_numeric_series(
        series,
        profile,
        quality,
    )

    assert prepared is None
