import numpy as np
import pandas as pd
import pytest

from src.patterns import (
    detect_group_differences,
    detect_numeric_outliers,
)

# ---------------------------------------------------------------------------
# Group differences
# ---------------------------------------------------------------------------


def test_detects_groups_with_different_means():
    df = pd.DataFrame(
        {
            "country": ["A"] * 10 + ["B"] * 10,
            "revenue": [
                10.0,
                11.0,
                9.0,
                10.5,
                9.5,
                10.2,
                9.8,
                10.1,
                9.9,
                10.0,
                100.0,
                101.0,
                99.0,
                100.5,
                99.5,
                100.2,
                99.8,
                100.1,
                99.9,
                100.0,
            ],
        }
    )

    patterns = detect_group_differences(
        df,
        "country",
        "revenue",
        min_group_size=5,
    )

    assert len(patterns) == 2

    by_group = {pattern.group_value: pattern for pattern in patterns}

    assert by_group["A"].group_mean == pytest.approx(10.0)
    assert by_group["B"].group_mean == pytest.approx(100.0)
    assert by_group["A"].difference < 0
    assert by_group["B"].difference > 0

    for pattern in patterns:
        assert pattern.group_size == 10
        assert abs(pattern.standardized_difference) >= 1.0


def test_returns_no_patterns_when_group_means_are_equal():
    df = pd.DataFrame(
        {
            "country": ["A"] * 5 + ["B"] * 5,
            "revenue": [10.0, 20.0, 30.0, 40.0, 50.0] * 2,
        }
    )

    patterns = detect_group_differences(df, "country", "revenue")

    assert patterns == []


def test_ignores_groups_below_minimum_size():
    df = pd.DataFrame(
        {
            "country": ["A"] * 10 + ["B"] * 2,
            "revenue": [10.0] * 10 + [1000.0] * 2,
        }
    )

    patterns = detect_group_differences(
        df,
        "country",
        "revenue",
        min_group_size=5,
    )

    assert all(pattern.group_value != "B" for pattern in patterns)


def test_returns_no_patterns_for_missing_columns():
    df = pd.DataFrame({"country": ["A", "B"], "revenue": [10, 20]})

    assert detect_group_differences(df, "missing_country", "revenue") == []

    assert detect_group_differences(df, "country", "missing_revenue") == []


def test_returns_no_patterns_when_numeric_values_are_constant():
    df = pd.DataFrame(
        {
            "country": ["A"] * 5 + ["B"] * 5,
            "revenue": [10.0] * 10,
        }
    )

    patterns = detect_group_differences(df, "country", "revenue")

    assert patterns == []


def test_ignores_missing_and_infinite_values_in_group_differences():
    df = pd.DataFrame(
        {
            "country": ["A"] * 5 + ["B"] * 5,
            "revenue": [10.0] * 5 + [100.0] * 5,
        }
    )
    df.loc[0, "revenue"] = np.nan
    df.loc[5, "revenue"] = np.inf

    patterns = detect_group_differences(df, "country", "revenue", min_group_size=4)

    by_group = {pattern.group_value: pattern for pattern in patterns}

    assert by_group["A"].group_size == 4
    assert by_group["B"].group_size == 4


def test_rejects_invalid_group_difference_parameters():
    df = pd.DataFrame({"country": ["A", "A", "B", "B"], "revenue": [1, 2, 3, 4]})

    with pytest.raises(ValueError):
        detect_group_differences(df, "country", "revenue", min_group_size=0)

    with pytest.raises(ValueError):
        detect_group_differences(
            df, "country", "revenue", min_standardized_difference=-1
        )

    with pytest.raises(ValueError):
        detect_group_differences(df, "country", "revenue", max_groups=1)


# ---------------------------------------------------------------------------
# Numeric outliers
# ---------------------------------------------------------------------------


def test_detects_iqr_outlier():
    df = pd.DataFrame({"revenue": [10, 11, 12, 13, 14, 100]})

    patterns = detect_numeric_outliers(df, ["revenue"])

    assert len(patterns) == 1

    pattern = patterns[0]

    assert pattern.column == "revenue"
    assert pattern.n_observations == 6
    assert pattern.outlier_count == 1
    assert pattern.outlier_positions == [5]
    assert pattern.outlier_values == [100.0]
    assert 100.0 > pattern.upper_bound


def test_returns_no_outliers_for_regular_values():
    df = pd.DataFrame({"revenue": [10, 11, 12, 13, 14, 15]})

    patterns = detect_numeric_outliers(df, ["revenue"])

    assert patterns == []


def test_ignores_missing_and_infinite_values_in_outliers():
    df = pd.DataFrame({"revenue": [10, 11, 12, 13, 14, 100, np.nan, np.inf]})

    patterns = detect_numeric_outliers(df, ["revenue"])

    assert len(patterns) == 1
    assert patterns[0].n_observations == 6
    assert patterns[0].outlier_values == [100.0]
    assert patterns[0].outlier_positions == [5]


def test_skips_missing_columns():
    df = pd.DataFrame({"revenue": [10, 11, 12, 13, 14, 100]})

    patterns = detect_numeric_outliers(df, ["missing_column", "revenue"])

    assert len(patterns) == 1
    assert patterns[0].column == "revenue"


def test_skips_columns_with_too_few_valid_observations():
    df = pd.DataFrame({"revenue": [10, 11, 100, np.nan]})

    patterns = detect_numeric_outliers(df, ["revenue"])

    assert patterns == []


def test_does_not_modify_original_dataframe():
    df = pd.DataFrame({"revenue": [10, 11, 12, 13, 14, 100]})
    original = df.copy(deep=True)

    detect_numeric_outliers(df, ["revenue"])

    pd.testing.assert_frame_equal(df, original)


def test_preserves_original_row_positions_with_non_default_index():
    df = pd.DataFrame(
        {"revenue": [10, 11, 12, 13, 14, 100]},
        index=[100, 200, 300, 400, 500, 600],
    )

    patterns = detect_numeric_outliers(df, ["revenue"])

    assert patterns[0].outlier_positions == [5]


def test_rejects_invalid_outlier_parameters():
    df = pd.DataFrame({"revenue": [10, 11, 12, 13, 14, 100]})

    with pytest.raises(ValueError):
        detect_numeric_outliers(df, ["revenue"], min_observations=3)

    with pytest.raises(ValueError):
        detect_numeric_outliers(df, ["revenue"], iqr_multiplier=-1)

    with pytest.raises(ValueError):
        detect_numeric_outliers(df, ["revenue"], iqr_multiplier=np.inf)
