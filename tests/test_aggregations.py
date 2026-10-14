import pandas as pd
import pytest

from src.aggregations import (
    calculate_categorical_numeric_aggregation,
    generate_categorical_numeric_aggregations,
    select_categorical_numeric_pairs,
)
from src.profiling import profile_dataset


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "city": ["A", "A", "A", "B", "B", "B"],
            "revenue": [10, 20, 30, 40, 40, 60],
            "customer_id": [1, 2, 3, 4, 5, 6],
        }
    )


def test_calculates_group_counts_means_and_medians(sample_df):
    result = calculate_categorical_numeric_aggregation(sample_df, "city", "revenue")

    assert result is not None
    assert result.column_categorical == "city"
    assert result.column_numeric == "revenue"
    assert result.n_observations == 6

    groups = {group.group_value: group for group in result.groups}

    assert groups["A"].count == 3
    assert groups["A"].mean == pytest.approx(20)
    assert groups["A"].median == pytest.approx(20)

    assert groups["B"].count == 3
    assert groups["B"].mean == pytest.approx(140 / 3)
    assert groups["B"].median == pytest.approx(40)


def test_calculates_totals_and_shares_when_additive(sample_df):
    result = calculate_categorical_numeric_aggregation(
        sample_df, "city", "revenue", additive=True
    )

    assert result is not None

    groups = {group.group_value: group for group in result.groups}

    assert groups["A"].total == pytest.approx(60)
    assert groups["B"].total == pytest.approx(140)

    assert groups["A"].share_pct == pytest.approx(30)
    assert groups["B"].share_pct == pytest.approx(70)

    assert sum(group.share_pct for group in result.groups) == pytest.approx(100)


def test_does_not_calculate_totals_by_default(sample_df):
    result = calculate_categorical_numeric_aggregation(sample_df, "city", "revenue")

    assert result is not None
    assert all(group.total is None for group in result.groups)
    assert all(group.share_pct is None for group in result.groups)


def test_returns_none_when_there_is_only_one_group():
    df = pd.DataFrame(
        {
            "city": ["A", "A", "A"],
            "revenue": [10, 20, 30],
        }
    )

    result = calculate_categorical_numeric_aggregation(df, "city", "revenue")

    assert result is None


@pytest.mark.parametrize(
    ("categorical_column", "numeric_column"),
    [
        ("missing_city", "revenue"),
        ("city", "missing_revenue"),
    ],
)
def test_returns_none_when_a_column_does_not_exist(
    sample_df, categorical_column, numeric_column
):
    result = calculate_categorical_numeric_aggregation(
        sample_df, categorical_column, numeric_column
    )

    assert result is None


def test_selects_categorical_numeric_pairs(sample_df):
    profile = profile_dataset(sample_df)

    pairs = select_categorical_numeric_pairs(profile.columns)

    assert ("city", "revenue") in pairs
    assert ("city", "customer_id") not in pairs


def test_does_not_select_categories_above_maximum_cardinality():
    df = pd.DataFrame(
        {
            "category": [f"category_{i}" for i in range(25)],
            "revenue": list(range(25)),
        }
    )
    profile = profile_dataset(df)

    pairs = select_categorical_numeric_pairs(profile.columns, max_categories=20)

    assert ("category", "revenue") not in pairs


def test_generates_aggregations_for_selected_pairs(sample_df):
    profile = profile_dataset(sample_df)

    results = generate_categorical_numeric_aggregations(
        sample_df,
        profile.columns,
        additive_columns={"revenue"},
    )

    assert len(results) == 1

    result = results[0]
    assert result.column_categorical == "city"
    assert result.column_numeric == "revenue"
    assert result.n_observations == 6

    groups = {group.group_value: group for group in result.groups}
    assert groups["A"].total == pytest.approx(60)
    assert groups["B"].total == pytest.approx(140)


def test_aggregation_reports_coverage():
    df = pd.DataFrame(
        {
            "category": ["a"] * 20 + ["b"] * 20 + [None] * 160,
            "value": [100.0] * 40 + [10.0] * 160,
        }
    )

    result = calculate_categorical_numeric_aggregation(df, "category", "value")

    assert result is not None
    assert result.total_rows == 200
    assert result.n_observations == 40
    assert result.coverage_pct == pytest.approx(20.0)


def test_aggregation_full_coverage_is_100_pct():
    df = pd.DataFrame(
        {
            "category": ["a"] * 20 + ["b"] * 20,
            "value": [100.0] * 20 + [10.0] * 20,
        }
    )

    result = calculate_categorical_numeric_aggregation(df, "category", "value")

    assert result is not None
    assert result.total_rows == 40
    assert result.coverage_pct == pytest.approx(100.0)
