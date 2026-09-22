import pytest
import pandas as pd

from src.relationships import (
    calculate_categorical_numeric_relationship,
)


def test_group_separation_measures_difference_between_group_means():
    numeric = pd.Series([10, 10, 12, 12, 20, 20, 22, 22])

    categorical = pd.Series(["A", "A", "A", "A", "B", "B", "B", "B"])

    relationship = calculate_categorical_numeric_relationship(
        numeric_series=numeric,
        categorical_series=categorical,
        column_numeric="value",
        column_categorical="group",
    )

    assert relationship is not None

    expected_mean_range = 21 - 11
    expected_std = numeric.std()
    expected_separation = expected_mean_range / expected_std

    assert relationship.group_separation == pytest.approx(expected_separation)


def test_group_separation_is_zero_when_group_means_are_equal():
    numeric = pd.Series([10, 20, 10, 20])

    categorical = pd.Series(["A", "A", "B", "B"])

    relationship = calculate_categorical_numeric_relationship(
        numeric_series=numeric,
        categorical_series=categorical,
        column_numeric="value",
        column_categorical="group",
    )

    assert relationship is not None
    assert relationship.group_separation == pytest.approx(0.0)


def test_group_separation_handles_constant_numeric_values():
    numeric = pd.Series([10, 10, 10, 10])

    categorical = pd.Series(["A", "A", "B", "B"])

    relationship = calculate_categorical_numeric_relationship(
        numeric_series=numeric,
        categorical_series=categorical,
        column_numeric="value",
        column_categorical="group",
    )

    assert relationship is not None
    assert relationship.group_separation == pytest.approx(0.0)


def test_group_separation_ignores_missing_values():
    numeric = pd.Series([10, 10, 20, 20, None])

    categorical = pd.Series(["A", "A", "B", "B", "B"])

    relationship = calculate_categorical_numeric_relationship(
        numeric_series=numeric,
        categorical_series=categorical,
        column_numeric="value",
        column_categorical="group",
    )

    assert relationship is not None

    assert relationship.n_observations == 4
    assert relationship.group_means["A"] == pytest.approx(10.0)
    assert relationship.group_means["B"] == pytest.approx(20.0)
