import pandas as pd

from src.relationships import (
    CategoricalNumericRelationship,
    calculate_categorical_numeric_relationship,
)


def test_categorical_numeric_relationship():
    numeric = pd.Series([10, 20, 30, 40, 50])
    categorical = pd.Series(["A", "A", "B", "B", "B"])

    result = calculate_categorical_numeric_relationship(
        numeric,
        categorical,
        "sales",
        "category",
    )

    assert isinstance(result, CategoricalNumericRelationship)

    assert result.column_numeric == "sales"
    assert result.column_categorical == "category"

    assert result.group_means == {
        "A": 15.0,
        "B": 40.0,
    }

    assert result.group_medians == {
        "A": 15.0,
        "B": 40.0,
    }

    assert result.group_counts == {
        "A": 2,
        "B": 3,
    }

    assert result.n_observations == 5


def test_categorical_numeric_relationship_ignores_missing_values():
    numeric = pd.Series([10, 20, None, 40, 50])
    categorical = pd.Series(["A", "A", "B", None, "B"])

    result = calculate_categorical_numeric_relationship(
        numeric,
        categorical,
        "sales",
        "category",
    )

    assert result is not None

    assert result.group_means == {
        "A": 15.0,
        "B": 50.0,
    }

    assert result.group_medians == {
        "A": 15.0,
        "B": 50.0,
    }

    assert result.group_counts == {
        "A": 2,
        "B": 1,
    }

    assert result.n_observations == 3


def test_categorical_numeric_relationship_converts_numeric_strings():
    numeric = pd.Series(["10", "20", "30", "40"])
    categorical = pd.Series(["A", "A", "B", "B"])

    result = calculate_categorical_numeric_relationship(
        numeric,
        categorical,
        "sales",
        "category",
    )

    assert result is not None

    assert result.group_means == {
        "A": 15.0,
        "B": 35.0,
    }


def test_categorical_numeric_relationship_returns_none_with_insufficient_data():
    numeric = pd.Series([10])
    categorical = pd.Series(["A"])

    result = calculate_categorical_numeric_relationship(
        numeric,
        categorical,
        "sales",
        "category",
    )

    assert result is None
