import numpy as np
import pandas as pd

from src.relationships import (
    CategoricalCategoricalRelationship,
    CategoricalNumericRelationship,
    calculate_categorical_categorical_relationship,
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


# ---------------------------------------------------------------------------
# Categorical ↔ categorical
# ---------------------------------------------------------------------------


def test_categorical_categorical_detects_strong_association():
    rng = np.random.default_rng(42)

    # color depends on size: S→red, L→blue
    sizes = rng.choice(["S", "L"], size=200)
    colors = np.where(sizes == "S", "red", "blue")

    x = pd.Series(sizes)
    y = pd.Series(colors)

    result = calculate_categorical_categorical_relationship(x, y, "size", "color")

    assert isinstance(result, CategoricalCategoricalRelationship)
    assert result.column_x == "size"
    assert result.column_y == "color"
    assert result.n_observations == 200
    assert result.n_groups_x == 2
    assert result.n_groups_y == 2
    assert result.cramers_v > 0.9
    assert result.p_value < 0.01
    assert result.top_associations


def test_categorical_categorical_detects_independence():
    rng = np.random.default_rng(7)

    x = pd.Series(rng.choice(["a", "b"], size=300))
    y = pd.Series(rng.choice(["x", "y"], size=300))

    result = calculate_categorical_categorical_relationship(x, y, "x1", "y1")

    assert result is not None
    assert result.cramers_v < 0.2


def test_categorical_categorical_returns_none_for_same_column():
    x = pd.Series(["a", "b"] * 50)

    assert calculate_categorical_categorical_relationship(x, x, "col", "col") is None


def test_categorical_categorical_returns_none_with_few_observations():
    x = pd.Series(["a", "b", "a"])
    y = pd.Series(["x", "y", "x"])

    assert calculate_categorical_categorical_relationship(x, y, "x1", "y1") is None


def test_categorical_categorical_returns_none_for_constant_column():
    x = pd.Series(["a"] * 100)
    y = pd.Series(["x", "y"] * 50)

    assert calculate_categorical_categorical_relationship(x, y, "x1", "y1") is None


def test_categorical_categorical_ignores_missing_values():
    rng = np.random.default_rng(1)

    sizes = rng.choice(["S", "L"], size=100)
    colors = np.where(sizes == "S", "red", "blue")

    x = pd.Series(sizes, dtype=object)
    y = pd.Series(colors, dtype=object)

    x.iloc[0] = None
    y.iloc[1] = None

    result = calculate_categorical_categorical_relationship(x, y, "size", "color")

    assert result is not None
    assert result.n_observations == 98


def test_categorical_categorical_excludes_high_cardinality():
    x = pd.Series([f"cat{i}" for i in range(100)])
    y = pd.Series([f"grp{i}" for i in range(100)])

    assert calculate_categorical_categorical_relationship(x, y, "x1", "y1") is None
