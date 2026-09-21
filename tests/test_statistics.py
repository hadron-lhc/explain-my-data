import pandas as pd
import pytest

from src.relationships import calculate_numeric_relationship


def test_numeric_relationship_detects_perfect_positive_relationship():
    x = pd.Series([1, 2, 3, 4, 5])
    y = pd.Series([2, 4, 6, 8, 10])

    result = calculate_numeric_relationship(
        x,
        y,
        "x",
        "y",
    )

    assert result is not None
    assert result.pearson == pytest.approx(1.0)
    assert result.spearman == pytest.approx(1.0)
    assert result.n_observations == 5


def test_numeric_relationship_detects_perfect_negative_relationship():
    x = pd.Series([1, 2, 3, 4, 5])
    y = pd.Series([10, 8, 6, 4, 2])

    result = calculate_numeric_relationship(
        x,
        y,
        "x",
        "y",
    )

    assert result is not None
    assert result.pearson == pytest.approx(-1.0)
    assert result.spearman == pytest.approx(-1.0)


def test_numeric_relationship_ignores_missing_values():
    x = pd.Series([1, 2, None, 4, 5])
    y = pd.Series([2, 4, 6, None, 10])

    result = calculate_numeric_relationship(
        x,
        y,
        "x",
        "y",
    )

    assert result is not None
    assert result.n_observations == 3


def test_numeric_relationship_handles_numeric_strings():
    x = pd.Series(["1", "2", "3", "4"])
    y = pd.Series(["2", "4", "6", "8"])

    result = calculate_numeric_relationship(
        x,
        y,
        "x",
        "y",
    )

    assert result is not None
    assert result.pearson == pytest.approx(1.0)


def test_numeric_relationship_returns_none_with_insufficient_data():
    x = pd.Series([1])
    y = pd.Series([2])

    result = calculate_numeric_relationship(
        x,
        y,
        "x",
        "y",
    )

    assert result is None
