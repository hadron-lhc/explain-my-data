from src.insights import generate_relationship_insights
from src.relationships import (
    CategoricalNumericRelationship,
    NumericRelationship,
)


def test_generates_strong_numeric_relationship_insight():
    relationship = NumericRelationship(
        column_x="quantity",
        column_y="total_sales",
        pearson=0.82,
        spearman=0.79,
        n_observations=450,
    )

    insights = generate_relationship_insights(
        numeric_relationships=[relationship],
        categorical_numeric_relationships=[],
    )

    assert len(insights) == 1
    assert insights[0].title == ("Strong association between quantity and total_sales")
    assert "positive association" in insights[0].message
    assert "Pearson: 0.82" in insights[0].message
    assert insights[0].severity == "info"


def test_ignores_weak_numeric_relationship():
    relationship = NumericRelationship(
        column_x="age",
        column_y="discount",
        pearson=0.21,
        spearman=0.18,
        n_observations=450,
    )

    insights = generate_relationship_insights(
        numeric_relationships=[relationship],
        categorical_numeric_relationships=[],
    )

    assert insights == []


def test_generates_categorical_numeric_relationship_insight():
    relationship = CategoricalNumericRelationship(
        column_categorical="product",
        column_numeric="price",
        group_means={
            "laptop": 1200.0,
            "phone": 700.0,
            "mouse": 25.0,
        },
        group_medians={
            "laptop": 1150.0,
            "phone": 680.0,
            "mouse": 20.0,
        },
        group_counts={
            "laptop": 100,
            "phone": 150,
            "mouse": 200,
        },
        n_observations=450,
        group_separation=2.52,
        n_groups=3,
        group_outliers={
            "laptop": 2,
            "phone": 1,
            "mouse": 0,
        },
    )

    insights = generate_relationship_insights(
        numeric_relationships=[],
        categorical_numeric_relationships=[relationship],
    )

    assert len(insights) == 1
    assert insights[0].title == ("price differs across product groups")
    assert "varies substantially" in insights[0].message
    assert "2.52" in insights[0].message


def test_ignores_weak_categorical_numeric_relationship():
    relationship = CategoricalNumericRelationship(
        column_categorical="category",
        column_numeric="discount",
        group_means={
            "a": 10.0,
            "b": 10.5,
        },
        group_medians={
            "a": 10.0,
            "b": 10.5,
        },
        group_counts={
            "a": 100,
            "b": 100,
        },
        n_observations=200,
        group_separation=0.05,
        n_groups=2,
        group_outliers={
            "a": 0,
            "b": 0,
        },
    )

    insights = generate_relationship_insights(
        numeric_relationships=[],
        categorical_numeric_relationships=[relationship],
    )

    assert insights == []
