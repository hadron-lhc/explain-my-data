import pytest

from src.planner import (
    AnalysisCandidate,
    plan_categorical_numeric_analysis,
)
from src.relationships import CategoricalNumericRelationship, NumericRelationship


from src.planner import (
    recommend_categorical_numeric_metrics,
    select_categorical_numeric_relationships,
    recommend_categorical_numeric_visualizations,
    select_numeric_numeric_relationships,
    plan_numeric_numeric_analysis,
    recommend_numeric_numeric_metrics,
    recommend_numeric_numeric_visualizations,
)


def test_plan_categorical_numeric_analysis_creates_candidate():
    relationship = CategoricalNumericRelationship(
        column_categorical="product",
        column_numeric="price",
        group_means={
            "laptop": 1000.0,
            "mouse": 25.0,
            "keyboard": 80.0,
        },
        group_medians={
            "laptop": 1000.0,
            "mouse": 25.0,
            "keyboard": 80.0,
        },
        group_counts={
            "laptop": 50,
            "mouse": 40,
            "keyboard": 30,
        },
        n_observations=120,
        group_separation=2.52,
        n_groups=3,
        group_outliers={
            "laptop": 5,
            "mouse": 2,
            "keyboard": 3,
        },
    )

    candidate = plan_categorical_numeric_analysis(relationship)

    assert isinstance(candidate, AnalysisCandidate)

    assert candidate.column_x == "product"
    assert candidate.column_y == "price"
    assert candidate.relationship_type == "categorical_numeric"

    assert candidate.relevance_score == pytest.approx(2.52)

    assert candidate.recommended_metrics == [
        "group_count",
        "group_mean",
        "group_median",
        "group_outliers",
    ]

    assert candidate.recommended_visualizations == ["bar", "boxplot"]


def make_relationship(
    categorical: str,
    numeric: str,
    separation: float,
) -> CategoricalNumericRelationship:
    return CategoricalNumericRelationship(
        column_categorical=categorical,
        column_numeric=numeric,
        group_means={"A": 10.0, "B": 20.0},
        group_medians={"A": 10.0, "B": 20.0},
        group_counts={"A": 10, "B": 10},
        n_observations=20,
        group_separation=separation,
        n_groups=2,
        group_outliers={"A": 0, "B": 0},
    )


def test_select_categorical_numeric_relationships_selects_high_scores():
    relationships = [
        make_relationship("A", "x", 0.1),
        make_relationship("B", "x", 0.2),
        make_relationship("C", "x", 0.3),
        make_relationship("D", "x", 0.4),
        make_relationship("E", "x", 2.0),
    ]

    selected = select_categorical_numeric_relationships(
        relationships,
        percentile=75,
    )

    assert len(selected) == 2
    assert selected[0].group_separation == pytest.approx(0.4)
    assert selected[1].group_separation == pytest.approx(2.0)


def test_select_categorical_numeric_relationships_returns_empty_for_empty_input():
    selected = select_categorical_numeric_relationships([])

    assert selected == []


def test_select_categorical_numeric_relationships_keeps_equal_scores():
    relationships = [
        make_relationship("A", "x", 1.0),
        make_relationship("B", "x", 1.0),
        make_relationship("C", "x", 1.0),
    ]

    selected = select_categorical_numeric_relationships(
        relationships,
        percentile=75,
    )

    assert len(selected) == 3


def test_planner_recommends_boxplot_when_outliers_exist():
    relationship = CategoricalNumericRelationship(
        column_categorical="product",
        column_numeric="price",
        group_means={
            "laptop": 1000.0,
            "mouse": 25.0,
        },
        group_medians={
            "laptop": 950.0,
            "mouse": 25.0,
        },
        group_counts={
            "laptop": 50,
            "mouse": 40,
        },
        n_observations=90,
        group_separation=2.52,
        n_groups=2,
        group_outliers={
            "laptop": 3,
            "mouse": 0,
        },
    )

    candidate = plan_categorical_numeric_analysis(relationship)

    assert candidate.recommended_visualizations == [
        "bar",
        "boxplot",
    ]


def test_recommend_metrics_without_outliers():
    relationship = CategoricalNumericRelationship(
        column_categorical="payment_method",
        column_numeric="discount",
        group_means={
            "cash": 10.0,
            "card": 11.0,
        },
        group_medians={
            "cash": 10.0,
            "card": 11.0,
        },
        group_counts={
            "cash": 50,
            "card": 40,
        },
        group_outliers={
            "cash": 0,
            "card": 0,
        },
        n_observations=90,
        group_separation=0.10,
        n_groups=2,
    )

    metrics = recommend_categorical_numeric_metrics(relationship)

    assert metrics == [
        "group_count",
        "group_mean",
    ]


def test_recommend_metrics_with_outliers():
    relationship = CategoricalNumericRelationship(
        column_categorical="product",
        column_numeric="price",
        group_means={
            "laptop": 1000.0,
            "mouse": 25.0,
        },
        group_medians={
            "laptop": 950.0,
            "mouse": 25.0,
        },
        group_counts={
            "laptop": 50,
            "mouse": 40,
        },
        group_outliers={
            "laptop": 3,
            "mouse": 0,
        },
        n_observations=90,
        group_separation=2.52,
        n_groups=2,
    )

    metrics = recommend_categorical_numeric_metrics(relationship)

    assert metrics == [
        "group_count",
        "group_mean",
        "group_median",
        "group_outliers",
    ]


def test_recommend_bar_for_few_groups_without_outliers():
    relationship = make_relationship("product", "price", 2.0)
    relationship.n_groups = 3
    relationship.group_outliers = {
        "A": 0,
        "B": 0,
        "C": 0,
    }

    visualizations = recommend_categorical_numeric_visualizations(relationship)

    assert visualizations == ["bar"]


def test_recommend_bar_and_boxplot_for_few_groups_with_outliers():
    relationship = make_relationship("product", "price", 2.0)
    relationship.n_groups = 3
    relationship.group_outliers = {
        "A": 2,
        "B": 0,
        "C": 0,
    }

    visualizations = recommend_categorical_numeric_visualizations(relationship)

    assert visualizations == ["bar", "boxplot"]


def test_recommend_boxplot_for_many_groups():
    relationship = make_relationship("product", "price", 2.0)
    relationship.n_groups = 6
    relationship.group_outliers = {
        "A": 0,
        "B": 0,
        "C": 0,
        "D": 0,
        "E": 0,
        "F": 0,
    }

    visualizations = recommend_categorical_numeric_visualizations(relationship)

    assert visualizations == ["boxplot"]


def test_recommend_only_boxplot_for_many_groups_with_outliers():
    relationship = make_relationship("product", "price", 2.0)
    relationship.n_groups = 6
    relationship.group_outliers = {
        "A": 2,
        "B": 0,
        "C": 1,
        "D": 0,
        "E": 0,
        "F": 0,
    }

    visualizations = recommend_categorical_numeric_visualizations(relationship)

    assert visualizations == ["boxplot"]


def make_numeric_relationship(
    column_x: str,
    column_y: str,
    pearson: float,
    spearman: float,
) -> NumericRelationship:
    return NumericRelationship(
        column_x=column_x,
        column_y=column_y,
        pearson=pearson,
        spearman=spearman,
        n_observations=100,
    )


def test_select_numeric_numeric_relationships():
    relationships = [
        make_numeric_relationship("a", "b", 0.2, 0.3),
        make_numeric_relationship("a", "c", 0.5, 0.6),
        make_numeric_relationship("a", "d", 0.9, 0.8),
        make_numeric_relationship("a", "e", -0.95, -0.9),
    ]

    selected = select_numeric_numeric_relationships(
        relationships,
        percentile=75.0,
    )

    assert len(selected) == 1
    assert selected[0].column_y == "e"


def test_numeric_numeric_selection_uses_absolute_correlation():
    relationships = [
        make_numeric_relationship("a", "b", 0.2, 0.3),
        make_numeric_relationship("a", "c", -0.9, -0.8),
    ]

    selected = select_numeric_numeric_relationships(
        relationships,
        percentile=75.0,
    )

    assert len(selected) == 1
    assert selected[0].column_y == "c"


def test_numeric_numeric_selection_uses_strongest_of_pearson_and_spearman():
    relationships = [
        make_numeric_relationship("a", "b", 0.2, 0.3),
        make_numeric_relationship("a", "c", 0.4, 0.9),
    ]

    selected = select_numeric_numeric_relationships(
        relationships,
        percentile=75.0,
    )

    assert len(selected) == 1
    assert selected[0].column_y == "c"


def test_numeric_numeric_selection_empty():
    selected = select_numeric_numeric_relationships([])

    assert selected == []


def test_select_numeric_numeric_relationships_keeps_high_relative_scores():
    relationships = [
        make_numeric_relationship("a", "b", 0.1, 0.2),
        make_numeric_relationship("a", "c", 0.3, 0.4),
        make_numeric_relationship("a", "d", 0.5, 0.6),
        make_numeric_relationship("a", "e", 0.9, 0.8),
    ]

    selected = select_numeric_numeric_relationships(
        relationships,
        percentile=75.0,
    )

    selected_names = {relationship.column_y for relationship in selected}

    assert "e" in selected_names


def test_numeric_numeric_metrics_recommend_pearson_and_spearman():
    relationship = make_numeric_relationship(
        "quantity",
        "price",
        pearson=0.8,
        spearman=0.85,
    )

    metrics = recommend_numeric_numeric_metrics(relationship)

    assert metrics == ["pearson", "spearman"]


def test_numeric_numeric_metrics_recommend_only_spearman():
    relationship = make_numeric_relationship(
        "quantity",
        "price",
        pearson=0.3,
        spearman=0.8,
    )

    metrics = recommend_numeric_numeric_metrics(relationship)

    assert metrics == ["spearman"]


def test_numeric_numeric_metrics_recommend_only_pearson():
    relationship = make_numeric_relationship(
        "quantity",
        "price",
        pearson=0.8,
        spearman=0.3,
    )

    metrics = recommend_numeric_numeric_metrics(relationship)

    assert metrics == ["pearson"]


def test_numeric_numeric_metrics_ignore_direction():
    relationship = make_numeric_relationship(
        "quantity",
        "price",
        pearson=-0.8,
        spearman=-0.9,
    )

    metrics = recommend_numeric_numeric_metrics(relationship)

    assert metrics == ["pearson", "spearman"]


def test_numeric_numeric_visualizations_recommend_scatter():
    relationship = make_numeric_relationship(
        "quantity",
        "price",
        pearson=0.8,
        spearman=0.85,
    )

    visualizations = recommend_numeric_numeric_visualizations(relationship)

    assert visualizations == ["scatter"]


def test_plan_numeric_numeric_analysis():
    relationship = make_numeric_relationship(
        "quantity",
        "price",
        pearson=0.8,
        spearman=0.85,
    )

    candidate = plan_numeric_numeric_analysis(relationship)

    assert candidate.column_x == "quantity"
    assert candidate.column_y == "price"
    assert candidate.relationship_type == "numeric_numeric"
    assert candidate.relevance_score == 0.85
    assert candidate.recommended_metrics == [
        "pearson",
        "spearman",
    ]
    assert candidate.recommended_visualizations == [
        "scatter",
    ]


def test_numeric_numeric_selection_requires_minimum_strength():
    relationships = [
        make_numeric_relationship("a", "b", 0.1, 0.2),
        make_numeric_relationship("a", "c", 0.3, 0.4),
        make_numeric_relationship("a", "d", 0.4, 0.45),
        make_numeric_relationship("a", "e", 0.9, 0.85),
    ]

    selected = select_numeric_numeric_relationships(
        relationships,
        percentile=75.0,
        minimum_strength=0.5,
    )

    selected_names = {relationship.column_y for relationship in selected}

    assert selected_names == {"e"}


def test_numeric_numeric_selection_uses_relative_threshold():
    relationships = [
        make_numeric_relationship("a", "b", 0.1, 0.2),
        make_numeric_relationship("a", "c", 0.4, 0.6),
        make_numeric_relationship("a", "d", 0.7, 0.8),
        make_numeric_relationship("a", "e", 0.9, 0.95),
    ]

    selected = select_numeric_numeric_relationships(
        relationships,
        percentile=50.0,
        minimum_strength=0.5,
    )

    selected_names = {relationship.column_y for relationship in selected}

    assert selected_names == {"d", "e"}
