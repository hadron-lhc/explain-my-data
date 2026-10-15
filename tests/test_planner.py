import pytest

from src.planner import (
    AnalysisCandidate,
    plan_categorical_numeric_analysis,
    plan_numeric_numeric_analysis,
    plan_numeric_numeric_visualization,
    recommend_categorical_numeric_metrics,
    recommend_categorical_numeric_visualizations,
    recommend_numeric_numeric_metrics,
    recommend_numeric_numeric_visualizations,
    select_categorical_categorical_relationships,
    select_categorical_numeric_relationships,
    select_numeric_numeric_relationships,
)
from src.profiling import ColumnProfile
from src.relationships import (
    CategoricalCategoricalRelationship,
    CategoricalNumericRelationship,
    NumericRelationship,
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

    # Percentil p75 deja D y E; el override top-1 por columna añade
    # C (0.3 >= MIN_GROUP_SEPARATION 0.25). A y B quedan fuera
    # por debajo del piso absoluto.
    assert len(selected) == 3
    assert selected[0].group_separation == pytest.approx(0.3)
    assert selected[1].group_separation == pytest.approx(0.4)
    assert selected[2].group_separation == pytest.approx(2.0)


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


def make_numeric_profile(
    name: str,
    *,
    is_discrete: bool,
) -> ColumnProfile:
    return ColumnProfile(
        name=name,
        dtype="int64" if is_discrete else "float64",
        role="numeric",
        n_missing=0,
        pct_missing=0.0,
        n_unique=5 if is_discrete else 100,
        pct_unique=5.0 if is_discrete else 100.0,
        sample_values=[1, 2, 3],
        is_discrete=is_discrete,
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

    # Percentil deja solo e; los overrides añaden d (strong floor 0.9)
    # y c (top-1 de la columna c, 0.6 >= piso 0.5). b queda fuera.
    selected_names = {relationship.column_y for relationship in selected}

    assert selected_names == {"c", "d", "e"}
    assert [relationship.column_y for relationship in selected] == [
        "c",
        "d",
        "e",
    ]


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

    visualizations = recommend_numeric_numeric_visualizations(
        relationship,
    )

    assert visualizations == ["scatter"]


def test_plan_numeric_numeric_analysis():
    relationship = make_numeric_relationship(
        "quantity",
        "price",
        pearson=0.8,
        spearman=0.85,
    )

    profile_x = make_numeric_profile(
        "quantity",
        is_discrete=True,
    )

    profile_y = make_numeric_profile(
        "price",
        is_discrete=False,
    )

    candidate = plan_numeric_numeric_analysis(
        relationship,
        profile_x,
        profile_y,
    )

    assert candidate.column_x == "quantity"
    assert candidate.column_y == "price"
    assert candidate.relationship_type == "numeric_numeric"
    assert candidate.relevance_score == pytest.approx(0.85)


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

    # Percentil 50 deja d y e; el top-1 por columna añade c
    # (0.6 >= piso 0.5). b queda fuera por debajo del piso.
    selected_names = {relationship.column_y for relationship in selected}

    assert selected_names == {"c", "d", "e"}


def test_numeric_numeric_visualization_plan_depends_on_sample_size():
    small = NumericRelationship(
        column_x="x",
        column_y="y",
        pearson=0.7,
        spearman=0.7,
        n_observations=500,
    )

    medium = NumericRelationship(
        column_x="x",
        column_y="y",
        pearson=0.7,
        spearman=0.7,
        n_observations=3000,
    )

    large = NumericRelationship(
        column_x="x",
        column_y="y",
        pearson=0.7,
        spearman=0.7,
        n_observations=10000,
    )

    profile_x = make_numeric_profile(
        "x",
        is_discrete=False,
    )

    profile_y = make_numeric_profile(
        "y",
        is_discrete=False,
    )

    small_plan = plan_numeric_numeric_visualization(
        small,
        profile_x,
        profile_y,
    )

    medium_plan = plan_numeric_numeric_visualization(
        medium,
        profile_x,
        profile_y,
    )

    large_plan = plan_numeric_numeric_visualization(
        large,
        profile_x,
        profile_y,
    )

    assert small_plan.chart_type == "scatter"
    assert small_plan.sampling is False
    assert small_plan.trend_line is True

    assert medium_plan.chart_type == "scatter"
    assert medium_plan.sampling is True
    assert medium_plan.max_points == 1500
    assert medium_plan.trend_line is True

    assert large_plan.chart_type == "scatter"
    assert large_plan.sampling is True
    assert large_plan.max_points == 2000
    assert large_plan.trend_line is True


def test_numeric_numeric_visualization_uses_scatter_for_continuous_variables():
    relationship = NumericRelationship(
        column_x="area",
        column_y="price",
        pearson=0.7,
        spearman=0.7,
        n_observations=1000,
    )

    profile_x = make_numeric_profile(
        "area",
        is_discrete=False,
    )

    profile_y = make_numeric_profile(
        "price",
        is_discrete=False,
    )

    plan = plan_numeric_numeric_visualization(
        relationship,
        profile_x,
        profile_y,
    )

    assert plan.chart_type == "scatter"


def test_numeric_numeric_visualization_uses_ranked_bar_for_low_cardinality_discrete_x():
    relationship = NumericRelationship(
        column_x="bathrooms",
        column_y="area",
        pearson=0.6,
        spearman=0.7,
        n_observations=1000,
    )

    profile_x = make_numeric_profile(
        "bathrooms",
        is_discrete=True,
    )

    profile_y = make_numeric_profile(
        "area",
        is_discrete=False,
    )

    plan = plan_numeric_numeric_visualization(
        relationship,
        profile_x,
        profile_y,
    )

    assert plan.chart_type == "ranked_bar"
    assert plan.group_by == "bathrooms"


def test_numeric_numeric_visualization_uses_ranked_bar_for_low_cardinality_discrete_y():
    relationship = NumericRelationship(
        column_x="area",
        column_y="bathrooms",
        pearson=0.6,
        spearman=0.7,
        n_observations=1000,
    )

    profile_x = make_numeric_profile(
        "area",
        is_discrete=False,
    )

    profile_y = make_numeric_profile(
        "bathrooms",
        is_discrete=True,
    )

    plan = plan_numeric_numeric_visualization(
        relationship,
        profile_x,
        profile_y,
    )

    assert plan.chart_type == "ranked_bar"
    assert plan.group_by == "bathrooms"


def make_categorical_relationship(
    column_x: str,
    column_y: str,
    cramers_v: float,
) -> CategoricalCategoricalRelationship:
    return CategoricalCategoricalRelationship(
        column_x=column_x,
        column_y=column_y,
        n_observations=100,
        n_groups_x=3,
        n_groups_y=2,
        chi2=10.0,
        p_value=0.0,
        cramers_v=cramers_v,
    )


# ---------------------------------------------------------------------------
# Regresiones QA (playground explain-my-data-qa): los selectores deben
# recuperar relaciones meaningfuls que el percentil solo echaba, sin
# relajar los pisos absolutos.
# ---------------------------------------------------------------------------


def test_numnum_strong_pair_survives_high_percentile():
    # diamonds: los pares geométricos x/y/z/carat (~0.97-0.99) inflan
    # el p75 (0.98); price~carat (0.963) queda fuera del percentil y
    # tampoco es top-1 de price (price~y 0.97 es mejor) → solo el
    # strong floor (0.9) lo recupera.
    relationships = [
        make_numeric_relationship("price", "carat", 0.963, 0.961),
        make_numeric_relationship("price", "y", 0.955, 0.970),
        make_numeric_relationship("x", "y", 0.974, 0.982),
        make_numeric_relationship("x", "z", 0.966, 0.975),
        make_numeric_relationship("y", "z", 0.978, 0.989),
        make_numeric_relationship("carat", "x", 0.980, 0.985),
        make_numeric_relationship("carat", "y", 0.951, 0.967),
        make_numeric_relationship("carat", "z", 0.948, 0.961),
        make_numeric_relationship("price", "depth", -0.011, 0.010),
        make_numeric_relationship("price", "table", 0.120, 0.127),
    ]

    selected = select_numeric_numeric_relationships(relationships)

    selected_pairs = {
        (relationship.column_x, relationship.column_y)
        for relationship in selected
    }

    assert ("price", "carat") in selected_pairs
    assert ("price", "depth") not in selected_pairs
    assert ("price", "table") not in selected_pairs


def test_catnum_group_top1_survives_percentile():
    # titanic: parejas redundantes (who/alive/sex/adult_male ~
    # survived, separations 1.8-2.1) inflan el p75 (2.01);
    # survived×pclass (0.696) queda fuera del percentil pero es el
    # top-1 del lado categórico pclass y supera MIN_GROUP_SEPARATION
    # (0.25).
    relationships = [
        make_relationship("who", "survived", 2.10),
        make_relationship("alive", "survived", 2.05),
        make_relationship("sex", "survived", 1.90),
        make_relationship("adult_male", "survived", 1.80),
        make_relationship("pclass", "survived", 0.696),
        make_relationship("embarked", "pclass", 1.22),
    ]

    selected = select_categorical_numeric_relationships(relationships)

    selected_pairs = {
        (relationship.column_categorical, relationship.column_numeric)
        for relationship in selected
    }

    assert ("pclass", "survived") in selected_pairs
    assert ("embarked", "pclass") in selected_pairs


def test_numnum_top1_respects_absolute_floor():
    # tips: tip~total_bill (0.678) entra por percentil;
    # total_bill~size (0.595) entra como top-1 de size;
    # tip~size (0.489) NO entra: no es top-1 de ninguna columna y
    # queda por debajo de MIN_CORRELATION_STRENGTH (0.5).
    relationships = [
        make_numeric_relationship("total_bill", "tip", 0.676, 0.678),
        make_numeric_relationship("total_bill", "size", 0.569, 0.595),
        make_numeric_relationship("tip", "size", 0.452, 0.489),
    ]

    selected = select_numeric_numeric_relationships(relationships)

    selected_pairs = {
        (relationship.column_x, relationship.column_y)
        for relationship in selected
    }

    assert ("total_bill", "tip") in selected_pairs
    assert ("total_bill", "size") in selected_pairs
    assert ("tip", "size") not in selected_pairs


def test_catcat_top1_respects_cramers_floor():
    # adult: education~income (V 0.366) entra como top-1 de education
    # (supera MIN_CRAMERS_V 0.3); sex~income (V 0.215) NO entra: no es
    # top-1 de ninguna columna (sex su mejor par es relationship~sex
    # 0.65; income es relationship~income 0.45) y el piso la deja
    # fuera.
    relationships = [
        make_categorical_relationship("relationship", "sex", 0.65),
        make_categorical_relationship("marital_status", "relationship", 0.49),
        make_categorical_relationship("marital_status", "sex", 0.46),
        make_categorical_relationship("marital_status", "income", 0.45),
        make_categorical_relationship("relationship", "income", 0.45),
        make_categorical_relationship("occupation", "sex", 0.42),
        make_categorical_relationship("workclass", "occupation", 0.40),
        make_categorical_relationship("education", "income", 0.366),
        make_categorical_relationship("sex", "income", 0.215),
    ]

    selected = select_categorical_categorical_relationships(relationships)

    selected_pairs = {
        (relationship.column_x, relationship.column_y)
        for relationship in selected
    }

    assert ("education", "income") in selected_pairs
    assert ("marital_status", "income") in selected_pairs
    assert ("sex", "income") not in selected_pairs
