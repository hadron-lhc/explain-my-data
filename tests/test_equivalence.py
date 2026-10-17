"""Tests de detección de columnas redundantes y derivadas (P1)."""

import numpy as np
import pandas as pd

from src.analysis import analyze_dataset
from src.equivalence import (
    DerivedColumn,
    EquivalenceMap,
    EquivalenceRelation,
    _is_step_function,
    _one_to_one,
    detect_equivalences,
    equivalence_card,
    filter_categorical_numeric_pairs,
    filter_insights,
    filter_relationships,
)
from src.insights import Insight
from src.relationships import (
    CategoricalNumericRelationship,
    calculate_categorical_categorical_relationship,
)


def _profiles(columns: dict[str, str]) -> dict[str, str]:
    return columns


def _prepared(df: pd.DataFrame) -> dict[str, pd.Series]:
    return {name: series for name, series in df.items()}


def test_detect_equivalences_cat_cat_bijective():
    df = pd.DataFrame(
        {
            "embarked": ["S", "C", "Q", "S", "C"],
            "embark_town": [
                "Southampton",
                "Cherbourg",
                "Queenstown",
                "Southampton",
                "Cherbourg",
            ],
        }
    )
    profiles = _profiles({"embarked": "categorical", "embark_town": "categorical"})

    result = detect_equivalences(profiles, _prepared(df))

    pair = {
        (relation.column_source, relation.column_duplicate)
        for relation in result.relations
    }

    # Empate de preferencia (mismos missing y cardinalidad): gana el
    # orden alfabetico determinista.
    assert pair == {("embark_town", "embarked")}
    assert result.duplicate_columns == {"embarked"}


def test_detect_equivalences_num_num_linear_copy():
    df = pd.DataFrame({"price": [10.0, 20.0, 30.0], "price_copy": [10.0, 20.0, 30.0]})
    profiles = _profiles({"price": "numeric", "price_copy": "numeric"})

    result = detect_equivalences(profiles, _prepared(df))

    assert result.duplicate_columns == {"price_copy"}
    relation = result.relations[0]
    assert relation.kind == "num_num"
    assert relation.column_source == "price"


def test_detect_equivalences_cat_num_encodes():
    # education_num codifica exactamente a education (1..16 -> etiqueta).
    labels = [f"L{i}" for i in range(16)]
    df = pd.DataFrame(
        {
            "education": labels * 2,
            "education_num": list(range(1, 17)) * 2,
        }
    )
    profiles = _profiles({"education": "categorical", "education_num": "numeric"})

    cat_num = CategoricalNumericRelationship(
        column_categorical="education",
        column_numeric="education_num",
        group_means={},
        group_medians={},
        group_counts={},
        n_observations=32,
        group_separation=1.0,
        n_groups=16,
        group_outliers={},
    )

    result = detect_equivalences(
        profiles,
        _prepared(df),
        cat_num=[cat_num],
    )

    assert result.duplicate_columns == {"education"}
    relation = result.relations[0]
    assert relation.kind == "cat_num"
    assert relation.column_source == "education_num"
    assert relation.label == "encodes the same information as"


def test_detect_equivalences_prefers_fewer_missing():
    df = pd.DataFrame(
        {
            "a": ["x", "y", None, None],
            "b": ["x", "y", "x", "y"],
        }
    )
    profiles = _profiles({"a": "categorical", "b": "categorical"})

    result = detect_equivalences(profiles, _prepared(df))

    relation = result.relations[0]
    assert relation.column_source == "b"
    assert relation.column_duplicate == "a"


def test_detect_derived_step_function():
    rng = np.random.default_rng(7)
    # who esta casi determinado por sex (pocos ninos): V >= 0.9.
    # El grupo de ninos es grande enough (>=40) para que el test
    # funcional lo evalue y sex NO se marque como derivada.
    n = 300
    sex = rng.choice(["male", "female"], size=n)
    age = rng.uniform(20, 80, size=n)
    child_idx = rng.choice(range(n), size=40, replace=False)
    age[child_idx] = rng.uniform(1, 17, size=40)
    who = np.where(age < 18, "child", np.where(sex == "male", "man", "woman"))

    df = pd.DataFrame({"sex": sex, "age": age, "who": who})
    profiles = _profiles(
        {"sex": "categorical", "age": "numeric", "who": "categorical"}
    )

    cat_cat = [
        calculate_categorical_categorical_relationship(
            df["sex"],
            df["who"],
            "sex",
            "who",
        )
    ]

    result = detect_equivalences(profiles, _prepared(df), cat_cat=cat_cat)

    derived = {item.column for item in result.derived}
    assert "who" in derived
    assert "sex" not in derived

    item = result.derived_by_column["who"]
    assert item.derived_from[0] == "sex"
    assert item.derived_from[1] == "age"


def test_is_step_function():
    numeric = pd.Series([1, 2, 3, 4, 5, 6])
    labels = pd.Series(["a", "a", "a", "b", "b", "b"])
    assert _is_step_function(numeric, labels, max_runs=2)

    noisy = pd.Series(["a", "b", "a", "b", "a", "b"])
    assert not _is_step_function(numeric, noisy, max_runs=2)


def test_one_to_one_rejects_non_bijective():
    x = pd.Series([1, 1, 2, 2])
    y = pd.Series([1, 2, 1, 2])
    assert not _one_to_one(x, y)
    assert _one_to_one(pd.Series([1, 2, 3]), pd.Series([4, 5, 6]))


def _relation(source: str, duplicate: str) -> EquivalenceRelation:
    return EquivalenceRelation(
        column_source=source,
        column_duplicate=duplicate,
        kind="cat_cat",
        strength=1.0,
        n_shared=10,
        label="contains the same information as",
    )


def _insight(columns: list[str], title: str = "x") -> Insight:
    return Insight(
        title=title,
        message="m",
        severity="info",
        category="relationship",
        columns=columns,
    )


def test_filter_insights_single_column_on_duplicate_dropped():
    eq = EquivalenceMap(relations=[_relation("survived", "alive")])
    kept = filter_insights([_insight(["alive"]), _insight(["survived"])], eq)

    assert [insight.columns for insight in kept] == [["survived"]]


def test_filter_insights_pair_on_fold_dropped():
    eq = EquivalenceMap(relations=[_relation("survived", "alive")])
    kept = filter_insights([_insight(["sex", "alive"])], eq)

    assert kept == []


def test_filter_insights_unrelated_kept():
    eq = EquivalenceMap(relations=[_relation("survived", "alive")])
    kept = filter_insights([_insight(["sex", "age"])], eq)

    assert len(kept) == 1


def test_filter_insights_no_map_passthrough():
    insights = [_insight(["alive"]), _insight(["sex"])]
    assert filter_insights(insights, EquivalenceMap()) == insights


def test_filter_insights_unnamed_kept():
    eq = EquivalenceMap(relations=[_relation("survived", "alive")])
    unnamed = Insight(title="t", message="m", severity="info")
    assert filter_insights([unnamed], eq) == [unnamed]


def test_filter_categorical_numeric_pairs_drops_redundant():
    eq = EquivalenceMap(
        relations=[_relation("survived", "alive")],
        derived=[DerivedColumn("who", ("sex", "age"), "evidence")],
    )
    pairs = [("who", "age"), ("sex", "who"), ("sex", "pclass"), ("alive", "fare")]
    kept = filter_categorical_numeric_pairs(pairs, eq)

    assert kept == [("sex", "pclass")]


def test_filter_relationships_drops_redundant():
    eq = EquivalenceMap(relations=[_relation("survived", "alive")])
    rel = CategoricalNumericRelationship(
        column_categorical="alive",
        column_numeric="fare",
        group_means={},
        group_medians={},
        group_counts={},
        n_observations=10,
        group_separation=1.0,
        n_groups=2,
        group_outliers={},
    )

    assert filter_relationships([rel], eq) == []


def test_equivalence_card_single_card_per_fold():
    eq = EquivalenceMap(
        relations=[
            _relation("survived", "alive"),
            _relation("embarked", "embark_town"),
        ]
    )
    card = equivalence_card(eq.relations)

    assert card is not None
    assert card.title == "Redundant columns carry the same information"
    assert "survived contains the same information as alive" in card.message
    assert "embarked and embark_town" in card.message


def test_equivalence_card_none_without_relations():
    assert equivalence_card([]) is None


def test_analyze_dataset_folds_duplicates():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=60, freq="D"),
            "region": ["north"] * 30 + ["south"] * 30,
            "region_label": ["Norte"] * 30 + ["Sur"] * 30,
            "revenue": list(range(100, 130)) + list(range(50, 80)),
            "revenue_copy": [float(v) for v in list(range(100, 130)) + list(range(50, 80))],
            "orders": list(range(10, 40)) + list(range(5, 35)),
            "profit": list(range(20, 50)) + list(range(10, 40)),
        }
    )

    report = analyze_dataset(df)

    structure = [insight for insight in report.structure_insights]
    equivalence_titles = [
        insight.title
        for insight in structure
        if insight.title == "Redundant columns carry the same information"
    ]

    assert len(equivalence_titles) == 1

    all_insights = (
        report.quality_insights
        + report.relationship_insights
        + report.distribution_insights
        + report.pattern_insights
        + report.structure_insights
        + report.aggregation_insights
        + report.comparative_insights
        + report.temporal_insights
        + report.inference_insights
        + report.ml_insights
    )

    for insight in all_insights:
        # La tarjeta de equivalencia es la unica que nombra la copia
        # (es su proposito); el resto no debe tocarla.
        if insight.title == "Redundant columns carry the same information":
            continue
        assert "revenue_copy" not in insight.columns
        assert "region_label" not in insight.columns

    temporal_titles = [insight.title for insight in report.temporal_insights]
    assert any("revenue trends down" in title for title in temporal_titles)
    assert not any("revenue_copy" in title for title in temporal_titles)
