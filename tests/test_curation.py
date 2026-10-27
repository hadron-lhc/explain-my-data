"""Tests de la curación (un hallazgo = una tarjeta)."""

from src.curation import (
    cap_pattern_insights,
    curate_insights,
    fold_derived_cards,
    fold_group_differs_insights,
    fold_leader_cards,
    fold_outlier_cards,
)
from src.insights import Insight


def _standout(
    title: str,
    num_col: str,
    group_col: str,
    std: float,
    score: float,
) -> Insight:
    return Insight(
        title=f"Group '{title}' of {group_col} stands out in {num_col}",
        message=f"Group '{title}' of {group_col} stands out in {num_col}",
        severity="warning",
        score=score,
        columns=[group_col, num_col],
        category="pattern",
        evidence={"standardized_difference": std, "tag": title},
    )


def _differs(num_col: str, group_col: str, separation: float, n: int) -> Insight:
    return Insight(
        title=f"{num_col} differs across {n} {group_col} groups",
        message=f"{num_col} differs across {n} {group_col} groups",
        severity="warning",
        score=5.0,
        columns=[group_col, num_col],
        category="relationship",
        evidence={"group_separation": separation, "n_groups": n},
    )


def test_cap_keeps_top_by_standardized_difference():
    patterns = [
        _standout("a", "x", "g", -1.2, 1.2),
        _standout("b", "x", "g", -4.1, 4.1),
        _standout("c", "x", "g", 1.8, 1.8),
        _standout("d", "x", "g", 3.0, 3.0),
    ]

    kept = cap_pattern_insights(patterns, cap=2)

    tags = [insight.evidence["tag"] for insight in kept]

    assert tags == ["b", "d"]


def test_cap_keeps_both_signed_extremes_not_top2_abs():
    # Caso gapminder: Oceania (+1.26), Europe (+0.80), Africa (-0.95).
    # Top-2 por |std| daría Oceania+Europe; la historia real es
    # Oceania (más alto) y Africa (más bajo).
    patterns = [
        _standout("Oceania", "x", "g", 1.26, 1.26),
        _standout("Europe", "x", "g", 0.80, 0.80),
        _standout("Africa", "x", "g", -0.95, 0.95),
    ]

    kept = cap_pattern_insights(patterns, cap=2)

    tags = {insight.evidence["tag"] for insight in kept}

    assert tags == {"Oceania", "Africa"}


def test_cap_preserves_order_of_survivors():
    patterns = [
        _standout("a", "x", "g", -1.2, 1.2),
        _standout("b", "x", "g", -4.1, 4.1),
        _standout("mid", "x", "g", 1.5, 1.5),
        _standout("c", "other", "g", 0.5, 0.5),
    ]

    kept = cap_pattern_insights(patterns, cap=2)

    assert [insight.evidence["tag"] for insight in kept] == ["b", "mid", "c"]


def test_cap_does_not_touch_non_standout_patterns():
    other = Insight(
        title="Some outlier pattern",
        message="",
        severity="warning",
        score=2.0,
        columns=["x"],
        category="pattern",
        evidence={"standardized_difference": 9.9},
    )
    patterns = [
        _standout("a", "x", "g", -1.2, 1.2),
        _standout("b", "x", "g", -4.1, 4.1),
        _standout("c", "x", "g", 3.0, 3.0),
        other,
    ]

    kept = cap_pattern_insights(patterns, cap=2)

    assert other in kept


def test_cap_is_per_pair_not_global():
    patterns = [
        _standout("a", "x", "g", -1.2, 1.2),
        _standout("b", "x", "g", -4.1, 4.1),
        _standout("c", "y", "g", 1.0, 1.0),
        _standout("d", "y", "g", 2.0, 2.0),
    ]

    kept = cap_pattern_insights(patterns, cap=2)

    assert len(kept) == 4


def test_fold_transfers_separation_to_primary_pattern():
    primary = _standout("p1", "x", "g", -4.0, 4.0)
    secondary = _standout("p2", "x", "g", 2.0, 2.0)
    differs = _differs("x", "g", 1.85, 5)

    kept_rel = fold_group_differs_insights([differs], [primary, secondary])

    assert kept_rel == []
    assert primary.evidence["group_separation"] == 1.85
    assert primary.evidence["n_groups"] == 5
    assert any("group separation" in caveat for caveat in primary.caveats)
    # El patrón secundario no toca.
    assert "group_separation" not in secondary.evidence


def test_fold_promotes_primary_severity_from_relationship():
    primary = _standout("p1", "x", "g", -4.0, 4.0)
    primary.severity = "info"
    differs = _differs("x", "g", 1.85, 5)

    fold_group_differs_insights([differs], [primary])

    assert primary.evidence["group_separation"] == 1.85
    assert primary.severity == "warning"


def test_fold_uses_highest_score_pattern_as_primary():
    weak = _standout("p1", "x", "g", 2.0, 2.0)
    strong = _standout("p2", "x", "g", -4.0, 4.0)
    differs = _differs("x", "g", 2.1, 5)

    fold_group_differs_insights([differs], [weak, strong])

    assert strong.evidence["group_separation"] == 2.1
    assert "group_separation" not in weak.evidence


def test_fold_keeps_differs_when_no_pattern_for_pair():
    differs = _differs("x", "g", 2.0, 4)
    pattern_other_pair = _standout("p", "z", "g", 3.0, 3.0)

    kept_rel = fold_group_differs_insights([differs], [pattern_other_pair])

    assert kept_rel == [differs]
    assert "group_separation" not in pattern_other_pair.evidence


def test_fold_keeps_non_differs_relationships():
    ranking = Insight(
        title="Ranking of g by x",
        message="",
        severity="info",
        score=90.0,
        columns=["g", "x"],
        category="relationship",
        evidence={},
    )
    differs = _differs("x", "g", 2.0, 5)
    pattern = _standout("p", "x", "g", -3.0, 3.0)

    kept_rel = fold_group_differs_insights([ranking, differs], [pattern])

    assert ranking in kept_rel
    assert differs not in kept_rel


def test_curate_insights_end_to_end():
    patterns = [
        _standout("p1", "x", "g", -4.0, 4.0),
        _standout("p2", "x", "g", 2.0, 2.0),
        _standout("p3", "x", "g", 1.2, 1.2),
    ]
    differs = _differs("x", "g", 1.9, 5)
    other_rel = _differs("z", "g", 2.5, 5)

    relationships, curated_patterns = curate_insights([differs, other_rel], patterns)

    assert relationships == [other_rel]
    assert [i.evidence["tag"] for i in curated_patterns] == ["p1", "p2"]
    assert curated_patterns[0].evidence["group_separation"] == 1.9


# --- F4: folds de ruido ----------------------------------------------------


def _outlier_card(column: str, count: int, most_extreme: float) -> Insight:
    return Insight(
        title=f"{column} contains {count} IQR outliers",
        message=(
            f"{column} has {count} values outside the IQR range; "
            f"Most extreme value: {most_extreme:.2f}."
        ),
        severity="info",
        category="pattern",
        columns=[column],
        score=float(count),
        evidence={
            "outlier_count": count,
            "n_observations": 100,
            "outlier_percentage": float(count),
            "lower_bound": 0.0,
            "upper_bound": 10.0,
            "most_extreme": most_extreme,
        },
    )


def test_fold_outlier_cards_merges_and_preserves_extremes():
    cards = [
        _outlier_card("wt", 3, 5.42),
        _outlier_card("carb", 1, 8.0),
        _outlier_card("qsec", 2, 22.9),
    ]

    folded = fold_outlier_cards(cards)

    assert len(folded) == 1
    message = folded[0].message
    assert "most extreme value: 5.42" in message
    assert "most extreme value: 8.00" in message
    assert "most extreme value: 22.90" in message
    assert folded[0].columns == ["wt", "carb", "qsec"]
    assert folded[0].evidence["folded_outlier_columns"] == 3


def test_fold_outlier_cards_keeps_single_card():
    card = _outlier_card("dist", 1, 120.0)

    folded = fold_outlier_cards([card])

    assert folded == [card]


def test_fold_leader_cards_merges_also_leads():
    primary = Insight(
        title="Dinner accounts for the highest total total_bill",
        message="Dinner totals 4,000.",
        severity="info",
        category="comparative",
        columns=["time", "total_bill"],
        score=10.0,
    )
    secondary = Insight(
        title="Dinner also leads in tip",
        message="Dinner tips 546.07, 2.9x Lunch.",
        severity="info",
        category="comparative",
        columns=["time", "tip"],
        score=8.0,
    )
    orphan = Insight(
        title="Lunch also leads in size",
        message="Lunch size.",
        severity="info",
        category="comparative",
        columns=["time", "size"],
        score=2.0,
    )

    folded = fold_leader_cards([primary, secondary, orphan])

    assert primary in folded
    assert secondary not in folded
    assert orphan in folded
    assert "Dinner also leads in tip." in primary.message
    assert "546.07" in primary.message
    assert primary.columns == ["time", "total_bill", "tip"]


def test_fold_derived_cards_merges_and_keeps_equivalence():
    derived = [
        Insight(
            title="adult_male looks derived from other columns",
            message="adult_male appears to be derived from sex, age.",
            severity="info",
            category="structure",
            columns=["adult_male", "sex", "age"],
        ),
        Insight(
            title="who looks derived from other columns",
            message="who appears to be derived from sex, age.",
            severity="info",
            category="structure",
            columns=["who", "sex", "age"],
        ),
    ]
    equivalence = Insight(
        title="Redundant columns carry the same information",
        message="education_num encodes education.",
        severity="info",
        category="structure",
        columns=["education", "education_num"],
    )

    folded = fold_derived_cards(derived + [equivalence])

    assert equivalence in folded
    assert len(
        [i for i in folded if "derived from other columns" in i.title]
    ) == 1
    merged = next(
        i for i in folded if "derived from other columns" in i.title
    )
    assert "adult_male appears to be derived from sex, age" in merged.message
    assert "who appears to be derived from sex, age" in merged.message
    assert "adult_male" in merged.evidence["derived_columns"]
    assert "who" in merged.evidence["derived_columns"]
    assert merged.score == 2.0
