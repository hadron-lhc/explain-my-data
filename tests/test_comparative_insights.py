import pandas as pd

from src.aggregations import calculate_categorical_numeric_aggregation
from src.analysis import analyze_dataset
from src.comparative_insights import (
    format_number,
    generate_balanced_insights,
    generate_comparative_insights,
    generate_consistency_insights,
    generate_driver_insights,
    generate_null_insights,
    generate_ranking_insights,
)
from src.insights import Insight, generate_executive_summary
from src.relationships import (
    calculate_categorical_numeric_relationship,
)
from src.statistics import calculate_categorical_statistics


def _agg(df, cat, num):
    result = calculate_categorical_numeric_aggregation(df, cat, num)
    assert result is not None
    return result


# ---------------------------------------------------------------------------
# Formato
# ---------------------------------------------------------------------------


def test_format_number_compact_and_grouped():
    assert format_number(5_002_341) == "5M"
    assert format_number(5_400_000) == "5.4M"
    assert format_number(411_203) == "411,203"
    assert format_number(1234) == "1,234"
    assert format_number(235.61) == "235.61"


# ---------------------------------------------------------------------------
# A. Ranking de grupo por total
# ---------------------------------------------------------------------------


def _ranking_df():
    return pd.DataFrame(
        {
            "category": ["electronics"] * 120 + ["audio"] * 120 + ["tv"] * 120,
            "spend": [1000.0] * 120 + [50.0] * 120 + [40.0] * 120,
        }
    )


def test_ranking_finds_dominant_group():
    df = _ranking_df()

    insights = generate_ranking_insights([_agg(df, "category", "spend")])

    assert len(insights) == 1
    insight = insights[0]
    assert "electronics" in insight.title
    assert "highest total spend" in insight.title
    assert "120,000" in insight.message
    assert insight.category == "comparative"
    assert insight.chart_id == "catnum:category:spend"


def test_ranking_skips_near_equal_totals():
    df = pd.DataFrame(
        {
            "category": ["a"] * 100 + ["b"] * 100,
            "spend": [100.0] * 100 + [95.0] * 100,
        }
    )

    assert generate_ranking_insights([_agg(df, "category", "spend")]) == []


def test_ranking_keeps_one_insight_per_category_column():
    df = _ranking_df()
    df["amount"] = df["spend"] * 3

    insights = generate_ranking_insights(
        [
            _agg(df, "category", "spend"),
            _agg(df, "category", "amount"),
        ]
    )

    assert len(insights) == 1


def test_ranking_skips_columns_where_totals_are_meaningless():
    df = pd.DataFrame(
        {
            "category": ["a"] * 60 + ["b"] * 60,
            "customer_age": [10.0] * 60 + [50.0] * 60,
        }
    )

    assert generate_ranking_insights([_agg(df, "category", "customer_age")]) == []


def test_ranking_avoids_total_total_phrasing():
    df = pd.DataFrame(
        {
            "category": ["a"] * 60 + ["b"] * 60,
            "total_sales": [100.0] * 60 + [10.0] * 60,
        }
    )

    insights = generate_ranking_insights([_agg(df, "category", "total_sales")])

    assert insights
    assert "total total" not in insights[0].title
    assert insights[0].title.endswith("highest total_sales")


def test_ranking_title_uses_accounts_for():
    df = _ranking_df()

    insights = generate_ranking_insights([_agg(df, "category", "spend")])

    assert insights
    assert insights[0].title.startswith(
        "electronics accounts for the highest"
    )
    assert "generates" not in insights[0].title


def test_ranking_full_coverage_claims_of_all():
    df = _ranking_df()

    insights = generate_ranking_insights([_agg(df, "category", "spend")])

    assert insights
    assert "% of all spend)" in insights[0].message
    assert insights[0].evidence["coverage_pct"] == 100.0


def test_ranking_partial_coverage_qualifies_subset():
    # refund_reason solo está seteado en una fracción de las filas:
    # el share es del subconjunto y el texto debe decirlo.
    df = pd.DataFrame(
        {
            "refund_reason": ["late_delivery"] * 20
            + ["damaged"] * 20
            + [None] * 160,
            "purchase_value": [200.0] * 20 + [7_000.0] * 20 + [1_000.0] * 160,
        }
    )

    insights = generate_ranking_insights(
        [_agg(df, "refund_reason", "purchase_value")]
    )

    assert insights
    message = insights[0].message
    assert "of all purchase_value" not in message
    assert "where refund_reason is set" in message
    assert "covers 20% of rows" in message
    assert insights[0].evidence["coverage_pct"] == 20.0


def test_ranking_skips_boolean_columns():
    df = pd.DataFrame(
        {
            "is_refunded": ["0"] * 180 + ["1"] * 20,
            "purchase_value": [500.0] * 180 + [80.0] * 20,
        }
    )

    roles = {"is_refunded": "boolean"}

    aggregations = [_agg(df, "is_refunded", "purchase_value")]

    assert generate_ranking_insights(aggregations, roles) == []


def test_consistency_skips_boolean_columns():
    df = _ranking_df()
    df["is_refunded"] = ["0"] * 360
    df.loc[df.index[:20], "is_refunded"] = "1"
    df["amount"] = df["spend"] * 2

    roles = {"is_refunded": "boolean", "category": "categorical"}
    aggregations = [
        _agg(df, "is_refunded", "spend"),
        _agg(df, "is_refunded", "amount"),
    ]

    # Sin roles el insight se emite; con roles boolean se descarta.
    assert generate_consistency_insights(aggregations)
    assert generate_consistency_insights(aggregations, roles) == []


# ---------------------------------------------------------------------------
# B. Driver volumen vs valor
# ---------------------------------------------------------------------------


def test_driver_detects_value_driven_differences():
    df = pd.DataFrame(
        {
            "category": ["a"] * 100 + ["b"] * 100 + ["c"] * 100,
            "spend": [10.0] * 100 + [100.0] * 100 + [1000.0] * 100,
        }
    )

    insights = generate_driver_insights([_agg(df, "category", "spend")])

    assert len(insights) == 1
    assert "average values rather than group sizes" in insights[0].title


def test_driver_detects_volume_driven_differences():
    df = pd.DataFrame(
        {
            "category": ["a"] * 10 + ["b"] * 500 + ["c"] * 500,
            "spend": [50.0] * 10 + [51.0] * 500 + [49.0] * 500,
        }
    )

    insights = generate_driver_insights([_agg(df, "category", "spend")])

    assert len(insights) == 1
    assert "group sizes rather than averages" in insights[0].title


def test_driver_stays_quiet_when_nothing_standout():
    df = pd.DataFrame(
        {
            "category": ["a"] * 100 + ["b"] * 150 + ["c"] * 120,
            "spend": [100.0] * 100 + [110.0] * 150 + [90.0] * 120,
        }
    )

    assert generate_driver_insights([_agg(df, "category", "spend")]) == []


# ---------------------------------------------------------------------------
# C. Consistencia entre métricas
# ---------------------------------------------------------------------------


def test_consistency_reports_same_leader_on_second_metric():
    df = pd.DataFrame(
        {
            "category": ["a"] * 120 + ["b"] * 120,
            "spend": [100.0] * 120 + [10.0] * 120,
            "refunds": [80.0] * 120 + [5.0] * 120,
        }
    )

    insights = generate_consistency_insights(
        [
            _agg(df, "category", "spend"),
            _agg(df, "category", "refunds"),
        ]
    )

    assert len(insights) == 1
    insight = insights[0]
    # La primaria es la métrica con mayor gap (refunds), por lo que
    # el "also leads" se emite sobre la otra (spend).
    assert insight.title == "a also leads in spend"
    assert "Beyond refunds" in insight.message


def test_consistency_stays_quiet_when_leaders_differ():
    df = pd.DataFrame(
        {
            "category": ["a"] * 120 + ["b"] * 120,
            "spend": [100.0] * 120 + [10.0] * 120,
            "refunds": [5.0] * 120 + [90.0] * 120,
        }
    )

    insights = generate_consistency_insights(
        [
            _agg(df, "category", "spend"),
            _agg(df, "category", "refunds"),
        ]
    )

    assert insights == []


# ---------------------------------------------------------------------------
# D. Distribuciones balanceadas
# ---------------------------------------------------------------------------


def test_balanced_distribution_detected():
    values = ["late", "wrong", "damaged", "changed"] * 100
    statistics = calculate_categorical_statistics(pd.Series(values), "reason")

    insights = generate_balanced_insights([statistics])

    assert len(insights) == 1
    assert "no dominant value" in insights[0].title
    assert "roughly 25%" in insights[0].message


def test_dominant_distribution_is_not_balanced():
    values = ["late"] * 240 + ["wrong"] * 40 + ["damaged"] * 40 + ["other"] * 80
    statistics = calculate_categorical_statistics(pd.Series(values), "reason")

    assert generate_balanced_insights([statistics]) == []


def test_binary_distribution_is_not_balanced():
    values = ["yes", "no"] * 100
    statistics = calculate_categorical_statistics(pd.Series(values), "flag")

    assert generate_balanced_insights([statistics]) == []


# ---------------------------------------------------------------------------
# E. Hallazgos nulos
# ---------------------------------------------------------------------------


def test_binary_null_finding_with_close_means():
    # Medias casi iguales con varianza real intra-grupo: la
    # separación en σ es chiquísima → hallazgo nulo.
    import numpy as np

    offsets = np.array([-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0])
    values = np.concatenate(
        [235.0 + np.resize(offsets, 70), 234.0 + np.resize(offsets, 70)]
    )

    df = pd.DataFrame(
        {
            "discounted": ["yes"] * 70 + ["no"] * 70,
            "purchase_value": values,
        }
    )

    relationship = calculate_categorical_numeric_relationship(
        df["purchase_value"],
        df["discounted"],
        "purchase_value",
        "discounted",
    )

    insights = generate_null_insights([], [relationship], [])

    assert len(insights) == 1
    insight = insights[0]
    assert "nearly identical" in insight.title
    assert "235.00" in insight.message
    assert "234.00" in insight.message
    assert "0.4%" in insight.message
    assert "not strongly associated" in insight.message


def test_binary_null_finding_rejects_large_difference():
    df = pd.DataFrame(
        {
            "discounted": ["yes", "no"] * 60,
            "purchase_value": [
                300.0 if flag == "yes" else 100.0
                for flag in (["yes", "no"] * 60)
            ],
        }
    )

    relationship = calculate_categorical_numeric_relationship(
        df["purchase_value"],
        df["discounted"],
        "purchase_value",
        "discounted",
    )

    assert generate_null_insights([], [relationship], []) == []


def test_weak_correlation_null_finding():
    from src.relationships import NumericRelationship

    relationship = NumericRelationship(
        column_x="price",
        column_y="discount",
        pearson=0.05,
        spearman=-0.03,
        n_observations=200,
    )

    insights = generate_null_insights([relationship], [], [])

    assert len(insights) == 1
    assert "No meaningful association" in insights[0].title
    assert "0.05" in insights[0].message


def test_strong_correlation_is_not_a_null_finding():
    from src.relationships import NumericRelationship

    relationship = NumericRelationship(
        column_x="price",
        column_y="revenue",
        pearson=0.9,
        spearman=0.85,
        n_observations=200,
    )

    assert generate_null_insights([relationship], [], []) == []


def test_independent_pair_null_finding():
    from src.relationships import CategoricalCategoricalRelationship

    relationship = CategoricalCategoricalRelationship(
        column_x="city",
        column_y="payment",
        n_observations=500,
        n_groups_x=4,
        n_groups_y=3,
        chi2=3.2,
        p_value=0.52,
        cramers_v=0.08,
    )

    insights = generate_null_insights([], [], [relationship])

    assert len(insights) == 1
    assert "appear independent" in insights[0].title
    assert "0.08" in insights[0].message


def test_null_insights_respect_cap():
    from src.relationships import NumericRelationship

    relationships = [
        NumericRelationship(
            column_x=f"x{i}",
            column_y=f"y{i}",
            pearson=0.0,
            spearman=0.0,
            n_observations=100,
        )
        for i in range(12)
    ]

    insights = generate_null_insights(relationships, [], [])

    assert len(insights) == 5


# ---------------------------------------------------------------------------
# Entry point + integración con el reporte
# ---------------------------------------------------------------------------


def test_generate_comparative_insights_combines_families():
    df = _ranking_df()

    insights = generate_comparative_insights(
        [_agg(df, "category", "spend")],
        [],
        [],
        [],
        [],
    )

    assert insights
    assert all(insight.category == "comparative" for insight in insights)


def test_full_report_includes_comparative_insights():
    df = pd.DataFrame(
        {
            "category": ["a"] * 80 + ["b"] * 80 + ["c"] * 80,
            "spend": [10.0] * 80 + [500.0] * 80 + [12.0] * 80,
            "refunds": [1.0] * 80 + [40.0] * 80 + [2.0] * 80,
        }
    )

    report = analyze_dataset(df)

    assert report.comparative_insights
    assert any(
        insight in report.executive_summary
        for insight in report.comparative_insights
    )


def test_executive_summary_caps_quality_warnings():
    insights = [
        Insight(
            title=f"warning {i}",
            message="",
            severity="warning",
            score=10.0,
        )
        for i in range(6)
    ] + [
        Insight(
            title=f"comparative {i}",
            message="",
            severity="info",
            score=90.0 - i,
        )
        for i in range(6)
    ]

    summary = generate_executive_summary(insights, top_n=8)

    warnings = [i for i in summary if i.severity == "warning"]

    assert len(summary) == 8
    assert len(warnings) == 2


def test_executive_summary_backfills_with_warnings_when_needed():
    insights = [
        Insight(title="w1", message="", severity="warning", score=10.0),
        Insight(title="w2", message="", severity="warning", score=9.0),
        Insight(title="w3", message="", severity="warning", score=8.0),
        Insight(title="i1", message="", severity="info", score=95.0),
    ]

    summary = generate_executive_summary(insights, top_n=5)

    assert len(summary) == 4
    assert sum(1 for i in summary if i.severity == "warning") == 3
