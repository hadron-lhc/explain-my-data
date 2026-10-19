
from src.aggregations import CategoricalNumericAggregation, GroupAggregation
from src.insights import (
    Insight,
    generate_aggregation_insights,
    generate_executive_summary,
    generate_pattern_insights,
    generate_quality_insights,
    generate_relationship_insights,
    generate_structure_insights,
    humanize_group_value,
)
from src.patterns import GroupDifferencePattern, NumericOutlierPattern
from src.profiling import ColumnProfile, DatasetProfile
from src.quality import ColumnQuality, QualityReport
from src.relationships import CategoricalNumericRelationship


def make_column_quality(**overrides) -> ColumnQuality:
    data = {
        "column_name": "price",
        "role": "numeric",
        "missing_count": 0,
        "missing_percentage": 0.0,
        "type_mismatch_count": 0,
        "is_type_mismatch": False,
        "conversion_rate": None,
        "n_outliers": 0,
        "outlier_percentage": 0.0,
        "unique_percentage": 0.5,
        "top_values": [],
        "n_invalid_values": 0,
        "invalid_percentage": 0.0,
        "n_extreme_values": 0,
        "extreme_percentage": 0.0,
    }

    data.update(overrides)

    return ColumnQuality(**data)


def make_quality_report(
    columns=None,
    n_duplicates=0,
    duplicate_percentage=0.0,
) -> QualityReport:
    return QualityReport(
        n_duplicates=n_duplicates,
        duplicate_percentage=duplicate_percentage,
        columns=columns or [],
    )


def test_generates_missing_values_insight():
    column = make_column_quality(
        missing_count=10,
        missing_percentage=5.0,
    )

    report = make_quality_report(
        columns=[column],
    )

    insights = generate_quality_insights(report)

    assert len(insights) == 1
    assert insights[0].title == "Missing values in price"
    assert "10 missing values" in insights[0].message
    assert "5.0%" in insights[0].message
    assert insights[0].severity == "warning"


def test_generates_type_mismatch_insight():
    column = make_column_quality(
        is_type_mismatch=True,
        conversion_rate=0.98,
        type_mismatch_count=2,
    )

    report = make_quality_report(
        columns=[column],
    )

    insights = generate_quality_insights(report)

    assert len(insights) == 1
    assert "Possible type mismatch" in insights[0].title
    assert "98.0%" in insights[0].message


def test_generates_extreme_values_insight():
    column = make_column_quality(
        column_name="price",
        n_extreme_values=3,
        extreme_percentage=1.5,
    )

    report = make_quality_report(
        columns=[column],
    )

    insights = generate_quality_insights(report)

    assert len(insights) == 1
    assert "Potentially extreme values" in insights[0].title
    assert "3 values" in insights[0].message
    assert "1.5%" in insights[0].message


def test_generates_invalid_values_insight():
    column = make_column_quality(
        column_name="price",
        n_invalid_values=2,
        invalid_percentage=1.0,
    )

    report = make_quality_report(
        columns=[column],
    )

    insights = generate_quality_insights(report)

    assert len(insights) == 1
    assert "Invalid values" in insights[0].title
    assert "2 non-finite numeric values" in insights[0].message
    assert "1.0%" in insights[0].message


def test_generates_duplicate_rows_insight():
    report = make_quality_report(
        n_duplicates=6,
        duplicate_percentage=2.0,
    )

    insights = generate_quality_insights(report)

    assert len(insights) == 1
    assert insights[0].title == "Duplicate rows detected"
    assert "6 duplicate rows" in insights[0].message


def test_generates_multiple_insights():
    column = make_column_quality(
        missing_count=3,
        missing_percentage=1.5,
        n_extreme_values=2,
        extreme_percentage=1.0,
    )

    report = make_quality_report(
        columns=[column],
        n_duplicates=4,
        duplicate_percentage=2.0,
    )

    insights = generate_quality_insights(report)

    assert len(insights) == 3

    assert any("Duplicate rows detected" in insight.title for insight in insights)

    assert any("Missing values" in insight.title for insight in insights)

    assert any("Potentially extreme values" in insight.title for insight in insights)


def test_returns_empty_list_when_quality_is_clean():
    report = make_quality_report(
        columns=[make_column_quality()],
    )

    insights = generate_quality_insights(report)

    assert insights == []


# ---------------------------------------------------------------------------
# Estructura del dataset
# ---------------------------------------------------------------------------


def _make_column_profile(name, role, n_unique, pct_unique, sample=None):
    return ColumnProfile(
        name=name,
        dtype="object",
        role=role,
        n_missing=0,
        pct_missing=0.0,
        n_unique=n_unique,
        pct_unique=pct_unique,
        sample_values=sample or [],
    )


def _make_dataset_profile(columns, n_rows=200):
    return DatasetProfile(
        n_rows=n_rows,
        n_columns=len(columns),
        n_duplicates=0,
        columns=columns,
    )


def test_structure_insights_detect_constant_column():
    profile = _make_dataset_profile(
        [
            _make_column_profile("flag", "categorical", 1, 0.5, sample=["on"]),
            _make_column_profile("city", "categorical", 5, 2.5),
        ]
    )

    insights = generate_structure_insights(profile)

    assert any("flag is constant" in insight.title for insight in insights)
    assert not any("city is constant" in insight.title for insight in insights)


def test_structure_insights_report_identifiers():
    profile = _make_dataset_profile(
        [
            _make_column_profile("customer_id", "identifier", 200, 100.0),
            _make_column_profile("city", "categorical", 5, 2.5),
        ]
    )

    insights = generate_structure_insights(profile)

    identifier_insight = next(
        insight for insight in insights if "Identifier" in insight.title
    )

    assert "customer_id" in identifier_insight.columns


# ---------------------------------------------------------------------------
# Patrones
# ---------------------------------------------------------------------------


def test_pattern_insights_from_group_differences():
    pattern = GroupDifferencePattern(
        group_column="country",
        numeric_column="revenue",
        group_value="B",
        group_size=50,
        group_mean=200.0,
        overall_mean=100.0,
        difference=100.0,
        standardized_difference=2.5,
        group_median=190.0,
        overall_median=95.0,
        median_difference=95.0,
        robust_standardized_difference=2.4,
    )

    insights = generate_pattern_insights([pattern], [])

    assert len(insights) == 1
    assert "country" in insights[0].title
    assert insights[0].category == "pattern"
    assert insights[0].chart_id == "catnum:country:revenue"
    assert insights[0].evidence["standardized_difference"] == 2.5


def test_pattern_insights_from_outliers():
    pattern = NumericOutlierPattern(
        column="revenue",
        n_observations=100,
        outlier_count=3,
        lower_bound=0.0,
        upper_bound=200.0,
        outlier_positions=[97, 98, 99],
        outlier_values=[500.0, 900.0, 5000.0],
    )

    insights = generate_pattern_insights([], [pattern])

    assert len(insights) == 1
    assert "IQR outliers" in insights[0].title
    assert insights[0].evidence["most_extreme"] == 5000.0
    assert insights[0].caveats


def test_pattern_insights_skip_weak_group_differences():
    pattern = GroupDifferencePattern(
        group_column="country",
        numeric_column="revenue",
        group_value="B",
        group_size=50,
        group_mean=105.0,
        overall_mean=100.0,
        difference=5.0,
        standardized_difference=0.2,
        group_median=104.0,
        overall_median=100.0,
        median_difference=4.0,
        robust_standardized_difference=0.15,
    )

    assert generate_pattern_insights([pattern], []) == []


# ---------------------------------------------------------------------------
# Agregaciones
# ---------------------------------------------------------------------------


def test_aggregation_insights_detect_dominant_group():
    aggregation = CategoricalNumericAggregation(
        column_categorical="country",
        column_numeric="revenue",
        n_observations=100,
        groups=[
            GroupAggregation(group_value="AR", count=80, mean=100.0, median=95.0),
            GroupAggregation(group_value="CL", count=20, mean=80.0, median=75.0),
        ],
    )

    insights = generate_aggregation_insights([aggregation])

    assert len(insights) == 1
    assert "AR" in insights[0].title
    assert insights[0].category == "aggregation"
    assert insights[0].evidence["dominant_share"] == 80.0


def test_aggregation_insights_skip_balanced_groups():
    aggregation = CategoricalNumericAggregation(
        column_categorical="country",
        column_numeric="revenue",
        n_observations=100,
        groups=[
            GroupAggregation(group_value="AR", count=50, mean=100.0, median=95.0),
            GroupAggregation(group_value="CL", count=50, mean=80.0, median=75.0),
        ],
    )

    assert generate_aggregation_insights([aggregation]) == []


def test_aggregation_insights_skip_boolean_negative_dominance():
    aggregation = CategoricalNumericAggregation(
        column_categorical="is_refunded",
        column_numeric="purchase_value",
        n_observations=1000,
        groups=[
            GroupAggregation(group_value="0", count=920, mean=100.0, median=95.0),
            GroupAggregation(group_value="1", count=80, mean=80.0, median=75.0),
        ],
    )

    roles = {"is_refunded": "boolean"}

    assert generate_aggregation_insights([aggregation], roles) == []


def test_aggregation_insights_alert_boolean_positive_dominance():
    aggregation = CategoricalNumericAggregation(
        column_categorical="is_refunded",
        column_numeric="purchase_value",
        n_observations=1000,
        groups=[
            GroupAggregation(group_value="1", count=920, mean=100.0, median=95.0),
            GroupAggregation(group_value="0", count=80, mean=80.0, median=75.0),
        ],
    )

    roles = {"is_refunded": "boolean"}
    insights = generate_aggregation_insights([aggregation], roles)

    assert len(insights) == 1
    insight = insights[0]
    assert insight.severity == "warning"
    assert "'refunded'" in insight.title
    assert "are 'refunded'" in insight.message
    # La evidencia conserva el valor crudo.
    assert insight.evidence["dominant_group"] == "1"
    assert insight.evidence["dominant_share"] == 92.0


def test_aggregation_insights_skip_weak_boolean_positive_dominance():
    aggregation = CategoricalNumericAggregation(
        column_categorical="is_refunded",
        column_numeric="purchase_value",
        n_observations=1000,
        groups=[
            GroupAggregation(group_value="1", count=700, mean=100.0, median=95.0),
            GroupAggregation(group_value="0", count=300, mean=80.0, median=75.0),
        ],
    )

    roles = {"is_refunded": "boolean"}

    assert generate_aggregation_insights([aggregation], roles) == []


def test_aggregation_insights_boolean_without_roles_keeps_default_behavior():
    aggregation = CategoricalNumericAggregation(
        column_categorical="is_refunded",
        column_numeric="purchase_value",
        n_observations=1000,
        groups=[
            GroupAggregation(group_value="0", count=920, mean=100.0, median=95.0),
            GroupAggregation(group_value="1", count=80, mean=80.0, median=75.0),
        ],
    )

    insights = generate_aggregation_insights([aggregation])

    assert len(insights) == 1
    assert "'0'" in insights[0].title


def test_humanize_group_value_translates_boolean_labels():
    roles = {"is_refunded": "boolean", "category": "categorical"}

    assert humanize_group_value("is_refunded", "0", roles) == "not refunded"
    assert humanize_group_value("is_refunded", "1", roles) == "refunded"
    assert humanize_group_value("is_refunded", "True", roles) == "refunded"
    assert humanize_group_value("category", "0", roles) == "0"
    assert humanize_group_value("is_refunded", "1", None) == "1"

    # Sin stem derivable se conserva el valor crudo.
    assert (
        humanize_group_value("codigo", "0", {"codigo": "boolean"}) == "0"
    )


def test_insight_id_is_stable_and_distinct():
    base = {
        "message": "m",
        "severity": "info",
        "category": "comparative",
        "columns": ["country", "revenue"],
    }

    first = Insight(title="same title", **base)
    second = Insight(title="same title", **base)
    other = Insight(title="other title", **base)

    assert first.id == second.id
    assert len(first.id) == 12
    assert first.id != other.id


# ---------------------------------------------------------------------------
# Executive summary
# ---------------------------------------------------------------------------


def test_executive_summary_is_narrative_by_family():
    # El resumen narrativo (P4) selecciona por familia usando la
    # `category` real de los insights (quality, relationship, …),
    # una frase por familia, en orden narrativo.
    insights = [
        Insight(
            title="warn",
            message="Missing values in age",
            severity="warning",
            category="quality",
            score=10.0,
        ),
        Insight(
            title="rel",
            message="age differs across income groups",
            severity="info",
            category="relationship",
            score=5.0,
        ),
        Insight(
            title="dist",
            message="age is right-skewed",
            severity="info",
            category="distribution",
            score=3.0,
        ),
    ]

    summary = generate_executive_summary(
        insights,
        n_rows=100,
        n_columns=3,
        top_n=8,
    )

    assert [insight.category for insight in summary] == [
        "executive_summary",
        "executive_summary",
        "executive_summary",
    ]
    assert summary[0].message == (
        "Dataset with 100 rows × 3 columns. "
        "Main quality signal: Missing values in age."
    )
    assert "age differs across income groups" in summary[1].message
    assert "(see: rel)" in summary[1].message
    assert "age is right-skewed" in summary[2].message


def test_executive_summary_without_warnings_reports_clean_dataset():
    summary = generate_executive_summary(
        [
            Insight(
                title="rel",
                message="a ~ b correlated",
                severity="info",
                category="relationship",
                score=5.0,
            ),
        ],
        n_rows=10,
        n_columns=2,
    )

    assert summary[0].message == "Dataset with 10 rows × 2 columns. No quality warnings."


def test_executive_summary_respects_top_n():
    # Cada familia aporta a lo sumo una frase; el tope se respeta.
    insights = [
        Insight(
            title=f"insight {i}",
            message=f"msg {i}",
            severity="info",
            category=f"family_{i}",
            score=float(i),
        )
        for i in range(20)
    ]

    summary = generate_executive_summary(insights, top_n=5)

    assert len(summary) <= 5


def test_executive_summary_returns_empty_when_no_insights():
    summary = generate_executive_summary([], top_n=8)

    assert summary == []


def test_insight_budget_caps_family_at_quota():
    from src.insights import apply_insight_budget

    insights = [
        Insight(
            title=f"rel {i}",
            message=f"msg {i}",
            severity="info",
            category="relationship_insights",
            score=float(10 - i),
        )
        for i in range(15)
    ]

    result = apply_insight_budget({"relationship_insights": insights})

    assert len(result["relationship_insights"]) == 12


def test_insight_budget_keeps_shared_chart_id_insights():
    # Regresión P4: varios insights pueden prometer el mismo chart
    # (la serie temporal completa, el histograma de una columna) y
    # cada uno es un hallazgo distinto. El corte es por score, nunca
    # por chart compartido — la dedupe anterior eliminaba régimen,
    # estacionalidad y extremos temporales (mismo chart de serie).
    from src.insights import apply_insight_budget

    insights = [
        Insight(
            title=title,
            message=msg,
            severity="info",
            category="temporal_insights",
            score=score,
            chart_id="temporal:date:temp",
        )
        for title, msg, score in (
            ("trend", "temp trends up", 21.8),
            ("cycle", "temp shows a 7-day cycle", 5.6),
            ("regime", "temp shifted around 2023-06-30", 8.1),
        )
    ]

    result = apply_insight_budget({"temporal_insights": insights})

    titles = {insight.title for insight in result["temporal_insights"]}
    assert titles == {"trend", "cycle", "regime"}


def test_insight_budget_caps_global_total():
    from src.insights import apply_insight_budget

    by_family = {
        "relationship_insights": [
            Insight(
                title=f"rel {i}",
                message=f"msg {i}",
                severity="info",
                category="relationship_insights",
                score=float(10 - i),
            )
            for i in range(8)
        ],
        "distribution_insights": [
            Insight(
                title=f"dist {i}",
                message=f"msg {i}",
                severity="info",
                category="distribution_insights",
                score=float(10 - i),
            )
            for i in range(4)
        ],
    }

    result = apply_insight_budget(by_family)

    total = sum(len(v) for v in result.values())
    assert total <= 40


def test_insight_budget_prefers_quality_over_context():
    from src.insights import apply_insight_budget

    by_family = {
        "quality_insights": [
            Insight(
                title=f"q {i}",
                message=f"msg {i}",
                severity="warning",
                category="quality_insights",
                score=float(10 - i),
            )
            for i in range(8)
        ],
        "aggregation_insights": [
            Insight(
                title=f"agg {i}",
                message=f"msg {i}",
                severity="info",
                category="aggregation_insights",
                score=float(100 - i),
            )
            for i in range(6)
        ],
    }

    result = apply_insight_budget(by_family)

    # La calidad se queda con su cupo completo; la de contexto
    # (aggregation) se recorta antes que la de sustancia, pero con
    # cupo suficiente para los need del corpus QA.
    assert len(result["quality_insights"]) == 8
    assert len(result["aggregation_insights"]) == 6


# ---------------------------------------------------------------------------
# P3: validez contextual (coverage + significancia práctica)
# ---------------------------------------------------------------------------


def make_cat_num_relationship(**overrides) -> CategoricalNumericRelationship:
    data = {
        "column_categorical": "deck",
        "column_numeric": "pclass",
        "group_means": {"A": 1.0, "B": 2.0},
        "group_medians": {"A": 1.0, "B": 2.0},
        "group_counts": {"A": 100, "B": 100},
        "n_observations": 200,
        "group_separation": 3.0,
        "n_groups": 2,
        "group_outliers": {},
        "coverage_pct": 100.0,
    }
    data.update(overrides)
    return CategoricalNumericRelationship(**data)


def test_cat_num_insight_skipped_below_minimum_coverage():
    # < 15%: el "hallazgo" describe un subconjunto demasiado chico.
    relationship = make_cat_num_relationship(coverage_pct=10.0)

    assert generate_relationship_insights([], [relationship]) == []


def test_cat_num_insight_caveats_on_partial_coverage():
    # 15–30%: se emite pero con caveat de coverage.
    relationship = make_cat_num_relationship(coverage_pct=22.8)

    insights = generate_relationship_insights([], [relationship])

    assert len(insights) == 1
    assert any("of rows" in caveat for caveat in insights[0].caveats)


def test_cat_num_insight_full_coverage_has_no_coverage_caveat():
    relationship = make_cat_num_relationship(coverage_pct=100.0)

    insights = generate_relationship_insights([], [relationship])

    assert len(insights) == 1
    assert not any("of rows" in caveat for caveat in insights[0].caveats)


def test_cat_num_insight_requires_practical_significance():
    # sep=0.28 (sobre el piso de 0.20) pero las medias solo se separan
    # 10%: diferencia estadística sin sustancia práctica.
    relationship = make_cat_num_relationship(
        column_categorical="sex",
        column_numeric="pclass",
        group_means={"female": 1.9, "male": 2.1},
        group_separation=0.28,
    )

    assert generate_relationship_insights([], [relationship]) == []


def test_cat_num_insight_passes_practical_significance():
    # sep=0.90 y medias 42% aparte: pasa el gate de práctica.
    relationship = make_cat_num_relationship(
        column_categorical="pclass",
        column_numeric="age",
        group_means={"1": 38.2, "2": 29.9, "3": 25.1},
        group_separation=0.90,
    )

    insights = generate_relationship_insights([], [relationship])

    assert len(insights) == 1


def make_group_difference_pattern(**overrides) -> GroupDifferencePattern:
    data = {
        "group_column": "deck",
        "numeric_column": "pclass",
        "group_value": "F",
        "group_size": 13,
        "group_mean": 2.38,
        "overall_mean": 1.20,
        "difference": 1.19,
        "standardized_difference": 3.77,
        "group_median": 2.0,
        "overall_median": 1.0,
        "median_difference": 1.0,
        "robust_standardized_difference": 3.5,
    }
    data.update(overrides)
    return GroupDifferencePattern(**data)


def test_pattern_insight_skipped_below_minimum_coverage():
    pattern = make_group_difference_pattern()
    outliers = []

    insights = generate_pattern_insights(
        [pattern],
        outliers,
        coverage_by_pair={("deck", "pclass"): 10.0},
    )

    assert insights == []


def test_pattern_insight_caveats_on_partial_coverage():
    pattern = make_group_difference_pattern()
    outliers = []

    insights = generate_pattern_insights(
        [pattern],
        outliers,
        coverage_by_pair={("deck", "pclass"): 22.8},
    )

    assert len(insights) == 1
    assert any("of rows" in caveat for caveat in insights[0].caveats)


def test_pattern_insight_full_coverage_by_default():
    # Sin cobertura calculada (compat) no se califica: se emite.
    pattern = make_group_difference_pattern()
    outliers = []

    insights = generate_pattern_insights([pattern], outliers)

    assert len(insights) == 1
