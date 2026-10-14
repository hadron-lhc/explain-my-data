
from src.aggregations import CategoricalNumericAggregation, GroupAggregation
from src.insights import (
    Insight,
    generate_aggregation_insights,
    generate_executive_summary,
    generate_pattern_insights,
    generate_quality_insights,
    generate_structure_insights,
    humanize_group_value,
)
from src.patterns import GroupDifferencePattern, NumericOutlierPattern
from src.profiling import ColumnProfile, DatasetProfile
from src.quality import ColumnQuality, QualityReport


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


def test_executive_summary_prioritizes_warnings_and_high_scores():
    insights = [
        Insight(title="info low", message="", severity="info", score=1.0),
        Insight(title="warn mid", message="", severity="warning", score=10.0),
        Insight(title="info high", message="", severity="info", score=50.0),
    ]

    summary = generate_executive_summary(insights, top_n=2)

    assert [insight.title for insight in summary] == ["warn mid", "info high"]


def test_executive_summary_respects_top_n():
    insights = [
        Insight(title=f"insight {i}", message="", severity="info", score=float(i))
        for i in range(20)
    ]

    summary = generate_executive_summary(insights, top_n=5)

    assert len(summary) == 5


def test_executive_summary_returns_all_when_fewer_insights():
    insights = [
        Insight(title="a", message="", severity="info", score=1.0),
    ]

    summary = generate_executive_summary(insights, top_n=8)

    assert len(summary) == 1
