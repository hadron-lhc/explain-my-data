from src.insights import generate_quality_insights
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
    report = make_quality_report(columns=[column])

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
    report = make_quality_report(columns=[column])

    insights = generate_quality_insights(report)

    assert len(insights) == 1
    assert "Possible type mismatch" in insights[0].title
    assert "98.0%" in insights[0].message


def test_generates_outlier_insight():
    column = make_column_quality(
        n_outliers=4,
        outlier_percentage=2.0,
    )
    report = make_quality_report(columns=[column])

    insights = generate_quality_insights(report)

    assert len(insights) == 1
    assert any("Potential unusual values" in insight.title for insight in insights)
    assert "4 values outside the overall IQR range" in insights[0].message


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
        n_outliers=2,
        outlier_percentage=1.0,
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
    assert any("Potential unusual values" in insight.title for insight in insights)


def test_returns_empty_list_when_quality_is_clean():
    report = make_quality_report(
        columns=[make_column_quality()],
    )

    insights = generate_quality_insights(report)

    assert insights == []
