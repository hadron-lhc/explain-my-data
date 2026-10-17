import pandas as pd
import pytest

from src.analysis import analyze_dataset


def _mixed_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": range(1001, 1041),
            "country": ["AR", "CL", "AR", "CL"] * 10,
            "active": ["yes", "no"] * 20,
            "revenue": [float(i % 7) * 10 + 50 for i in range(40)],
            "units": [1 + (i % 3) for i in range(40)],
            "comment": [f"customer note {i} with some free text" for i in range(40)],
        }
    )


def test_analysis_report_includes_aggregations_and_group_differences():
    # revenue varia dentro de cada country: no es un codificado
    # biyectivo de country, asi que no se pliega.
    df = pd.DataFrame(
        {
            "country": ["A"] * 10 + ["B"] * 10,
            "revenue": [10.0, 12.0, 9.0, 14.0, 11.0, 10.0, 13.0, 9.0, 12.0, 11.0]
            + [100.0, 104.0, 97.0, 108.0, 101.0, 100.0, 106.0, 97.0, 104.0, 101.0],
        }
    )

    report = analyze_dataset(df)

    assert report.aggregations
    assert report.group_difference_patterns

    assert any(
        result.column_categorical == "country" and result.column_numeric == "revenue"
        for result in report.aggregations
    )

    assert any(
        pattern.group_column == "country" and pattern.numeric_column == "revenue"
        for pattern in report.group_difference_patterns
    )


def test_analysis_report_includes_numeric_outlier_patterns():
    df = pd.DataFrame(
        {
            "country": ["A"] * 10 + ["B"] * 10,
            "revenue": [10.0] * 10 + [100.0] * 9 + [10000.0],
        }
    )

    report = analyze_dataset(df)

    assert any(
        pattern.column == "revenue" and 10000.0 in pattern.outlier_values
        for pattern in report.numeric_outlier_patterns
    )


def test_full_pipeline_on_mixed_dataset():
    df = _mixed_dataset()

    report = analyze_dataset(df)

    roles = {column.name: column.role for column in report.profile.columns}

    assert roles["customer_id"] == "identifier"
    assert roles["country"] == "categorical"
    assert roles["active"] == "boolean"
    assert roles["revenue"] == "numeric"
    assert roles["comment"] == "text"

    numeric_names = {stats.column_name for stats in report.numeric_statistics}
    assert "revenue" in numeric_names
    assert "units" in numeric_names
    assert "customer_id" not in numeric_names

    assert isinstance(report.quality_insights, list)
    assert report.analysis_candidates

    assert report.profile.n_rows == 40
    assert report.profile.n_columns == 6


def test_analyze_dataset_does_not_modify_original_dataframe():
    df = _mixed_dataset()
    original = df.copy(deep=True)

    analyze_dataset(df)

    pd.testing.assert_frame_equal(df, original)


def test_boolean_columns_are_analyzed_as_categorical():
    df = pd.DataFrame(
        {
            "active": ["yes", "no"] * 20,
            "revenue": [float(i) for i in range(40)],
        }
    )

    report = analyze_dataset(df)

    assert any(
        stats.column_name == "active" for stats in report.categorical_statistics
    )

    assert any(
        relationship.column_categorical == "active"
        and relationship.column_numeric == "revenue"
        for relationship in report.categorical_numeric_relationships
    )


def test_full_pipeline_detects_categorical_categorical_association():
    # size y color fuertemente asociadas pero NO biyectivas (hay ruido):
    # una relacion 1:1 se pliegaria como equivalencia redundante.
    df = pd.DataFrame(
        {
            "size": ["S"] * 50 + ["M"] * 50,
            "color": ["red"] * 49 + ["blue"] * 1 + ["blue"] * 49 + ["red"] * 1,
            "revenue": [float(i) for i in range(100)],
        }
    )

    report = analyze_dataset(df)

    assert any(
        {relationship.column_x, relationship.column_y} == {"size", "color"}
        and relationship.cramers_v > 0.9
        for relationship in report.categorical_categorical_relationships
    )

    assert any(
        candidate.relationship_type == "categorical_categorical"
        for candidate in report.analysis_candidates
    )

    assert any(
        "size" in insight.title and "color" in insight.title
        for insight in report.relationship_insights
    )

    assert any(
        {heatmap.column_x, heatmap.column_y} == {"size", "color"}
        for heatmap in report.heatmaps
    )


def test_full_pipeline_builds_heatmap_for_discrete_numeric_pair():
    df = pd.DataFrame(
        {
            "rating": [1, 2, 3, 4] * 25,
            "score": [1, 1, 2, 2] * 25,
            "revenue": [float(i) for i in range(100)],
        }
    )

    report = analyze_dataset(df)

    assert any(
        {heatmap.column_x, heatmap.column_y} == {"rating", "score"}
        for heatmap in report.heatmaps
    )


def test_analyze_dataset_handles_dataset_without_numeric_columns():
    df = pd.DataFrame(
        {
            "city": ["ba", "mdq", "ba", "cri"] * 5,
            "zone": ["norte", "sur"] * 10,
        }
    )

    report = analyze_dataset(df)

    assert report.numeric_relationships == []
    assert report.categorical_numeric_relationships == []
    assert report.profile.n_columns == 2


def test_analyze_dataset_handles_dataset_without_categorical_columns():
    # y no es lineal con x: una copia lineal se pliega como redundante.
    df = pd.DataFrame(
        {
            "x": [float(i) for i in range(20)],
            "y": [float(i * i) for i in range(20)],
        }
    )

    report = analyze_dataset(df)

    assert report.categorical_statistics == []
    assert report.numeric_relationships


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame({"a": [1, 2, 3]}),
        _mixed_dataset(),
    ],
)
def test_analyze_dataset_always_returns_report(df: pd.DataFrame):
    report = analyze_dataset(df)
    assert report.profile.n_rows == len(df)


def test_null_insight_chart_ids_resolve_to_plots():
    n = 60
    df = pd.DataFrame(
        {
            "group": ["A"] * n + ["B"] * n,
            "value": [100.0 + (i % 3) * 0.1 for i in range(2 * n)],
            "x": [float(i) for i in range(2 * n)],
            "y": [float((i * 7919) % 97) for i in range(2 * n)],
        }
    )

    report = analyze_dataset(df)

    comparative = report.comparative_insights

    # Hallazgos nulos: el par queda fuera de la selección, pero el
    # gráfico que el insight promete se construye on-demand.
    assert any(
        insight.chart_id == "catnum:group:value" for insight in comparative
    )
    assert any(
        plot.column_categorical == "group" and plot.column_numeric == "value"
        for plot in report.bar_plots
    )

    assert any(
        insight.chart_id == "scatter:x:y" for insight in comparative
    )
    assert any(
        {plot.column_x, plot.column_y} == {"x", "y"}
        for plot in report.scatter_plots
    )

    # Todo chart_id emitido por cualquier familia debe resolver.
    insights = (
        report.relationship_insights
        + report.comparative_insights
        + report.pattern_insights
        + report.aggregation_insights
        + report.distribution_insights
    )

    checked = 0

    for insight in insights:
        if insight.chart_id is None:
            continue

        checked += 1
        kind, *parts = insight.chart_id.split(":")

        if kind == "scatter":
            resolved = any(
                {plot.column_x, plot.column_y} == set(parts[:2])
                for plot in report.scatter_plots
            )
        elif kind == "catnum":
            resolved = any(
                plot.column_categorical == parts[0]
                and plot.column_numeric == parts[1]
                for plot in report.bar_plots
            )
        elif kind == "catcat":
            resolved = any(
                {plot.column_x, plot.column_y} == set(parts[:2])
                for plot in report.heatmaps
            )
        elif kind == "hist":
            resolved = any(
                plot.column_name == parts[0]
                for plot in report.numeric_distributions
            )
        else:
            resolved = False

        assert resolved, f"unresolved chart_id: {insight.chart_id}"

    assert checked >= 5
