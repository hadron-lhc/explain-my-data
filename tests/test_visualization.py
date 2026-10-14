import pandas as pd

from src.visualizations import (
    VisualizationPlan,
    build_bar_plot,
    build_boxplot,
    build_contingency_heatmap,
    build_scatter_plot,
)


def test_build_scatter_plot_includes_trend_line():
    df = pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5],
            "y": [2, 4, 6, 8, 10],
        }
    )

    plan = VisualizationPlan(
        chart_type="scatter",
        trend_line=True,
    )

    result = build_scatter_plot(
        df,
        "x",
        "y",
        plan=plan,
    )

    assert result is not None
    assert result.trend_x == [1.0, 5.0]
    assert result.trend_y == [2.0, 10.0]


def test_build_scatter_plot_without_trend_line():
    df = pd.DataFrame(
        {
            "x": [1, 2, 3],
            "y": [2, 4, 6],
        }
    )

    plan = VisualizationPlan(
        chart_type="scatter",
        trend_line=False,
    )

    result = build_scatter_plot(
        df,
        "x",
        "y",
        plan=plan,
    )

    assert result is not None
    assert result.trend_x is None
    assert result.trend_y is None


def test_build_contingency_heatmap_returns_matrix():
    df = pd.DataFrame(
        {
            "size": ["S", "S", "L", "L"] * 5,
            "color": ["red", "blue", "red", "blue"] * 5,
        }
    )

    result = build_contingency_heatmap(df, "size", "color")

    assert result is not None
    assert result.value_label == "count"
    assert len(result.x_labels) == 2
    assert len(result.y_labels) == 2
    assert sum(sum(row) for row in result.values) == 20


def test_build_contingency_heatmap_rejects_high_cardinality():
    df = pd.DataFrame(
        {
            "id": [f"id_{i}" for i in range(40)],
            "color": ["red", "blue"] * 20,
        }
    )

    assert build_contingency_heatmap(df, "id", "color") is None


def test_build_contingency_heatmap_rejects_missing_columns():
    df = pd.DataFrame({"a": ["x", "y"]})

    assert build_contingency_heatmap(df, "a", "missing") is None


def test_build_bar_plot_respects_max_categories_and_sort():
    df = pd.DataFrame(
        {
            "category": ["a", "b", "c", "d"] * 5,
            "value": [1.0, 4.0, 2.0, 3.0] * 5,
        }
    )

    plan = VisualizationPlan(
        chart_type="ranked_bar",
        aggregation="mean",
        sort="ascending",
        max_categories=2,
    )

    result = build_bar_plot(df, "category", "value", plan=plan)

    assert result is not None
    assert result.categories == ["a", "c"]


def test_build_bar_plot_median_aggregation():
    df = pd.DataFrame(
        {
            "category": ["a", "a", "a", "b"],
            "value": [1.0, 1.0, 10.0, 5.0],
        }
    )

    plan = VisualizationPlan(
        chart_type="ranked_bar",
        aggregation="median",
        sort="descending",
    )

    result = build_bar_plot(df, "category", "value", plan=plan)

    assert result is not None
    assert dict(zip(result.categories, result.values)) == {"a": 1.0, "b": 5.0}


def test_build_boxplot_caps_points():
    df = pd.DataFrame(
        {
            "category": ["a"] * 300 + ["b"] * 300,
            "value": [float(i) for i in range(600)],
        }
    )

    result = build_boxplot(df, "category", "value", max_points=100)

    assert result is not None
    assert len(result.values) == 100
    assert len(result.categories) == 100
