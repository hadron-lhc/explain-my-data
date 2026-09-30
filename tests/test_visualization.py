import pandas as pd

from src.visualizations import build_scatter_plot, VisualizationPlan


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
