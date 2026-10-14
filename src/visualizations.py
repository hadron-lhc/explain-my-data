from dataclasses import dataclass

import pandas as pd
from scipy.stats import linregress

from src.config import BOXPLOT_MAX_POINTS, HEATMAP_MAX_CATEGORIES
from src.planner import VisualizationPlan


@dataclass
class ScatterPlotData:
    column_x: str
    column_y: str
    x_values: list[float]
    y_values: list[float]

    trend_x: list[float] | None = None
    trend_y: list[float] | None = None


@dataclass
class BarPlotData:
    column_categorical: str
    column_numeric: str
    categories: list[str]
    values: list[float]


@dataclass
class BoxPlotData:
    column_categorical: str
    column_numeric: str
    categories: list[str]
    values: list[float]


@dataclass
class HeatmapPlotData:
    """Matriz de valores entre dos columnas de baja cardinalidad."""

    column_x: str
    column_y: str
    value_label: str
    x_labels: list[str]
    y_labels: list[str]
    values: list[list[float]]


def build_scatter_plot(
    df: pd.DataFrame,
    column_x: str,
    column_y: str,
    plan: VisualizationPlan | None = None,
) -> ScatterPlotData | None:
    x = pd.to_numeric(df[column_x], errors="coerce")
    y = pd.to_numeric(df[column_y], errors="coerce")

    data = pd.DataFrame(
        {
            "x": x,
            "y": y,
        }
    ).dropna()

    if len(data) < 2:
        return None

    if (
        plan is not None
        and plan.sampling
        and plan.max_points is not None
        and len(data) > plan.max_points
    ):
        data = data.sample(
            n=plan.max_points,
            random_state=42,
        )

    trend_x = None
    trend_y = None

    if plan is not None and plan.trend_line:
        result = linregress(
            data["x"],
            data["y"],
        )

        trend_x = [
            float(data["x"].min()),
            float(data["x"].max()),
        ]

        trend_y = [
            float(result.intercept + result.slope * trend_x[0]),
            float(result.intercept + result.slope * trend_x[1]),
        ]

    return ScatterPlotData(
        column_x=column_x,
        column_y=column_y,
        x_values=data["x"].tolist(),
        y_values=data["y"].tolist(),
        trend_x=trend_x,
        trend_y=trend_y,
    )


def build_bar_plot(
    df: pd.DataFrame,
    column_categorical: str,
    column_numeric: str,
    plan: VisualizationPlan | None = None,
) -> BarPlotData | None:
    numeric = pd.to_numeric(
        df[column_numeric],
        errors="coerce",
    )

    data = pd.DataFrame(
        {
            "category": (df[column_categorical].astype("string").str.strip()),
            "value": numeric,
        }
    ).dropna()

    if data.empty:
        return None

    aggregation = plan.aggregation if plan is not None else "mean"
    sort = plan.sort if plan is not None else "descending"
    max_categories = plan.max_categories if plan is not None else None

    grouped = data.groupby("category", observed=True)["value"]

    if aggregation == "median":
        series = grouped.median()
    else:
        series = grouped.mean()

    if sort == "ascending":
        series = series.sort_values(ascending=True)
    else:
        series = series.sort_values(ascending=False)

    if max_categories is not None and len(series) > max_categories:
        series = series.head(max_categories)

    if series.empty:
        return None

    return BarPlotData(
        column_categorical=column_categorical,
        column_numeric=column_numeric,
        categories=[str(category) for category in series.index.tolist()],
        values=series.tolist(),
    )


def build_boxplot(
    df: pd.DataFrame,
    column_categorical: str,
    column_numeric: str,
    *,
    max_points: int = BOXPLOT_MAX_POINTS,
) -> BoxPlotData | None:
    numeric = pd.to_numeric(
        df[column_numeric],
        errors="coerce",
    )

    data = pd.DataFrame(
        {
            "category": (df[column_categorical].astype("string").str.strip()),
            "value": numeric,
        }
    ).dropna()

    if data.empty:
        return None

    if len(data) > max_points:
        data = data.sample(n=max_points, random_state=42)

    return BoxPlotData(
        column_categorical=column_categorical,
        column_numeric=column_numeric,
        categories=data["category"].tolist(),
        values=data["value"].astype(float).tolist(),
    )


def build_contingency_heatmap(
    df: pd.DataFrame,
    column_x: str,
    column_y: str,
    *,
    max_categories: int = HEATMAP_MAX_CATEGORIES,
) -> HeatmapPlotData | None:
    """Heatmap de conteos de la tabla de contingencia de dos columnas."""
    if column_x not in df or column_y not in df:
        return None

    data = pd.DataFrame(
        {
            "x": df[column_x].astype("string").str.strip(),
            "y": df[column_y].astype("string").str.strip(),
        }
    ).dropna()

    if data.empty:
        return None

    contingency = pd.crosstab(data["x"], data["y"])

    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return None

    if (
        contingency.shape[0] > max_categories
        or contingency.shape[1] > max_categories
    ):
        return None

    return HeatmapPlotData(
        column_x=column_x,
        column_y=column_y,
        value_label="count",
        x_labels=[str(label) for label in contingency.columns.tolist()],
        y_labels=[str(label) for label in contingency.index.tolist()],
        values=contingency.astype(float).values.tolist(),
    )
