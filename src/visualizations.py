from dataclasses import dataclass

import pandas as pd

from src.planner import VisualizationPlan
from scipy.stats import linregress


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

    grouped = (
        data.groupby(
            "category",
            observed=True,
        )["value"]
        .mean()
        .sort_values(
            ascending=False,
        )
    )

    if plan is not None:
        if plan.aggregation == "median":
            grouped = data.groupby(
                "category",
                observed=True,
            )["value"].median()

            if plan.sort == "descending":
                grouped = grouped.sort_values(
                    ascending=False,
                )
            elif plan.sort == "ascending":
                grouped = grouped.sort_values(
                    ascending=True,
                )

        if plan.max_categories is not None and len(grouped) > plan.max_categories:
            grouped = grouped.head(
                plan.max_categories,
            )

    if grouped.empty:
        return None

    return BarPlotData(
        column_categorical=column_categorical,
        column_numeric=column_numeric,
        categories=grouped.index.tolist(),
        values=grouped.tolist(),
    )


def build_boxplot(
    df: pd.DataFrame,
    column_categorical: str,
    column_numeric: str,
    min_group_size: int | None = None,
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

    if min_group_size is not None:
        group_sizes = data.groupby("category").size()

        valid_categories = group_sizes[group_sizes >= min_group_size].index

        data = data[data["category"].isin(valid_categories)]

    if data.empty:
        return None

    return BoxPlotData(
        column_categorical=column_categorical,
        column_numeric=column_numeric,
        categories=data["category"].tolist(),
        values=data["value"].tolist(),
    )
