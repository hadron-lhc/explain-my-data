from dataclasses import dataclass

import pandas as pd


@dataclass
class ScatterPlotData:
    column_x: str
    column_y: str
    x_values: list[float]
    y_values: list[float]


def build_scatter_plot(
    df: pd.DataFrame,
    column_x: str,
    column_y: str,
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

    return ScatterPlotData(
        column_x=column_x,
        column_y=column_y,
        x_values=data["x"].tolist(),
        y_values=data["y"].tolist(),
    )
