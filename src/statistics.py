"""
statistics.py

Calcula estadísticas descriptivas.

No genera gráficos, conclusiones ni modifica el dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import pandas as pd


@dataclass
class NumericStatistics:
    column_name: str

    count: int

    mean: float
    median: float
    std: float

    min: float
    q1: float
    q3: float
    max: float

    skewness: float


@dataclass
class CategoricalStatistics:
    column_name: str

    count: int
    n_unique: int

    top_values: List[Tuple[str, int]] = field(default_factory=list)

    dominant_value: str | None = None
    dominant_percentage: float = 0.0


def calculate_numeric_statistics(
    series: pd.Series,
    column_name: str,
) -> NumericStatistics | None:
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if numeric.empty:
        return None

    return NumericStatistics(
        column_name=column_name,
        count=int(numeric.count()),
        mean=float(numeric.mean()),
        median=float(numeric.median()),
        std=float(numeric.std()),
        min=float(numeric.min()),
        q1=float(numeric.quantile(0.25)),
        q3=float(numeric.quantile(0.75)),
        max=float(numeric.max()),
        skewness=float(numeric.skew()),
    )


def calculate_categorical_statistics(
    series: pd.Series,
    column_name: str,
) -> CategoricalStatistics | None:
    non_null = series.dropna()

    if non_null.empty:
        return None

    value_counts = non_null.astype(str).value_counts()

    top_values = [
        (str(value), int(count)) for value, count in value_counts.head(5).items()
    ]

    dominant_value = str(value_counts.index[0])

    dominant_percentage = round(
        float(value_counts.iloc[0] / len(non_null) * 100),
        2,
    )

    return CategoricalStatistics(
        column_name=column_name,
        count=int(len(non_null)),
        n_unique=int(non_null.nunique()),
        top_values=top_values,
        dominant_value=dominant_value,
        dominant_percentage=dominant_percentage,
    )
