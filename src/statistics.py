"""
statistics.py

Calcula estadísticas descriptivas.

No genera gráficos, conclusiones ni modifica el dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

# Cantidad máxima de valores retenidos por columna categórica.
VALUE_COUNTS_CAP = 20


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

    top_values: list[tuple[str, int]] = field(default_factory=list)

    # Conteo completo de valores (hasta VALUE_COUNTS_CAP) para poder
    # evaluar concentración/balance en insights.
    value_counts: list[tuple[str, int]] = field(default_factory=list)

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

    counts_all = non_null.astype(str).value_counts()

    value_counts = [
        (str(value), int(count))
        for value, count in counts_all.head(VALUE_COUNTS_CAP).items()
    ]

    top_values = value_counts[:5]

    dominant_value = str(counts_all.index[0])

    dominant_percentage = round(
        float(counts_all.iloc[0] / len(non_null) * 100),
        2,
    )

    return CategoricalStatistics(
        column_name=column_name,
        count=len(non_null),
        n_unique=int(non_null.nunique()),
        top_values=top_values,
        value_counts=value_counts,
        dominant_value=dominant_value,
        dominant_percentage=dominant_percentage,
    )
