"""
relationships.py

Analiza relaciones entre variables.

Este módulo:
- mide asociaciones entre variables
- no modifica el DataFrame original
- no genera gráficos
- no genera conclusiones
- no intenta inferir causalidad
- no decide qué variable es un target

V1:
- numérico ↔ numérico
    - Pearson
    - Spearman
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class NumericRelationship:
    """Relación entre dos variables numéricas."""

    column_x: str
    column_y: str

    pearson: float
    spearman: float

    n_observations: int


@dataclass
class CategoricalNumericRelationship:
    column_categorical: str
    column_numeric: str

    group_means: dict[str, float]
    group_medians: dict[str, float]
    group_counts: dict[str, int]

    n_observations: int

    # Señal heurística para determinar si la relación
    # merece análisis adicional.
    group_separation: float


@dataclass
class ContextualRelationship:
    base_relationship: ...
    conditioning_columns: list[str]
    ...


def calculate_numeric_relationship(
    x: pd.Series,
    y: pd.Series,
    column_x: str,
    column_y: str,
) -> NumericRelationship | None:
    """
    Calcula la relación entre dos variables numéricas.

    Pearson mide asociación lineal.

    Spearman mide asociación monotónica y es menos sensible
    a relaciones no lineales que Pearson.

    Los pares con valores faltantes se eliminan.

    Retorna None si no existen suficientes observaciones.
    """

    data = pd.DataFrame(
        {
            "x": pd.to_numeric(x, errors="coerce"),
            "y": pd.to_numeric(y, errors="coerce"),
        }
    ).dropna()

    if len(data) < 2:
        return None

    pearson = data["x"].corr(
        data["y"],
        method="pearson",
    )

    spearman = data["x"].corr(
        data["y"],
        method="spearman",
    )

    return NumericRelationship(
        column_x=column_x,
        column_y=column_y,
        pearson=float(pearson),
        spearman=float(spearman),
        n_observations=len(data),
    )


def calculate_categorical_numeric_relationship(
    numeric_series: pd.Series,
    categorical_series: pd.Series,
    column_numeric: str,
    column_categorical: str,
) -> CategoricalNumericRelationship | None:
    """
    Calcula estadísticas agrupadas entre una variable categórica
    y una variable numérica.

    Retorna None si no existen suficientes observaciones.
    """

    data = pd.DataFrame(
        {
            "numeric": pd.to_numeric(numeric_series, errors="coerce"),
            "categorical": categorical_series,
        }
    ).dropna()

    if len(data) < 2:
        return None

    grouped = data.groupby("categorical")["numeric"]

    group_means = grouped.mean().to_dict()
    group_medians = grouped.median().to_dict()
    group_counts = grouped.count().to_dict()

    overall_std = data["numeric"].std()

    mean_range = max(group_means.values()) - min(group_means.values())

    if overall_std > 0:
        group_separation = mean_range / overall_std
    else:
        group_separation = 0.0

    return CategoricalNumericRelationship(
        column_categorical=column_categorical,
        column_numeric=column_numeric,
        group_means=group_means,
        group_medians=group_medians,
        group_counts=group_counts,
        n_observations=len(data),
        group_separation=group_separation,
    )
