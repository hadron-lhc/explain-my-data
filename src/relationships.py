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
