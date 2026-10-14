"""
aggregations.py

Calcula agregaciones por grupo a partir de un dataset desconocido.

Responsabilidades:
- Seleccionar combinaciones categórica-numérica razonables.
- Calcular métricas por grupo.
- Excluir combinaciones con cardinalidad excesiva.
- Calcular sumas y participaciones solo para columnas
  identificadas explícitamente como aditivas.

No detecta patrones ni genera insights.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.profiling import CATEGORICAL_LIKE_ROLES, ColumnProfile


@dataclass
class GroupAggregation:
    group_value: str
    count: int
    mean: float
    median: float
    total: float | None = None
    share_pct: float | None = None


@dataclass
class CategoricalNumericAggregation:
    column_categorical: str
    column_numeric: str
    n_observations: int
    groups: list[GroupAggregation] = field(default_factory=list)
    total_rows: int = 0
    coverage_pct: float = 100.0


def select_categorical_numeric_pairs(
    profiles: list[ColumnProfile],
    max_categories: int = 20,
) -> list[tuple[str, str]]:
    """
    Selecciona pares (categoría, variable numérica).

    Excluye identificadores y categorías con demasiados valores únicos.
    La selección se basa en los roles inferidos durante el perfilado.
    """
    categorical_columns = [
        profile
        for profile in profiles
        if profile.role in CATEGORICAL_LIKE_ROLES
        and 2 <= profile.n_unique <= max_categories
    ]

    numeric_columns = [profile for profile in profiles if profile.role == "numeric"]

    return [
        (categorical.name, numeric.name)
        for categorical in categorical_columns
        for numeric in numeric_columns
        if categorical.name != numeric.name
    ]


def calculate_categorical_numeric_aggregation(
    df: pd.DataFrame,
    column_categorical: str,
    column_numeric: str,
    *,
    additive: bool = False,
) -> CategoricalNumericAggregation | None:
    """
    Calcula métricas numéricas para cada grupo categórico.

    additive=True permite calcular sumas y participación porcentual.
    Debe utilizarse únicamente cuando sumar la variable tenga sentido
    analítico. No se infiere esa propiedad a partir del dtype.
    """
    if column_categorical not in df.columns:
        return None

    if column_numeric not in df.columns:
        return None

    numeric = pd.to_numeric(df[column_numeric], errors="coerce")

    # Excluir infinitos y valores faltantes.
    numeric = numeric.where(np.isfinite(numeric))

    data = pd.DataFrame(
        {
            "group": df[column_categorical],
            "value": numeric,
        }
    ).dropna()

    if len(data) < 2 or data["group"].nunique() < 2:
        return None

    total_rows = len(df)
    coverage_pct = (
        len(data) / total_rows * 100.0 if total_rows > 0 else 100.0
    )

    grouped = data.groupby("group", observed=True, sort=False)["value"]

    aggregation = grouped.agg(
        count="count",
        mean="mean",
        median="median",
    )

    totals: pd.Series | None = None
    shares: pd.Series | None = None

    if additive:
        totals = grouped.sum(min_count=1)
        overall_total = float(totals.sum())

        # Una participación porcentual solo es interpretable bajo
        # condiciones conservadoras: total positivo y grupos no negativos.
        if overall_total > 0 and (totals >= 0).all():
            shares = totals / overall_total * 100

    groups = []

    for group_value, row in aggregation.iterrows():
        total = float(totals.loc[group_value]) if totals is not None else None

        share_pct = float(shares.loc[group_value]) if shares is not None else None

        groups.append(
            GroupAggregation(
                group_value=str(group_value),
                count=int(row["count"]),
                mean=float(row["mean"]),
                median=float(row["median"]),
                total=total,
                share_pct=share_pct,
            )
        )

    return CategoricalNumericAggregation(
        column_categorical=column_categorical,
        column_numeric=column_numeric,
        n_observations=len(data),
        groups=groups,
        total_rows=total_rows,
        coverage_pct=coverage_pct,
    )


def generate_categorical_numeric_aggregations(
    df: pd.DataFrame,
    profiles: list[ColumnProfile],
    *,
    max_categories: int = 20,
    additive_columns: set[str] | None = None,
) -> list[CategoricalNumericAggregation]:
    """
    Selecciona combinaciones relevantes y calcula sus agregaciones.

    Las sumas solo se habilitan para las columnas declaradas en
    additive_columns. Por defecto, no se asume que una variable
    numérica sea aditiva.
    """
    pairs = select_categorical_numeric_pairs(
        profiles,
        max_categories=max_categories,
    )

    additive_columns = additive_columns or set()
    results = []

    for categorical, numeric in pairs:
        result = calculate_categorical_numeric_aggregation(
            df,
            column_categorical=categorical,
            column_numeric=numeric,
            additive=numeric in additive_columns,
        )

        if result is not None:
            results.append(result)

    return results
