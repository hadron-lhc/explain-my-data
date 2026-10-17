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
- categórica ↔ numérica
    - medias/medianas por grupo
    - separación de grupos
- categórica ↔ categórica
    - chi-cuadrado
    - Cramér's V
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from src.utils_stats import OUTLIER_IQR_MULTIPLIER, grouped_iqr_bounds


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

    n_groups: int

    group_outliers: dict[str, int]

    # % de las filas donde ambas columnas tienen valor, respecto al
    # total original (antes del dropna). Es el ancla de missingness
    # (P3): una relación con coverage bajo suena fuerte pero solo
    # describe un subconjunto del dataset.
    coverage_pct: float = 100.0


@dataclass
class CategoricalCategoricalRelationship:
    """Relación entre dos variables categóricas."""

    column_x: str
    column_y: str

    n_observations: int

    n_groups_x: int
    n_groups_y: int

    chi2: float
    p_value: float
    cramers_v: float

    # Celdas con el mayor residuo estandarizado (co-ocurrencias fuertes).
    top_associations: list[tuple[str, str, float]] = field(default_factory=list)


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

    frame = pd.DataFrame(
        {
            "numeric": pd.to_numeric(numeric_series, errors="coerce"),
            "categorical": categorical_series,
        }
    )

    n_total = len(frame)
    data = frame.dropna()

    if len(data) < 2:
        return None

    coverage_pct = (len(data) / n_total * 100.0) if n_total > 0 else 0.0

    grouped = data.groupby("categorical")["numeric"]

    group_means = grouped.mean().to_dict()
    group_medians = grouped.median().to_dict()
    group_counts = grouped.count().to_dict()

    n_groups = len(group_means)

    overall_std = data["numeric"].std()

    mean_range = max(group_means.values()) - min(group_means.values())

    if overall_std > 0:
        group_separation = mean_range / overall_std
    else:
        group_separation = 0.0

    lower_bound, upper_bound = grouped_iqr_bounds(
        grouped,
        OUTLIER_IQR_MULTIPLIER,
    )

    categorical_values = data["categorical"].astype("string")

    outlier_mask = (data["numeric"] < categorical_values.map(lower_bound)) | (
        data["numeric"] > categorical_values.map(upper_bound)
    )

    group_outliers = (
        data.loc[outlier_mask]
        .groupby("categorical")["numeric"]
        .count()
        .reindex(grouped.groups.keys(), fill_value=0)
        .to_dict()
    )

    return CategoricalNumericRelationship(
        column_categorical=column_categorical,
        column_numeric=column_numeric,
        group_means=group_means,
        group_medians=group_medians,
        group_counts=group_counts,
        n_observations=len(data),
        group_separation=group_separation,
        n_groups=n_groups,
        group_outliers=group_outliers,
        coverage_pct=coverage_pct,
    )


def calculate_categorical_categorical_relationship(
    x: pd.Series,
    y: pd.Series,
    column_x: str,
    column_y: str,
    *,
    max_categories: int = 20,
    min_observations: int = 30,
) -> CategoricalCategoricalRelationship | None:
    """
    Calcula la asociación entre dos variables categóricas.

    Usa chi-cuadrado sobre la tabla de contingencia y Cramér's V
    como medida de fuerza (0 = independencia, 1 = asociación total).

    top_associations contiene las celdas con mayor residuo estandarizado:
    pares de valores que co-ocurren más (o menos) de lo esperado.
    """
    if column_x == column_y:
        return None

    data = pd.DataFrame({"x": x, "y": y}).dropna()

    if len(data) < min_observations:
        return None

    contingency = pd.crosstab(data["x"], data["y"])

    n_groups_x, n_groups_y = contingency.shape

    if n_groups_x < 2 or n_groups_y < 2:
        return None

    if n_groups_x > max_categories or n_groups_y > max_categories:
        return None

    try:
        chi2, p_value, _, _ = chi2_contingency(contingency)
    except ValueError:
        return None

    n = int(contingency.to_numpy().sum())

    min_dim = min(n_groups_x, n_groups_y)

    cramers_v = float(np.sqrt(chi2 / (n * (min_dim - 1)))) if n > 0 else 0.0

    # Residuos estandarizados para interpretar qué celdas impulsan la asociación.
    observed = contingency.to_numpy(dtype=float)
    row_totals = observed.sum(axis=1, keepdims=True)
    col_totals = observed.sum(axis=0, keepdims=True)
    expected = row_totals @ col_totals / n

    with np.errstate(divide="ignore", invalid="ignore"):
        residuals = np.where(expected > 0, (observed - expected) / np.sqrt(expected), 0.0)

    flat_indices = np.argsort(np.abs(residuals).ravel())[::-1][:5]

    top_associations: list[tuple[str, str, float]] = []

    for flat_index in flat_indices:
        row_index, col_index = divmod(int(flat_index), n_groups_y)

        top_associations.append(
            (
                str(contingency.index[row_index]),
                str(contingency.columns[col_index]),
                float(residuals[row_index, col_index]),
            )
        )

    return CategoricalCategoricalRelationship(
        column_x=column_x,
        column_y=column_y,
        n_observations=n,
        n_groups_x=n_groups_x,
        n_groups_y=n_groups_y,
        chi2=float(chi2),
        p_value=float(p_value),
        cramers_v=cramers_v,
        top_associations=top_associations,
    )
