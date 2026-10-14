"""
utils_stats.py

Ayudas estadísticas compartidas.

La lógica IQR vive aquí; los umbrales canónicos están en src.config
y se re-exportan para mantener los imports existentes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    EXTREME_IQR_MULTIPLIER,
    NUMERIC_CONVERSION_THRESHOLD,
    OUTLIER_IQR_MULTIPLIER,
)

__all__ = [
    "EXTREME_IQR_MULTIPLIER",
    "NUMERIC_CONVERSION_THRESHOLD",
    "OUTLIER_IQR_MULTIPLIER",
    "finite_numeric",
    "grouped_iqr_bounds",
    "iqr_bounds",
    "iqr_outlier_mask",
]


def finite_numeric(series: pd.Series) -> pd.Series:
    """Convierte a numérico tratando faltantes e infinitos como missing."""
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.where(np.isfinite(numeric))


def iqr_bounds(
    values: pd.Series,
    multiplier: float = OUTLIER_IQR_MULTIPLIER,
) -> tuple[float, float] | None:
    """
    Calcula los límites inferior y superior con la regla del IQR.

    Retorna None si no hay observaciones válidas o si el IQR es cero
    (distribución degenerada sin outliers posibles).
    """
    numeric = finite_numeric(values).dropna()

    if numeric.empty:
        return None

    q1 = float(numeric.quantile(0.25))
    q3 = float(numeric.quantile(0.75))
    iqr = q3 - q1

    if not np.isfinite(iqr) or iqr == 0:
        return None

    return q1 - multiplier * iqr, q3 + multiplier * iqr


def grouped_iqr_bounds(
    grouped: pd.SeriesGroupBy,
    multiplier: float = OUTLIER_IQR_MULTIPLIER,
) -> tuple[pd.Series, pd.Series]:
    """Límites IQR por grupo (vectorizado). Retorna (lower, upper)."""
    q1 = grouped.quantile(0.25)
    q3 = grouped.quantile(0.75)
    iqr = q3 - q1

    return q1 - multiplier * iqr, q3 + multiplier * iqr


def iqr_outlier_mask(
    values: pd.Series,
    multiplier: float = OUTLIER_IQR_MULTIPLIER,
) -> pd.Series:
    """Máscara booleana de outliers IQR. Si el IQR es cero no hay outliers."""
    bounds = iqr_bounds(values, multiplier)

    if bounds is None:
        return pd.Series(False, index=values.index)

    lower, upper = bounds
    numeric = pd.to_numeric(values, errors="coerce")
    mask = (numeric < lower) | (numeric > upper)

    return mask.fillna(False)
