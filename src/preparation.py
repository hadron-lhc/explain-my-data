"""
preparation.py

Prepara representaciones temporales de los datos para análisis.

Nunca modifica el DataFrame original.
"""

from __future__ import annotations

import pandas as pd

from src.profiling import ColumnProfile
from src.quality import ColumnQuality


def prepare_numeric_series(
    series: pd.Series,
    profile: ColumnProfile,
    quality: ColumnQuality,
) -> pd.Series | None:
    """
    Prepara una serie que fue identificada como numérica.

    Si la columna ya es numérica, devuelve una copia.

    Si es textual, solo intenta convertirla cuando
    quality indica una tasa de conversión >= 90%.
    """

    if profile.role != "numeric":
        return None

    if pd.api.types.is_numeric_dtype(series):
        return series.copy()

    if quality.conversion_rate is None:
        return None

    if quality.conversion_rate < 0.90:
        return None

    return pd.to_numeric(
        series,
        errors="coerce",
    )
