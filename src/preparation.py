"""
preparation.py

Prepara representaciones temporales de los datos para análisis.

Importante:
    Este módulo NO modifica el DataFrame original.
"""

from __future__ import annotations

import pandas as pd

from src.profiling import ColumnProfile
from src.quality import ColumnQuality


_NUMERIC_CONVERSION_THRESHOLD = 0.90


def prepare_numeric_series(
    series: pd.Series,
    profile: ColumnProfile,
    quality: ColumnQuality,
) -> pd.Series | None:
    """
    Prepara una serie identificada como numérica.

    Si la serie ya es numérica, devuelve una copia.

    Si es textual pero tiene una tasa suficiente de conversión
    numérica, la convierte temporalmente a numeric.

    Retorna None si la columna no es numérica o no puede
    convertirse de forma suficientemente confiable.
    """

    if profile.role != "numeric":
        return None

    if pd.api.types.is_numeric_dtype(series):
        return series.copy()

    if quality.conversion_rate is None:
        return None

    if quality.conversion_rate < _NUMERIC_CONVERSION_THRESHOLD:
        return None

    return pd.to_numeric(
        series,
        errors="coerce",
    )


def prepare_categorical_series(
    series: pd.Series,
    profile: ColumnProfile,
) -> pd.Series | None:
    """
    Prepara una serie identificada como categórica.

    Normaliza espacios y capitalización para evitar que
    diferencias de formato sean interpretadas como categorías
    diferentes.

    Retorna una representación temporal con dtype 'category'.

    No modifica la serie original.
    """

    if profile.role != "categorical":
        return None

    return series.astype("string").str.strip().str.lower().astype("category")
