"""
quality.py

Evalúa problemas y riesgos de calidad del dataset a partir
de los hechos descubiertos por profiling.py.

Distinción central:

    profiling.py → ¿Qué tengo delante?
    quality.py   → ¿Hay problemas o riesgos en esos datos?

Esta versión V1 cubre:

- valores faltantes
- posibles incompatibilidades de tipo
- duplicados
- valores más frecuentes

Todavía no intenta detectar:
- outliers
- inconsistencias semánticas
- valores inválidos según el dominio
- reglas de negocio
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import pandas as pd

from src.profiling import ColumnProfile, DatasetProfile


@dataclass
class ColumnQuality:
    """Resultados de calidad para una columna."""

    column_name: str
    role: str

    missing_count: int
    missing_percentage: float

    type_mismatch_count: int
    is_type_mismatch: bool
    conversion_rate: float | None

    unique_percentage: float

    n_outliers: int
    outlier_percentage: float

    top_values: List[Tuple[str, int]] = field(default_factory=list)


@dataclass
class QualityReport:
    """Resultados de calidad para todo el dataset."""

    n_duplicates: int
    duplicate_percentage: float

    columns: List[ColumnQuality]


def _check_numeric_conversion(
    series: pd.Series,
) -> tuple[int, float]:
    """
    Intenta convertir los valores no nulos de una serie a numérico.

    Retorna:
        - cantidad de valores que no pudieron convertirse
        - tasa de conversión

    Ejemplo:

        ["1", "2", "3"]       → (0, 1.0)
        ["1", "2", "three"]   → (1, 0.666...)
    """

    non_null = series.dropna()

    if non_null.empty:
        return 0, 0.0

    converted = pd.to_numeric(
        non_null,
        errors="coerce",
    )

    n_mismatches = int(converted.isna().sum())

    conversion_rate = float(converted.notna().mean())

    return n_mismatches, conversion_rate


def _check_type_mismatch(
    series: pd.Series,
    profile: ColumnProfile,
) -> tuple[int, float | None]:
    """
    Verifica si una columna cuyo rol fue inferido como numérico
    necesita una conversión de tipo.

    Una columna solamente se analiza aquí si:

    1. profiling.py la clasificó como numeric.
    2. Su dtype actual es textual.

    Las columnas mixtas quedan fuera por ahora.
    """

    if profile.role != "numeric":
        return 0, None

    if not pd.api.types.is_string_dtype(series):
        return 0, None

    return _check_numeric_conversion(series)


def _check_outliers(series: pd.Series) -> tuple[int, float]:
    """
    Q1 = 25%
    Q3 = 75%
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    """

    non_null = series.dropna()

    if non_null.empty:
        return 0, 0.0

    q1 = non_null.quantile(0.25)
    q3 = non_null.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = non_null[(non_null < lower_bound) | (non_null > upper_bound)]

    n_outliers = len(outliers)
    outlier_percentage = (
        round(n_outliers / len(non_null) * 100, 2) if len(non_null) else 0.0
    )

    return n_outliers, outlier_percentage


def check_column_quality(
    series: pd.Series,
    profile: ColumnProfile,
) -> ColumnQuality:
    """
    Evalúa la calidad de una columna utilizando:

    - los datos reales de la serie
    - la información obtenida previamente por profiling.py
    """

    n_missing = int(series.isna().sum())

    n_mismatches, conversion_rate = _check_type_mismatch(
        series,
        profile,
    )

    n_outliers = 0

    outlier_percentage = 0.0

    if profile.role == "numeric" and pd.api.types.is_numeric_dtype(series):
        n_outliers, outlier_percentage = _check_outliers(series)

    top_values = series.dropna().astype(str).value_counts().head(5).items()

    return ColumnQuality(
        column_name=profile.name,
        role=profile.role,
        missing_count=n_missing,
        missing_percentage=(
            round(
                n_missing / len(series) * 100,
                2,
            )
            if len(series)
            else 0.0
        ),
        type_mismatch_count=n_mismatches,
        is_type_mismatch=(conversion_rate is not None and conversion_rate >= 0.90),
        conversion_rate=conversion_rate,
        unique_percentage=profile.pct_unique,
        n_outliers=n_outliers,
        outlier_percentage=outlier_percentage,
        top_values=list(top_values),
    )


def quality_dataset(
    df: pd.DataFrame,
    profile: DatasetProfile,
) -> QualityReport:
    """
    Evalúa la calidad general del dataset utilizando
    el perfil generado previamente por profiling.py.
    """

    column_reports: List[ColumnQuality] = []

    for column_profile in profile.columns:
        series = df[column_profile.name]

        column_quality = check_column_quality(
            series,
            column_profile,
        )

        column_reports.append(column_quality)

    n_duplicates = int(df.duplicated().sum())

    duplicate_percentage = (
        round(
            n_duplicates / len(df) * 100,
            2,
        )
        if len(df)
        else 0.0
    )

    return QualityReport(
        n_duplicates=n_duplicates,
        duplicate_percentage=duplicate_percentage,
        columns=column_reports,
    )


def print_quality_report(
    report: QualityReport,
) -> None:
    """Imprime un resumen legible del reporte de calidad."""

    print("=== Dataset Quality ===")
    print(f"Duplicate rows: {report.n_duplicates} ({report.duplicate_percentage}%)")

    print()

    for column in report.columns:
        print(f"Column: {column.column_name}")
        print(f"Role: {column.role}")
        print(f"Missing: {column.missing_count} ({column.missing_percentage}%)")

        print(f"Type mismatches: {column.type_mismatch_count}")

        print(f"Conversion rate: {column.conversion_rate}")

        print(f"Is type mismatch: {column.is_type_mismatch}")

        print(f"Unique percentage: {column.unique_percentage}%")

        print(f"Top values: {column.top_values}")

        print()


if __name__ == "__main__":
    data = {
        "numeric_real": [1.0, 2.5, 3.3, 4.1, 5.0],
        "string_numeric": ["1", "2", "3", "4", "5"],
        "string_invalid": ["1", "2", "three", "4", "five"],
        "mixed": [1, "2", 3.0, "four", None],
    }

    df = pd.DataFrame(data)

    profiles = [
        # En una ejecución real estos perfiles vendrían
        # directamente de profile_dataset(df).
        ColumnProfile(
            name="numeric_real",
            dtype=str(df["numeric_real"].dtype),
            role="numeric",
            n_missing=0,
            pct_missing=0.0,
            n_unique=5,
            pct_unique=100.0,
            sample_values=df["numeric_real"].tolist(),
        ),
        ColumnProfile(
            name="string_numeric",
            dtype=str(df["string_numeric"].dtype),
            role="numeric",
            n_missing=0,
            pct_missing=0.0,
            n_unique=5,
            pct_unique=100.0,
            sample_values=df["string_numeric"].tolist(),
        ),
        ColumnProfile(
            name="string_invalid",
            dtype=str(df["string_invalid"].dtype),
            role="numeric",
            n_missing=0,
            pct_missing=0.0,
            n_unique=5,
            pct_unique=100.0,
            sample_values=df["string_invalid"].tolist(),
        ),
        ColumnProfile(
            name="mixed",
            dtype=str(df["mixed"].dtype),
            role="numeric",
            n_missing=1,
            pct_missing=20.0,
            n_unique=4,
            pct_unique=80.0,
            sample_values=df["mixed"].dropna().tolist(),
        ),
    ]

    dataset_profile = DatasetProfile(
        n_rows=len(df),
        n_columns=len(df.columns),
        n_duplicates=int(df.duplicated().sum()),
        columns=profiles,
    )

    quality_report = quality_dataset(
        df,
        dataset_profile,
    )

    print_quality_report(quality_report)
