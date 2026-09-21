"""
analysis.py

Orquesta el análisis de un dataset.

Este módulo conecta los diferentes componentes de
Explain My Data sin implementar la lógica específica
de cada análisis.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.profiling import (
    DatasetProfile,
    profile_dataset,
)
from src.quality import (
    QualityReport,
    quality_dataset,
)
from src.relationships import (
    NumericRelationship,
    calculate_numeric_relationship,
)
from src.statistics import (
    CategoricalStatistics,
    NumericStatistics,
    calculate_categorical_statistics,
    calculate_numeric_statistics,
)


@dataclass
class AnalysisReport:
    """Resultado completo de un análisis."""

    profile: DatasetProfile

    quality: QualityReport

    numeric_statistics: list[NumericStatistics] = field(default_factory=list)

    categorical_statistics: list[CategoricalStatistics] = field(default_factory=list)

    numeric_relationships: list[NumericRelationship] = field(default_factory=list)


def analyze_dataset(
    df: pd.DataFrame,
) -> AnalysisReport:
    """
    Ejecuta el pipeline completo de análisis.

    Orden:

        profiling
            ↓
        quality
            ↓
        statistics
            ↓
        relationships

    El DataFrame original nunca se modifica.
    """

    # ---------------------------------------------------------
    # 1. PROFILING
    # ---------------------------------------------------------

    profile = profile_dataset(df)

    # ---------------------------------------------------------
    # 2. QUALITY
    # ---------------------------------------------------------

    quality = quality_dataset(
        df,
        profile,
    )

    # ---------------------------------------------------------
    # 3. STATISTICS
    # ---------------------------------------------------------

    numeric_statistics: list[NumericStatistics] = []
    categorical_statistics: list[CategoricalStatistics] = []

    for column in profile.columns:
        series = df[column.name]

        if column.role == "numeric":
            statistics = calculate_numeric_statistics(
                series,
                column.name,
            )

            if statistics is not None:
                numeric_statistics.append(statistics)

        elif column.role == "categorical":
            statistics = calculate_categorical_statistics(
                series,
                column.name,
            )

            if statistics is not None:
                categorical_statistics.append(statistics)

    # ---------------------------------------------------------
    # 4. RELATIONSHIPS
    # ---------------------------------------------------------

    numeric_columns = [
        column.name for column in profile.columns if column.role == "numeric"
    ]

    numeric_relationships: list[NumericRelationship] = []

    for i, column_x in enumerate(numeric_columns):
        for column_y in numeric_columns[i + 1 :]:
            relationship = calculate_numeric_relationship(
                df[column_x],
                df[column_y],
                column_x,
                column_y,
            )

            if relationship is not None:
                numeric_relationships.append(relationship)

    return AnalysisReport(
        profile=profile,
        quality=quality,
        numeric_statistics=numeric_statistics,
        categorical_statistics=categorical_statistics,
        numeric_relationships=numeric_relationships,
    )
