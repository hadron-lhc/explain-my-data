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
from src.preparation import (
    prepare_categorical_series,
    prepare_numeric_series,
)
from src.relationships import (
    NumericRelationship,
    CategoricalNumericRelationship,
    calculate_numeric_relationship,
    calculate_categorical_numeric_relationship,
)
from src.statistics import (
    CategoricalStatistics,
    NumericStatistics,
    calculate_categorical_statistics,
    calculate_numeric_statistics,
)
from src.planner import (
    AnalysisCandidate,
    plan_categorical_numeric_analysis,
    plan_numeric_numeric_analysis,
    select_categorical_numeric_relationships,
    select_numeric_numeric_relationships,
)

from src.visualizations import ScatterPlotData, build_scatter_plot


@dataclass
class AnalysisReport:
    """Resultado completo de un análisis."""

    profile: DatasetProfile

    quality: QualityReport

    numeric_statistics: list[NumericStatistics] = field(default_factory=list)

    categorical_statistics: list[CategoricalStatistics] = field(default_factory=list)

    numeric_relationships: list[NumericRelationship] = field(default_factory=list)

    categorical_numeric_relationships: list[CategoricalNumericRelationship] = field(
        default_factory=list
    )
    analysis_candidates: list[AnalysisCandidate] = field(default_factory=list)

    scatter_plots: list[ScatterPlotData] = field(default_factory=list)


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
    # 3. PREPARATION
    # ---------------------------------------------------------

    quality_by_column = {column.column_name: column for column in quality.columns}

    prepared_columns: dict[str, pd.Series] = {}

    for column in profile.columns:
        series = df[column.name]

        if column.role == "numeric":
            prepared = prepare_numeric_series(
                series,
                column,
                quality_by_column[column.name],
            )

        elif column.role == "categorical":
            prepared = prepare_categorical_series(
                series,
                column,
            )

        else:
            prepared = None

        if prepared is not None:
            prepared_columns[column.name] = prepared

    # ---------------------------------------------------------
    # 4. STATISTICS
    # ---------------------------------------------------------

    numeric_statistics: list[NumericStatistics] = []
    categorical_statistics: list[CategoricalStatistics] = []

    for column in profile.columns:
        series = prepared_columns.get(column.name)

        if series is None:
            continue

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
    # 5. RELATIONSHIPS
    # ---------------------------------------------------------

    numeric_columns = [
        column.name
        for column in profile.columns
        if column.role == "numeric" and column.name in prepared_columns
    ]

    categorical_columns = [
        column.name
        for column in profile.columns
        if column.role == "categorical" and column.name in prepared_columns
    ]

    numeric_relationships: list[NumericRelationship] = []

    for i, column_x in enumerate(numeric_columns):
        for column_y in numeric_columns[i + 1 :]:
            relationship = calculate_numeric_relationship(
                prepared_columns[column_x],
                prepared_columns[column_y],
                column_x,
                column_y,
            )

            if relationship is not None:
                numeric_relationships.append(relationship)

    categorical_numeric_relationships: list[CategoricalNumericRelationship] = []

    for column_categorical in categorical_columns:
        for column_numeric in numeric_columns:
            relationship = calculate_categorical_numeric_relationship(
                prepared_columns[column_numeric],
                prepared_columns[column_categorical],
                column_numeric,
                column_categorical,
            )

            if relationship is not None:
                categorical_numeric_relationships.append(relationship)

    # ---------------------------------------------------------
    # 6. PLANNING
    # ---------------------------------------------------------

    selected_numeric_relationships = select_numeric_numeric_relationships(
        numeric_relationships,
    )

    selected_categorical_numeric_relationships = (
        select_categorical_numeric_relationships(
            categorical_numeric_relationships,
        )
    )

    analysis_candidates = [
        plan_numeric_numeric_analysis(relationship)
        for relationship in selected_numeric_relationships
    ]

    analysis_candidates.extend(
        plan_categorical_numeric_analysis(relationship)
        for relationship in selected_categorical_numeric_relationships
    )

    scatter_plots = []

    for relationship in selected_numeric_relationships:
        scatter_plot = build_scatter_plot(
            df,
            relationship.column_x,
            relationship.column_y,
        )

        if scatter_plot is not None:
            scatter_plots.append(scatter_plot)

    return AnalysisReport(
        profile=profile,
        quality=quality,
        numeric_statistics=numeric_statistics,
        categorical_statistics=categorical_statistics,
        numeric_relationships=numeric_relationships,
        categorical_numeric_relationships=categorical_numeric_relationships,
        analysis_candidates=analysis_candidates,
        scatter_plots=scatter_plots,
    )
