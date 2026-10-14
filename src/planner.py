"""
planner.py

Decide qué relaciones merecen análisis adicional,
qué métricas son relevantes y cómo comunicar
visualmente esas relaciones.

Este módulo NO calcula relaciones ni estadísticas.
Consume resultados ya calculados por otros módulos.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.config import (
    BAR_MAX_CATEGORIES,
    HEATMAP_MAX_CATEGORIES,
    MIN_CORRELATION_STRENGTH,
    MIN_CRAMERS_V,
    RELATIONSHIP_SELECTION_PERCENTILE,
    SCATTER_LIGHT_SAMPLING_MAX,
    SCATTER_LIGHT_SAMPLING_POINTS,
    SCATTER_MAX_POINTS,
    SCATTER_NO_SAMPLING_MAX,
)
from src.profiling import ColumnProfile
from src.relationships import (
    CategoricalCategoricalRelationship,
    CategoricalNumericRelationship,
    NumericRelationship,
)


@dataclass
class VisualizationPlan:
    """
    Describe cómo debería visualizarse una relación.

    El planner decide la intención del gráfico.
    El módulo de visualizaciones decide cómo construirlo.
    """

    chart_type: str

    aggregation: str | None = None
    sort: str | None = None
    max_categories: int | None = None

    sampling: bool = False
    max_points: int | None = None
    trend_line: bool = False

    group_by: str | None = None
    min_group_size: int | None = None


@dataclass
class AnalysisCandidate:
    column_x: str
    column_y: str
    relationship_type: str
    relevance_score: float
    recommended_metrics: list[str]

    visualization_plans: list[VisualizationPlan] = field(
        default_factory=list,
    )

    # Compatibility with the previous planner API.
    recommended_visualizations: list[str] = field(
        default_factory=list,
    )


# ---------------------------------------------------------------------------
# Categorical → numeric
# ---------------------------------------------------------------------------


def make_numeric_profile(
    name: str,
    *,
    is_discrete: bool,
) -> ColumnProfile:
    return ColumnProfile(
        name=name,
        dtype="int64" if is_discrete else "float64",
        role="numeric",
        n_missing=0,
        pct_missing=0.0,
        n_unique=5 if is_discrete else 100,
        pct_unique=5.0 if is_discrete else 100.0,
        sample_values=[1, 2, 3],
        is_discrete=is_discrete,
    )


def recommend_categorical_numeric_metrics(
    relationship: CategoricalNumericRelationship,
) -> list[str]:
    """
    Recomienda métricas para una relación categórica → numérica.
    """

    metrics = [
        "group_count",
        "group_mean",
    ]

    has_outliers = any(count > 0 for count in relationship.group_outliers.values())

    if has_outliers:
        metrics.append("group_median")
        metrics.append("group_outliers")

    return metrics


def recommend_categorical_numeric_visualizations(
    relationship: CategoricalNumericRelationship,
) -> list[str]:
    """
    Mantiene la API anterior.

    La nueva lógica de visualización vive en
    plan_categorical_numeric_visualization().

    Se conserva esta función temporalmente para no romper
    consumidores existentes mientras migramos el sistema.
    """

    n_groups = relationship.n_groups

    has_outliers = any(count > 0 for count in relationship.group_outliers.values())

    visualizations = []

    if n_groups <= 5:
        visualizations.append("bar")

    if n_groups > 5 or has_outliers:
        visualizations.append("boxplot")

    return visualizations


def plan_categorical_numeric_visualization(
    relationship: CategoricalNumericRelationship,
) -> VisualizationPlan:
    """
    Decide la representación principal para una relación
    categórica → numérica.

    La intención principal es comparar valores entre grupos.

    Por eso utilizamos un bar chart ordenado por la media,
    limitado a un número razonable de categorías.
    """

    return VisualizationPlan(
        chart_type="ranked_bar",
        aggregation="mean",
        sort="descending",
        max_categories=BAR_MAX_CATEGORIES,
    )


def plan_categorical_numeric_analysis(
    relationship: CategoricalNumericRelationship,
) -> AnalysisCandidate:
    """
    Convierte una relación categórica → numérica
    en un candidato de análisis.
    """

    visualization_plan = plan_categorical_numeric_visualization(
        relationship,
    )

    return AnalysisCandidate(
        column_x=relationship.column_categorical,
        column_y=relationship.column_numeric,
        relationship_type="categorical_numeric",
        relevance_score=relationship.group_separation,
        recommended_metrics=(
            recommend_categorical_numeric_metrics(
                relationship,
            )
        ),
        visualization_plans=[
            visualization_plan,
        ],
        # Temporary compatibility with the old API.
        recommended_visualizations=(
            recommend_categorical_numeric_visualizations(
                relationship,
            )
        ),
    )


def select_categorical_numeric_relationships(
    relationships: list[CategoricalNumericRelationship],
    percentile: float = RELATIONSHIP_SELECTION_PERCENTILE,
) -> list[CategoricalNumericRelationship]:
    """
    Selecciona relaciones categóricas → numéricas relevantes
    según su posición relativa dentro del dataset.

    Utiliza group_separation como señal de relevancia.
    """

    if not relationships:
        return []

    scores = np.array([relationship.group_separation for relationship in relationships])

    threshold = np.percentile(
        scores,
        percentile,
    )

    return [
        relationship
        for relationship in relationships
        if relationship.group_separation >= threshold
    ]


# ---------------------------------------------------------------------------
# Categorical → categorical
# ---------------------------------------------------------------------------


def select_categorical_categorical_relationships(
    relationships: list[CategoricalCategoricalRelationship],
    percentile: float = RELATIONSHIP_SELECTION_PERCENTILE,
    minimum_strength: float = MIN_CRAMERS_V,
) -> list[CategoricalCategoricalRelationship]:
    """
    Selecciona asociaciones categóricas relevantes.

    Cramér's V debe superar tanto el umbral relativo del dataset
    como una fuerza mínima absoluta (asociación media).
    """

    if not relationships:
        return []

    scores = np.array([relationship.cramers_v for relationship in relationships])

    threshold = np.percentile(scores, percentile)

    effective_threshold = max(threshold, minimum_strength)

    return [
        relationship
        for relationship in relationships
        if relationship.cramers_v >= effective_threshold
    ]


def plan_categorical_categorical_analysis(
    relationship: CategoricalCategoricalRelationship,
) -> AnalysisCandidate:
    """
    Convierte una asociación categórica ↔ categórica
    en un candidato de análisis.
    """

    return AnalysisCandidate(
        column_x=relationship.column_x,
        column_y=relationship.column_y,
        relationship_type="categorical_categorical",
        relevance_score=relationship.cramers_v,
        recommended_metrics=["chi2", "cramers_v", "top_associations"],
        visualization_plans=[
            VisualizationPlan(
                chart_type="heatmap",
                max_categories=HEATMAP_MAX_CATEGORIES,
            ),
        ],
        recommended_visualizations=["heatmap"],
    )


# ---------------------------------------------------------------------------
# Numeric → numeric
# ---------------------------------------------------------------------------


def select_numeric_numeric_relationships(
    relationships: list[NumericRelationship],
    percentile: float = RELATIONSHIP_SELECTION_PERCENTILE,
    minimum_strength: float = MIN_CORRELATION_STRENGTH,
) -> list[NumericRelationship]:
    """
    Selecciona relaciones numéricas relevantes.

    Una relación debe superar tanto el umbral relativo
    del dataset como una fuerza mínima absoluta.

    La fuerza se define como la mayor magnitud absoluta
    entre Pearson y Spearman.
    """

    if not relationships:
        return []

    scores = np.array(
        [
            max(
                abs(relationship.pearson),
                abs(relationship.spearman),
            )
            for relationship in relationships
        ]
    )

    threshold = np.percentile(
        scores,
        percentile,
    )

    effective_threshold = max(
        threshold,
        minimum_strength,
    )

    return [
        relationship
        for relationship in relationships
        if max(
            abs(relationship.pearson),
            abs(relationship.spearman),
        )
        >= effective_threshold
    ]


def recommend_numeric_numeric_metrics(
    relationship: NumericRelationship,
) -> list[str]:
    """
    Recomienda métricas para una relación numérica → numérica.

    Pearson se recomienda cuando existe una asociación
    lineal relevante.

    Spearman se recomienda cuando existe una asociación
    monotónica relevante.
    """

    metrics = []

    if abs(relationship.pearson) >= 0.5:
        metrics.append("pearson")

    if abs(relationship.spearman) >= 0.5:
        metrics.append("spearman")

    return metrics


def recommend_numeric_numeric_visualizations(
    relationship: NumericRelationship,
) -> list[str]:
    """
    Mantiene la API anterior.

    La configuración detallada de la visualización
    vive en plan_numeric_numeric_visualization().
    """
    return ["scatter"]


def plan_numeric_numeric_visualization(
    relationship: NumericRelationship,
    profile_x: ColumnProfile,
    profile_y: ColumnProfile,
) -> VisualizationPlan:
    if profile_x.is_discrete and not profile_y.is_discrete:
        chart_type = "ranked_bar" if profile_x.n_unique <= 5 else "boxplot"

        return VisualizationPlan(
            chart_type=chart_type,
            aggregation="mean",
            sort="descending",
            max_categories=BAR_MAX_CATEGORIES,
            group_by=profile_x.name,
        )

    if profile_y.is_discrete and not profile_x.is_discrete:
        chart_type = "ranked_bar" if profile_y.n_unique <= 5 else "boxplot"

        return VisualizationPlan(
            chart_type=chart_type,
            aggregation="mean",
            sort="descending",
            max_categories=BAR_MAX_CATEGORIES,
            group_by=profile_y.name,
        )

    if profile_x.is_discrete and profile_y.is_discrete:
        return VisualizationPlan(
            chart_type="grouped_comparison",
        )

    n = relationship.n_observations

    if n <= SCATTER_NO_SAMPLING_MAX:
        return VisualizationPlan(
            chart_type="scatter",
            sampling=False,
            trend_line=True,
        )

    if n <= SCATTER_LIGHT_SAMPLING_MAX:
        return VisualizationPlan(
            chart_type="scatter",
            sampling=True,
            max_points=SCATTER_LIGHT_SAMPLING_POINTS,
            trend_line=True,
        )

    return VisualizationPlan(
        chart_type="scatter",
        sampling=True,
        max_points=SCATTER_MAX_POINTS,
        trend_line=True,
    )


def plan_numeric_numeric_analysis(
    relationship: NumericRelationship,
    profile_x: ColumnProfile,
    profile_y: ColumnProfile,
) -> AnalysisCandidate:
    """
    Construye un candidato de análisis para una relación
    numérica → numérica.
    """

    relevance_score = max(
        abs(relationship.pearson),
        abs(relationship.spearman),
    )

    return AnalysisCandidate(
        column_x=relationship.column_x,
        column_y=relationship.column_y,
        relationship_type="numeric_numeric",
        relevance_score=relevance_score,
        recommended_metrics=(
            recommend_numeric_numeric_metrics(
                relationship,
            )
        ),
        visualization_plans=[
            plan_numeric_numeric_visualization(
                relationship,
                profile_x,
                profile_y,
            ),
        ],
        # Temporary compatibility with the old API.
        recommended_visualizations=(
            recommend_numeric_numeric_visualizations(
                relationship,
            )
        ),
    )
