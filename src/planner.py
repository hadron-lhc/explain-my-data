"""
planner.py

Decide qué relaciones merecen análisis adicional y
qué métricas/visualizaciones son apropiadas.

Este módulo NO calcula relaciones ni estadísticas.
Consume resultados ya calculados por otros módulos.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.relationships import CategoricalNumericRelationship, NumericRelationship

import numpy as np


@dataclass
class AnalysisCandidate:
    column_x: str
    column_y: str
    relationship_type: str
    relevance_score: float
    recommended_metrics: list[str]
    recommended_visualizations: list[str]


def recommend_categorical_numeric_visualizations(
    relationship: CategoricalNumericRelationship,
) -> list[str]:
    """
    Recomienda visualizaciones para una relación categórica → numérica.

    n_groups <= 5
        → bar

    n_groups > 5
        → boxplot

    hay outliers
        → boxplot
    """

    n_groups = relationship.n_groups

    has_outliers = any(count > 0 for count in relationship.group_outliers.values())

    visualizations = []

    if n_groups <= 5:
        visualizations.append("bar")

    if n_groups > 5 or has_outliers:
        visualizations.append("boxplot")

    return visualizations


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


def plan_categorical_numeric_analysis(
    relationship: CategoricalNumericRelationship,
) -> AnalysisCandidate:
    """
    Convierte una relación categórica → numérica en un candidato
    de análisis.

    La relevancia actual utiliza group_separation como señal.
    """

    return AnalysisCandidate(
        column_x=relationship.column_categorical,
        column_y=relationship.column_numeric,
        relationship_type="categorical_numeric",
        relevance_score=relationship.group_separation,
        recommended_metrics=recommend_categorical_numeric_metrics(relationship),
        recommended_visualizations=recommend_categorical_numeric_visualizations(
            relationship
        ),
    )


def select_categorical_numeric_relationships(
    relationships: list[CategoricalNumericRelationship],
    percentile: float = 75.0,
) -> list[CategoricalNumericRelationship]:
    """
    Selecciona relaciones categóricas → numéricas relevantes
    según su posición relativa dentro del dataset.

    Se utiliza group_separation como señal de relevancia.

    Parameters
    ----------
    relationships:
        Relaciones calculadas previamente.

    percentile:
        Percentil mínimo que debe alcanzar una relación
        para ser seleccionada.

    Returns
    -------
    list[CategoricalNumericRelationship]
        Relaciones seleccionadas.
    """

    if not relationships:
        return []

    scores = np.array([relationship.group_separation for relationship in relationships])

    threshold = np.percentile(scores, percentile)

    return [
        relationship
        for relationship in relationships
        if relationship.group_separation >= threshold
    ]


def select_numeric_numeric_relationships(
    relationships: list[NumericRelationship],
    percentile: float = 75.0,
    minimum_strength: float = 0.5,
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

    Pearson se recomienda cuando existe una asociación lineal
    relevante.

    Spearman se recomienda cuando existe una asociación
    monotónica relevante.
    """

    metrics = []

    pearson_strength = abs(relationship.pearson)
    spearman_strength = abs(relationship.spearman)

    if pearson_strength >= 0.5:
        metrics.append("pearson")

    if spearman_strength >= 0.5:
        metrics.append("spearman")

    return metrics


def recommend_numeric_numeric_visualizations(
    relationship: NumericRelationship,
) -> list[str]:
    """
    Recomienda visualizaciones para una relación numérica → numérica.
    """

    return ["scatter"]


def plan_numeric_numeric_analysis(
    relationship: NumericRelationship,
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
        recommended_metrics=recommend_numeric_numeric_metrics(relationship),
        recommended_visualizations=recommend_numeric_numeric_visualizations(
            relationship
        ),
    )


def test_planner_does_not_recommend_boxplot_without_outliers():
    relationship = CategoricalNumericRelationship(
        column_categorical="payment_method",
        column_numeric="discount",
        group_means={
            "cash": 10.0,
            "card": 11.0,
        },
        group_medians={
            "cash": 10.0,
            "card": 11.0,
        },
        group_counts={
            "cash": 50,
            "card": 40,
        },
        group_outliers={
            "cash": 0,
            "card": 0,
        },
        n_observations=90,
        group_separation=0.10,
        n_groups=2,
    )

    candidate = plan_categorical_numeric_analysis(relationship)

    assert candidate.recommended_visualizations == ["bar", "boxplot"]
