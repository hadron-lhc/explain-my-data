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

from src.aggregations import (
    CategoricalNumericAggregation,
    generate_categorical_numeric_aggregations,
    select_categorical_numeric_pairs,
)
from src.cleaning import (
    CleaningRecommendation,
    recommend_cleaning_actions,
)
from src.comparative_insights import generate_comparative_insights
from src.config import (
    BAR_MAX_CATEGORIES,
    INSIGHT_CHARTS_MAX_EXTRA,
    SCATTER_LIGHT_SAMPLING_MAX,
    SCATTER_LIGHT_SAMPLING_POINTS,
    SCATTER_MAX_POINTS,
    SCATTER_NO_SAMPLING_MAX,
)
from src.curation import curate_insights
from src.distributions import (
    NumericDistribution,
    calculate_numeric_distributions,
)
from src.equivalence import (
    detect_equivalences,
    equivalence_card,
    filter_categorical_numeric_pairs,
    filter_insights,
    filter_relationships,
)
from src.inference import (
    CategoricalNumericInference,
    generate_inference_insights,
    run_inference,
)
from src.insights import (
    Insight,
    apply_insight_budget,
    generate_aggregation_insights,
    generate_distribution_insights,
    generate_executive_summary,
    generate_pattern_insights,
    generate_quality_insights,
    generate_relationship_insights,
    generate_structure_insights,
)
from src.ml import MLRecommendation, build_ml_recommendation, generate_ml_insights
from src.patterns import (
    GroupDifferencePattern,
    NumericOutlierPattern,
    detect_group_differences,
    detect_numeric_outliers,
)
from src.planner import (
    AnalysisCandidate,
    VisualizationPlan,
    plan_categorical_categorical_analysis,
    plan_categorical_numeric_analysis,
    plan_numeric_numeric_analysis,
    select_categorical_categorical_relationships,
    select_categorical_numeric_relationships,
    select_numeric_numeric_relationships,
)
from src.preparation import (
    prepare_categorical_series,
    prepare_numeric_series,
)
from src.profiling import (
    CATEGORICAL_LIKE_ROLES,
    DatasetProfile,
    profile_dataset,
)
from src.quality import (
    QualityReport,
    quality_dataset,
)
from src.relationships import (
    CategoricalCategoricalRelationship,
    CategoricalNumericRelationship,
    NumericRelationship,
    calculate_categorical_categorical_relationship,
    calculate_categorical_numeric_relationship,
    calculate_numeric_relationship,
)
from src.statistics import (
    CategoricalStatistics,
    NumericStatistics,
    calculate_categorical_statistics,
    calculate_numeric_statistics,
)
from src.temporal import (
    TemporalSeries,
    analyze_temporal_series,
    generate_temporal_insights,
    select_temporal_pairs,
)
from src.visualizations import (
    BarPlotData,
    BoxPlotData,
    HeatmapPlotData,
    LinePlotData,
    ScatterPlotData,
    build_bar_plot,
    build_boxplot,
    build_contingency_heatmap,
    build_scatter_plot,
)


@dataclass
class AnalysisReport:
    """Resultado completo de un análisis."""

    profile: DatasetProfile
    quality: QualityReport
    numeric_statistics: list[NumericStatistics] = field(default_factory=list)
    categorical_statistics: list[CategoricalStatistics] = field(default_factory=list)
    numeric_distributions: list[NumericDistribution] = field(default_factory=list)
    numeric_relationships: list[NumericRelationship] = field(default_factory=list)
    categorical_numeric_relationships: list[CategoricalNumericRelationship] = field(
        default_factory=list
    )
    categorical_categorical_relationships: list[CategoricalCategoricalRelationship] = (
        field(default_factory=list)
    )
    analysis_candidates: list[AnalysisCandidate] = field(default_factory=list)
    quality_insights: list[Insight] = field(default_factory=list)
    relationship_insights: list[Insight] = field(default_factory=list)
    distribution_insights: list[Insight] = field(default_factory=list)
    pattern_insights: list[Insight] = field(default_factory=list)
    structure_insights: list[Insight] = field(default_factory=list)
    aggregation_insights: list[Insight] = field(default_factory=list)
    comparative_insights: list[Insight] = field(default_factory=list)
    temporal_insights: list[Insight] = field(default_factory=list)
    inference_insights: list[Insight] = field(default_factory=list)
    ml_insights: list[Insight] = field(default_factory=list)
    executive_summary: list[Insight] = field(default_factory=list)
    scatter_plots: list[ScatterPlotData] = field(default_factory=list)
    bar_plots: list[BarPlotData] = field(default_factory=list)
    boxplots: list[BoxPlotData] = field(default_factory=list)
    heatmaps: list[HeatmapPlotData] = field(default_factory=list)
    line_plots: list[LinePlotData] = field(default_factory=list)

    aggregations: list[CategoricalNumericAggregation] = field(default_factory=list)
    temporal_series: list[TemporalSeries] = field(default_factory=list)
    inference_tests: list[CategoricalNumericInference] = field(default_factory=list)
    cleaning_recommendations: list[CleaningRecommendation] = field(
        default_factory=list
    )
    ml_recommendation: MLRecommendation | None = None
    group_difference_patterns: list[GroupDifferencePattern] = field(
        default_factory=list
    )
    numeric_outlier_patterns: list[NumericOutlierPattern] = field(default_factory=list)


def _insight_scatter_plan(n_rows: int) -> VisualizationPlan:
    if n_rows <= SCATTER_NO_SAMPLING_MAX:
        return VisualizationPlan(
            chart_type="scatter",
            sampling=False,
            trend_line=True,
        )

    if n_rows <= SCATTER_LIGHT_SAMPLING_MAX:
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


def _insight_bar_plan() -> VisualizationPlan:
    return VisualizationPlan(
        chart_type="ranked_bar",
        aggregation="mean",
        sort="descending",
        max_categories=BAR_MAX_CATEGORIES,
    )


def _has_scatter(
    scatter_plots: list[ScatterPlotData],
    column_x: str,
    column_y: str,
) -> bool:
    return any(
        (plot.column_x == column_x and plot.column_y == column_y)
        or (plot.column_x == column_y and plot.column_y == column_x)
        for plot in scatter_plots
    )


def _has_heatmap(
    heatmaps: list[HeatmapPlotData],
    column_x: str,
    column_y: str,
) -> bool:
    return any(
        (plot.column_x == column_x and plot.column_y == column_y)
        or (plot.column_x == column_y and plot.column_y == column_x)
        for plot in heatmaps
    )


def _has_bar(
    bar_plots: list[BarPlotData],
    column_categorical: str,
    column_numeric: str,
) -> bool:
    return any(
        plot.column_categorical == column_categorical
        and plot.column_numeric == column_numeric
        for plot in bar_plots
    )


def _ensure_insight_charts(
    df: pd.DataFrame,
    prepared_columns: dict[str, pd.Series],
    insights: list[Insight],
    scatter_plots: list[ScatterPlotData],
    bar_plots: list[BarPlotData],
    heatmaps: list[HeatmapPlotData],
) -> None:
    """
    Construye on-demand los gráficos que los insights esperan.

    Las relaciones seleccionadas ya tienen sus plots. Los insights
    nulos y los patrones referencian pares que quedaron fuera de la
    selección: sin este paso, su chart_id no resuelve a nada.
    """

    added = 0
    seen: set[str] = set()

    for insight in insights:
        if added >= INSIGHT_CHARTS_MAX_EXTRA:
            break

        chart_id = insight.chart_id

        if chart_id is None or chart_id in seen:
            continue

        seen.add(chart_id)

        parts = chart_id.split(":")
        kind = parts[0]

        if kind == "scatter" and len(parts) >= 3:
            column_x, column_y = parts[1], parts[2]

            if (
                column_x not in prepared_columns
                or column_y not in prepared_columns
            ):
                continue

            if _has_scatter(scatter_plots, column_x, column_y):
                continue

            plot = build_scatter_plot(
                prepared_columns,
                column_x,
                column_y,
                plan=_insight_scatter_plan(len(df)),
            )

            if plot is not None:
                scatter_plots.append(plot)
                added += 1

        elif kind == "catnum" and len(parts) >= 3:
            column_categorical, column_numeric = parts[1], parts[2]

            if (
                column_categorical not in prepared_columns
                or column_numeric not in prepared_columns
            ):
                continue

            if _has_bar(bar_plots, column_categorical, column_numeric):
                continue

            plot = build_bar_plot(
                prepared_columns,
                column_categorical,
                column_numeric,
                plan=_insight_bar_plan(),
            )

            if plot is not None:
                bar_plots.append(plot)
                added += 1

        elif kind == "catcat" and len(parts) >= 3:
            column_x, column_y = parts[1], parts[2]

            if (
                column_x not in prepared_columns
                or column_y not in prepared_columns
            ):
                continue

            if _has_heatmap(heatmaps, column_x, column_y):
                continue

            plot = build_contingency_heatmap(
                prepared_columns,
                column_x,
                column_y,
            )

            if plot is not None:
                heatmaps.append(plot)
                added += 1


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

    quality_insights = generate_quality_insights(quality)

    # Recomendaciones de limpieza (sugerencias, nunca auto-aplicadas).
    cleaning_recommendations = recommend_cleaning_actions(quality, profile)

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

        elif column.role in CATEGORICAL_LIKE_ROLES:
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

        elif column.role in CATEGORICAL_LIKE_ROLES:
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

    numeric_distributions = calculate_numeric_distributions(
        df,
        numeric_columns,
    )

    distribution_insights = generate_distribution_insights(numeric_distributions)

    categorical_columns = [
        column.name
        for column in profile.columns
        if column.role in CATEGORICAL_LIKE_ROLES and column.name in prepared_columns
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

    categorical_categorical_relationships: list[CategoricalCategoricalRelationship] = []

    for i, column_x in enumerate(categorical_columns):
        for column_y in categorical_columns[i + 1 :]:
            relationship = calculate_categorical_categorical_relationship(
                prepared_columns[column_x],
                prepared_columns[column_y],
                column_x,
                column_y,
            )

            if relationship is not None:
                categorical_categorical_relationships.append(relationship)

    # ---------------------------------------------------------
    # 5b. EQUIVALENCIA (P1): detecta redundancia y deriva
    # ---------------------------------------------------------

    profiles_by_name = {column.name: column for column in profile.columns}
    roles_by_name = {column.name: column.role for column in profile.columns}

    equivalence_map = detect_equivalences(
        profiles=roles_by_name,
        prepared_columns=prepared_columns,
        cat_cat=categorical_categorical_relationships,
        cat_num=categorical_numeric_relationships,
    )

    # Preferencia fuente > derivada: no analizar columnas redundantes.
    numeric_columns = [
        name
        for name in numeric_columns
        if name not in equivalence_map.duplicate_columns
    ]
    categorical_columns = [
        name
        for name in categorical_columns
        if name not in equivalence_map.duplicate_columns
    ]
    numeric_relationships = [
        relationship
        for relationship in numeric_relationships
        if relationship.column_x not in equivalence_map.duplicate_columns
        and relationship.column_y not in equivalence_map.duplicate_columns
    ]
    categorical_numeric_relationships = filter_relationships(
        categorical_numeric_relationships,
        equivalence_map,
    )
    categorical_categorical_relationships = [
        relationship
        for relationship in categorical_categorical_relationships
        if relationship.column_x not in equivalence_map.duplicate_columns
        and relationship.column_y not in equivalence_map.duplicate_columns
    ]
    # Evita tarjetas de dominancia de columnas plegadas.
    categorical_statistics = [
        statistics
        for statistics in categorical_statistics
        if statistics.column_name not in equivalence_map.duplicate_columns
    ]

    # ---------------------------------------------------------
    # 6. AGGREGATIONS AND PATTERNS
    # ---------------------------------------------------------

    aggregations = generate_categorical_numeric_aggregations(
        df,
        profile.columns,
    )
    # Pliegue de redundancia: sin esto, los agregados y sus insights
    # comparativos resucitarían columnas derivadas (p.ej. class, who).
    aggregations = [
        aggregation
        for aggregation in aggregations
        if aggregation.column_categorical
        not in equivalence_map.duplicate_columns
        and aggregation.column_numeric
        not in equivalence_map.duplicate_columns
    ]

    # Reutilizamos los pares seleccionados para mantener coherencia
    # entre las agregaciones y la detección de diferencias entre grupos.
    categorical_numeric_pairs = filter_categorical_numeric_pairs(
        select_categorical_numeric_pairs(
            profile.columns,
        ),
        equivalence_map,
    )

    group_difference_patterns: list[GroupDifferencePattern] = []

    for categorical_column, numeric_column in categorical_numeric_pairs:
        patterns = detect_group_differences(
            df,
            categorical_column,
            numeric_column,
        )
        group_difference_patterns.extend(patterns)

    # El perfilado determina qué columnas numéricas son candidatas.
    # Los identificadores no deben analizarse como variables de negocio.
    numeric_columns_for_patterns = [
        column.name
        for column in profile.columns
        if column.role == "numeric" and column.name in prepared_columns
    ]

    # Validación contextual (P3): el coverage por par cat-num (una
    # relación con coverage bajo no es un hallazgo sobre el dataset
    # entero) se calcula una vez y se comparte con los patrones.
    coverage_by_pair = {
        (relationship.column_categorical, relationship.column_numeric): (
            relationship.coverage_pct
        )
        for relationship in categorical_numeric_relationships
    }

    numeric_outlier_patterns = detect_numeric_outliers(
        df,
        numeric_columns_for_patterns,
    )

    pattern_insights = generate_pattern_insights(
        group_difference_patterns,
        numeric_outlier_patterns,
        coverage_by_pair=coverage_by_pair,
    )

    structure_insights = generate_structure_insights(profile)

    roles = {column.name: column.role for column in profile.columns}

    aggregation_insights = generate_aggregation_insights(aggregations, roles)

    # Los comparativos usan las listas COMPLETAS de relaciones:
    # los hallazgos nulos viven justamente por debajo de los umbrales
    # de selección.
    comparative_insights = generate_comparative_insights(
        aggregations,
        categorical_statistics,
        numeric_relationships,
        categorical_numeric_relationships,
        categorical_categorical_relationships,
        roles,
    )

    # ---------------------------------------------------------
    # 6b. TEMPORAL (columnas datetime → series + insights)
    # ---------------------------------------------------------

    datetime_columns = [
        column.name for column in profile.columns if column.role == "datetime"
    ]
    numeric_columns = [
        column.name for column in profile.columns if column.role == "numeric"
    ]

    temporal_pairs = select_temporal_pairs(
        df,
        datetime_columns,
        numeric_columns,
    )

    temporal_series: list[TemporalSeries] = []
    temporal_insights: list[Insight] = []
    line_plots: list[LinePlotData] = []

    for date_column, value_column in temporal_pairs:
        series = analyze_temporal_series(df, date_column, value_column)

        if series is None:
            continue

        temporal_series.append(series)
        temporal_insights.extend(generate_temporal_insights(series))
        line_plots.append(
            LinePlotData(
                column_date=date_column,
                column_value=value_column,
                periods=series.periods,
                means=series.means,
                rolling_mean=series.rolling_mean,
                rolling_window=series.rolling_window,
            )
        )

    # ---------------------------------------------------------
    # 7. PLANNING
    # ---------------------------------------------------------

    selected_numeric_relationships = select_numeric_numeric_relationships(
        numeric_relationships,
    )

    selected_categorical_numeric_relationships = (
        select_categorical_numeric_relationships(
            categorical_numeric_relationships,
        )
    )

    selected_categorical_categorical_relationships = (
        select_categorical_categorical_relationships(
            categorical_categorical_relationships,
        )
    )

    relationship_insights = generate_relationship_insights(
        selected_numeric_relationships,
        selected_categorical_numeric_relationships,
        selected_categorical_categorical_relationships,
    )

    # Un hallazgo = una tarjeta: cap de patrones por par y dedupe
    # relationship↔pattern (src/curation.py).
    relationship_insights, pattern_insights = curate_insights(
        relationship_insights,
        pattern_insights,
    )

    # Plegado de redundancia: descarta las tarjetas que repiten la
    # misma información por columnas equivalentes/derivadas y emite
    # UNA tarjeta de equivalencia (src/equivalence.py).
    equivalence_insight = equivalence_card(equivalence_map.relations)

    if equivalence_insight is not None:
        structure_insights.append(equivalence_insight)

    for insight_column in equivalence_map.derived:
        structure_insights.append(
            Insight(
                title=f"{insight_column.column} looks derived from other columns",
                message=(
                    f"{insight_column.column} appears to be derived from "
                    f"{', '.join(insight_column.derived_from)} "
                    f"({insight_column.evidence})."
                ),
                severity="info",
                category="structure",
                columns=[insight_column.column, *insight_column.derived_from],
                evidence={},
            )
        )

    relationship_insights = filter_insights(relationship_insights, equivalence_map)
    pattern_insights = filter_insights(pattern_insights, equivalence_map)
    aggregation_insights = filter_insights(aggregation_insights, equivalence_map)
    comparative_insights = filter_insights(comparative_insights, equivalence_map)
    distribution_insights = filter_insights(distribution_insights, equivalence_map)
    temporal_insights = filter_insights(temporal_insights, equivalence_map)

    # ---------------------------------------------------------
    # 7b. INFERENCIA (significancia + tamaño de efecto, cat-num)
    # ---------------------------------------------------------

    inference_tests = run_inference(df, categorical_numeric_relationships)
    inference_insights = filter_insights(
        generate_inference_insights(inference_tests),
        equivalence_map,
    )

    # ---------------------------------------------------------
    # 7c. ML RECOMMENDATION (target + modelos, sin entrenar)
    # ---------------------------------------------------------

    ml_recommendation = build_ml_recommendation(
        profile,
        df,
        numeric_relationships,
        categorical_numeric_relationships,
        categorical_categorical_relationships,
    )
    ml_insights = filter_insights(
        generate_ml_insights(ml_recommendation),
        equivalence_map,
    )

    analysis_candidates = [
        plan_numeric_numeric_analysis(
            relationship,
            profiles_by_name[relationship.column_x],
            profiles_by_name[relationship.column_y],
        )
        for relationship in selected_numeric_relationships
    ]

    analysis_candidates.extend(
        plan_categorical_numeric_analysis(relationship)
        for relationship in selected_categorical_numeric_relationships
    )

    analysis_candidates.extend(
        plan_categorical_categorical_analysis(relationship)
        for relationship in selected_categorical_categorical_relationships
    )

    # ---------------------------------------------------------
    # 8. VISUALIZATION
    # ---------------------------------------------------------

    scatter_plots: list[ScatterPlotData] = []
    bar_plots: list[BarPlotData] = []
    boxplots: list[BoxPlotData] = []
    heatmaps: list[HeatmapPlotData] = []

    for candidate in analysis_candidates:
        if candidate.relationship_type == "numeric_numeric":
            plan = candidate.visualization_plans[0]

            if plan.chart_type == "scatter":
                scatter_plot = build_scatter_plot(
                    df,
                    candidate.column_x,
                    candidate.column_y,
                    plan=plan,
                )

                if scatter_plot is not None:
                    scatter_plots.append(scatter_plot)

            elif plan.chart_type == "ranked_bar":
                group_by = plan.group_by

                if group_by == candidate.column_x:
                    numeric_column = candidate.column_y
                else:
                    numeric_column = candidate.column_x

                bar_plot = build_bar_plot(
                    df,
                    group_by,
                    numeric_column,
                    plan=plan,
                )

                if bar_plot is not None:
                    bar_plots.append(bar_plot)

            elif plan.chart_type == "boxplot":
                group_by = plan.group_by

                if group_by == candidate.column_x:
                    numeric_column = candidate.column_y
                else:
                    numeric_column = candidate.column_x

                boxplot = build_boxplot(
                    df,
                    group_by,
                    numeric_column,
                )

                if boxplot is not None:
                    boxplots.append(boxplot)

            elif plan.chart_type == "grouped_comparison":
                heatmap = build_contingency_heatmap(
                    df,
                    candidate.column_x,
                    candidate.column_y,
                )

                if heatmap is not None:
                    heatmaps.append(heatmap)

        elif candidate.relationship_type == "categorical_numeric":
            for plan in candidate.visualization_plans:
                if plan.chart_type == "ranked_bar":
                    bar_plot = build_bar_plot(
                        prepared_columns,
                        candidate.column_x,
                        candidate.column_y,
                        plan=plan,
                    )

                    if bar_plot is not None:
                        bar_plots.append(bar_plot)

                elif plan.chart_type == "boxplot":
                    boxplot = build_boxplot(
                        prepared_columns,
                        candidate.column_x,
                        candidate.column_y,
                    )

                    if boxplot is not None:
                        boxplots.append(boxplot)

        elif candidate.relationship_type == "categorical_categorical":
            heatmap = build_contingency_heatmap(
                prepared_columns,
                candidate.column_x,
                candidate.column_y,
            )

            if heatmap is not None:
                heatmaps.append(heatmap)

    # ---------------------------------------------------------
    # 9. CHARTS FOR INSIGHTS
    # ---------------------------------------------------------

    _ensure_insight_charts(
        df,
        prepared_columns,
        relationship_insights
        + comparative_insights
        + pattern_insights
        + aggregation_insights
        + distribution_insights,
        scatter_plots,
        bar_plots,
        heatmaps,
    )

    # Presupuesto por familia (P4): recorta a INSIGHT_BUDGET_TOTAL
    # antes de construir el resumen, para que el informe se lea como
    # un resumen y no como un vertedero de tarjetas.
    budgeted = apply_insight_budget(
        {
            "quality_insights": quality_insights,
            "relationship_insights": relationship_insights,
            "distribution_insights": distribution_insights,
            "pattern_insights": pattern_insights,
            "structure_insights": structure_insights,
            "aggregation_insights": aggregation_insights,
            "comparative_insights": comparative_insights,
            "temporal_insights": temporal_insights,
            "inference_insights": inference_insights,
            "ml_insights": ml_insights,
        }
    )

    quality_insights = budgeted.get("quality_insights", [])
    relationship_insights = budgeted.get("relationship_insights", [])
    distribution_insights = budgeted.get("distribution_insights", [])
    pattern_insights = budgeted.get("pattern_insights", [])
    structure_insights = budgeted.get("structure_insights", [])
    aggregation_insights = budgeted.get("aggregation_insights", [])
    comparative_insights = budgeted.get("comparative_insights", [])
    temporal_insights = budgeted.get("temporal_insights", [])
    inference_insights = budgeted.get("inference_insights", [])
    ml_insights = budgeted.get("ml_insights", [])

    executive_summary = generate_executive_summary(
        quality_insights
        + relationship_insights
        + distribution_insights
        + pattern_insights
        + structure_insights
        + aggregation_insights
        + comparative_insights
        + temporal_insights
        + inference_insights
        + ml_insights,
        n_rows=profile.n_rows,
        n_columns=profile.n_columns,
    )

    return AnalysisReport(
        profile=profile,
        quality=quality,
        numeric_statistics=numeric_statistics,
        categorical_statistics=categorical_statistics,
        numeric_distributions=numeric_distributions,
        numeric_relationships=numeric_relationships,
        categorical_numeric_relationships=categorical_numeric_relationships,
        categorical_categorical_relationships=categorical_categorical_relationships,
        analysis_candidates=analysis_candidates,
        quality_insights=quality_insights,
        relationship_insights=relationship_insights,
        distribution_insights=distribution_insights,
        pattern_insights=pattern_insights,
        structure_insights=structure_insights,
        aggregation_insights=aggregation_insights,
        comparative_insights=comparative_insights,
        temporal_insights=temporal_insights,
        inference_insights=inference_insights,
        ml_insights=ml_insights,
        executive_summary=executive_summary,
        scatter_plots=scatter_plots,
        bar_plots=bar_plots,
        boxplots=boxplots,
        heatmaps=heatmaps,
        line_plots=line_plots,
        temporal_series=temporal_series,
        inference_tests=inference_tests,
        cleaning_recommendations=cleaning_recommendations,
        ml_recommendation=ml_recommendation,
        aggregations=aggregations,
        group_difference_patterns=group_difference_patterns,
        numeric_outlier_patterns=numeric_outlier_patterns,
    )
