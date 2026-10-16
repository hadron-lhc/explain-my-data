"""
insights.py

Genera insights en lenguaje natural a partir de la evidencia
cuantitativa producida por el resto del pipeline.

Todo es basado en reglas y plantillas: no se usa ningún modelo
de lenguaje ni servicio externo.

Cada insight declara:
- category: familia del hallazgo
- columns: columnas implicadas
- evidence: números crudos que respaldan el texto
- chart_id: identificador estable para enlazar con una visualización
- score: fuerza relativa del hallazgo (para ranking)
- caveats: limitaciones que el lector debe conocer
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from src.aggregations import CategoricalNumericAggregation
from src.config import (
    BOOLEAN_DOMINANT_ALERT_SHARE,
    EXECUTIVE_PER_CATEGORY_MAX,
    EXECUTIVE_SUMMARY_MAX_WARNINGS,
    EXECUTIVE_SUMMARY_TOP_N,
    MIN_GROUP_SEPARATION,
)
from src.distributions import NumericDistribution
from src.patterns import GroupDifferencePattern, NumericOutlierPattern
from src.profiling import ColumnProfile, DatasetProfile
from src.quality import ColumnQuality, QualityReport
from src.relationships import (
    CategoricalCategoricalRelationship,
    CategoricalNumericRelationship,
    NumericRelationship,
)

ASSOCIATION_CAVEAT = "Association does not imply causation."
OUTLIER_CAVEAT = "Statistical outliers are not necessarily errors."


@dataclass
class Insight:
    title: str
    message: str
    severity: str  # info | warning

    category: str = "general"
    columns: list[str] = field(default_factory=list)
    evidence: dict[str, float | int | str] = field(default_factory=dict)
    chart_id: str | None = None
    score: float = 0.0
    caveats: list[str] = field(default_factory=list)
    id: str = ""

    def __post_init__(self) -> None:
        # Identificador determinista y corto para poder ocultar/restaurar
        # insights desde la UI sin depender de posiciones de lista.
        if not self.id:
            raw = "|".join(
                (self.category, ",".join(self.columns), self.title)
            )
            self.id = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Etiquetas de grupo legibles
# ---------------------------------------------------------------------------

_BOOLEAN_POSITIVE_VALUES = frozenset({"1", "true", "yes", "y", "t", "si", "sí"})
_BOOLEAN_NEGATIVE_VALUES = frozenset({"0", "false", "no", "n", "f"})

_BOOLEAN_STEM_PREFIXES = (
    "is_",
    "has_",
    "was_",
    "were_",
    "es_",
    "tiene_",
    "fue_",
    "son_",
)
_BOOLEAN_STEM_SUFFIXES = ("_flag", "_bool", "_indicator")


def _boolean_stem(column_name: str) -> str | None:
    """
    'is_refunded' -> 'refunded'; 'has_issue_flag' -> 'issue'.

    Solo se considera stem si el nombre matchea un patrón de flag
    (prefijo/sufijo conocido); si no, devuelve None y se conserva
    el valor crudo (evita 'not codigo' para columnas arbitrarias).
    """
    lowered = column_name.strip().lower()
    matched = False

    for prefix in _BOOLEAN_STEM_PREFIXES:
        if lowered.startswith(prefix) and len(lowered) > len(prefix):
            lowered = lowered[len(prefix) :]
            matched = True
            break

    for suffix in _BOOLEAN_STEM_SUFFIXES:
        if lowered.endswith(suffix) and len(lowered) > len(suffix):
            lowered = lowered[: -len(suffix)]
            matched = True
            break

    if not matched:
        return None

    stem = lowered.replace("_", " ").strip()
    return stem or None


def humanize_group_value(
    column_name: str,
    value: object,
    roles: dict[str, str] | None,
) -> str:
    """
    Etiqueta legible para valores de grupo.

    Solo las columnas con role 'boolean' se traducen:
    is_refunded=0 -> 'not refunded'; is_refunded=1 -> 'refunded'.
    El valor crudo se conserva como fallback.
    """
    if not roles or roles.get(column_name) != "boolean":
        return str(value)

    raw = str(value).strip().lower()

    if raw not in _BOOLEAN_POSITIVE_VALUES and raw not in _BOOLEAN_NEGATIVE_VALUES:
        return str(value)

    stem = _boolean_stem(column_name)

    if not stem:
        return str(value)

    if raw in _BOOLEAN_POSITIVE_VALUES:
        return stem

    return f"not {stem}"


# ---------------------------------------------------------------------------
# Identificadores estables de visualización
# ---------------------------------------------------------------------------


def chart_id_scatter(column_x: str, column_y: str) -> str:
    return f"scatter:{column_x}:{column_y}"


def chart_id_categorical_numeric(column_categorical: str, column_numeric: str) -> str:
    return f"catnum:{column_categorical}:{column_numeric}"


def chart_id_categorical_categorical(column_x: str, column_y: str) -> str:
    return f"catcat:{column_x}:{column_y}"


def chart_id_histogram(column_name: str) -> str:
    return f"hist:{column_name}"


# ---------------------------------------------------------------------------
# Calidad
# ---------------------------------------------------------------------------


def _missing_values_insight(
    column: ColumnQuality,
) -> Insight | None:
    if column.missing_count == 0:
        return None

    return Insight(
        title=f"Missing values in {column.column_name}",
        message=(
            f"{column.column_name} contains {column.missing_count} "
            f"missing values ({column.missing_percentage:.1f}% of the column)."
        ),
        severity="warning",
        category="quality",
        columns=[column.column_name],
        evidence={
            "missing_count": column.missing_count,
            "missing_percentage": column.missing_percentage,
        },
        score=column.missing_percentage,
    )


def _type_mismatch_insight(
    column: ColumnQuality,
) -> Insight | None:
    if not column.is_type_mismatch:
        return None

    conversion_pct = column.conversion_rate * 100 if column.conversion_rate else 0.0

    return Insight(
        title=f"Possible type mismatch in {column.column_name}",
        message=(
            f"{column.column_name} is currently stored as text, but "
            f"{conversion_pct:.1f}% of its values can be "
            f"interpreted as numeric values."
        ),
        severity="warning",
        category="quality",
        columns=[column.column_name],
        evidence={
            "type_mismatch_count": column.type_mismatch_count,
            "conversion_rate": conversion_pct,
        },
        score=conversion_pct,
    )


def _invalid_values_insight(
    column: ColumnQuality,
) -> Insight | None:
    if column.n_invalid_values == 0:
        return None

    return Insight(
        title=f"Invalid values in {column.column_name}",
        message=(
            f"{column.column_name} contains {column.n_invalid_values} "
            f"non-finite numeric values "
            f"({column.invalid_percentage:.1f}% of the column). "
            f"These values may require review."
        ),
        severity="warning",
        category="quality",
        columns=[column.column_name],
        evidence={
            "n_invalid_values": column.n_invalid_values,
            "invalid_percentage": column.invalid_percentage,
        },
        score=column.invalid_percentage,
    )


def _extreme_values_insight(
    column: ColumnQuality,
) -> Insight | None:
    if column.n_extreme_values == 0:
        return None

    return Insight(
        title=f"Potentially extreme values in {column.column_name}",
        message=(
            f"{column.column_name} contains {column.n_extreme_values} "
            f"values outside the overall IQR range "
            f"({column.extreme_percentage:.1f}% of the column). "
            f"These values may be legitimate or may require review."
        ),
        severity="warning",
        category="quality",
        columns=[column.column_name],
        evidence={
            "n_extreme_values": column.n_extreme_values,
            "extreme_percentage": column.extreme_percentage,
        },
        score=column.extreme_percentage,
        caveats=[OUTLIER_CAVEAT],
    )


def _duplicates_insight(
    report: QualityReport,
) -> Insight | None:
    if report.n_duplicates == 0:
        return None

    return Insight(
        title="Duplicate rows detected",
        message=(
            f"The dataset contains {report.n_duplicates} duplicate rows "
            f"({report.duplicate_percentage:.1f}% of all rows)."
        ),
        severity="warning",
        category="quality",
        evidence={
            "n_duplicates": report.n_duplicates,
            "duplicate_percentage": report.duplicate_percentage,
        },
        score=report.duplicate_percentage,
    )


# ---------------------------------------------------------------------------
# Relaciones
# ---------------------------------------------------------------------------


def _numeric_relationship_insight(
    relationship: NumericRelationship,
) -> Insight | None:
    strength = max(
        abs(relationship.pearson),
        abs(relationship.spearman),
    )

    if strength < 0.5:
        return None

    if relationship.pearson >= 0.5:
        direction = "positive"
    elif relationship.pearson <= -0.5:
        direction = "negative"
    else:
        direction = "monotonic"

    return Insight(
        title=(
            f"Strong association between "
            f"{relationship.column_x} and {relationship.column_y}"
        ),
        message=(
            f"{relationship.column_x} and {relationship.column_y} "
            f"show a {direction} association "
            f"(Pearson: {relationship.pearson:.2f}, "
            f"Spearman: {relationship.spearman:.2f}, "
            f"n: {relationship.n_observations})."
        ),
        severity="info",
        category="relationship",
        columns=[relationship.column_x, relationship.column_y],
        evidence={
            "pearson": relationship.pearson,
            "spearman": relationship.spearman,
            "strength": strength,
            "n_observations": relationship.n_observations,
        },
        chart_id=chart_id_scatter(
            relationship.column_x,
            relationship.column_y,
        ),
        score=strength,
        caveats=[ASSOCIATION_CAVEAT],
    )


def _categorical_numeric_relationship_insight(
    relationship: CategoricalNumericRelationship,
) -> Insight | None:
    # El piso coincide con el de selección (MIN_GROUP_SEPARATION):
    # una relación seleccionada debe poder emitir su insight.
    if relationship.group_separation < MIN_GROUP_SEPARATION:
        return None

    top_group = max(relationship.group_means, key=relationship.group_means.get)
    bottom_group = min(relationship.group_means, key=relationship.group_means.get)

    return Insight(
        title=(
            f"{relationship.column_numeric} differs across "
            f"{relationship.column_categorical} groups"
        ),
        message=(
            f"Average {relationship.column_numeric} varies substantially "
            f"between {relationship.column_categorical} groups "
            f"(group separation: {relationship.group_separation:.2f}). "
            f"Highest average: {top_group} "
            f"({relationship.group_means[top_group]:.2f}); "
            f"lowest: {bottom_group} "
            f"({relationship.group_means[bottom_group]:.2f})."
        ),
        severity="info",
        category="relationship",
        columns=[relationship.column_categorical, relationship.column_numeric],
        evidence={
            "group_separation": relationship.group_separation,
            "n_groups": relationship.n_groups,
            "top_group_mean": relationship.group_means[top_group],
            "bottom_group_mean": relationship.group_means[bottom_group],
        },
        chart_id=chart_id_categorical_numeric(
            relationship.column_categorical,
            relationship.column_numeric,
        ),
        score=relationship.group_separation,
        caveats=[ASSOCIATION_CAVEAT],
    )


def _categorical_categorical_relationship_insight(
    relationship: CategoricalCategoricalRelationship,
) -> Insight | None:
    if relationship.cramers_v < 0.3:
        return None

    strongest = max(
        relationship.top_associations,
        key=lambda association: abs(association[2]),
        default=None,
    )

    message = (
        f"{relationship.column_x} and {relationship.column_y} "
        f"are associated (Cramér's V: {relationship.cramers_v:.2f}, "
        f"chi2 p-value: {relationship.p_value:.2g}, "
        f"n: {relationship.n_observations})."
    )

    if strongest is not None and abs(strongest[2]) >= 2.0:
        direction = "over-represented" if strongest[2] > 0 else "under-represented"
        message += (
            f" The combination {relationship.column_x}={strongest[0]} with "
            f"{relationship.column_y}={strongest[1]} is {direction} "
            f"compared to independence (residual: {strongest[2]:.2f})."
        )

    evidence: dict[str, float | int | str] = {
        "cramers_v": relationship.cramers_v,
        "chi2": relationship.chi2,
        "p_value": relationship.p_value,
        "n_observations": relationship.n_observations,
    }

    if strongest is not None:
        evidence["strongest_residual"] = strongest[2]

    return Insight(
        title=(
            f"Association between "
            f"{relationship.column_x} and {relationship.column_y}"
        ),
        message=message,
        severity="info",
        category="relationship",
        columns=[relationship.column_x, relationship.column_y],
        evidence=evidence,
        chart_id=chart_id_categorical_categorical(
            relationship.column_x,
            relationship.column_y,
        ),
        score=relationship.cramers_v,
        caveats=[ASSOCIATION_CAVEAT],
    )


def generate_relationship_insights(
    numeric_relationships: list[NumericRelationship],
    categorical_numeric_relationships: list[CategoricalNumericRelationship],
    categorical_categorical_relationships: list[CategoricalCategoricalRelationship]
    | None = None,
) -> list[Insight]:
    insights: list[Insight] = []

    for relationship in numeric_relationships:
        insight = _numeric_relationship_insight(relationship)

        if insight is not None:
            insights.append(insight)

    for relationship in categorical_numeric_relationships:
        insight = _categorical_numeric_relationship_insight(relationship)

        if insight is not None:
            insights.append(insight)

    for relationship in categorical_categorical_relationships or []:
        insight = _categorical_categorical_relationship_insight(relationship)

        if insight is not None:
            insights.append(insight)

    return insights


# ---------------------------------------------------------------------------
# Distribuciones
# ---------------------------------------------------------------------------


def _skewness_insight(distribution: NumericDistribution) -> Insight | None:
    skewness = distribution.skewness

    if abs(skewness) < 1.0:
        return None

    if skewness > 0:
        direction = "right-skewed (long right tail)"
        explanation = (
            "The median is likely a better representation of a typical "
            "value than the mean, and a few large values may dominate "
            "totals."
        )
    else:
        direction = "left-skewed (long left tail)"
        explanation = (
            "A few unusually low values may be pulling the mean below "
            "the typical observation."
        )

    return Insight(
        title=f"{distribution.column_name} is {direction}",
        message=(
            f"{distribution.column_name} has a strong skew "
            f"(skewness: {skewness:.2f}) across "
            f"{distribution.n_observations} observations. {explanation}"
        ),
        severity="info",
        category="distribution",
        columns=[distribution.column_name],
        evidence={
            "skewness": skewness,
            "kurtosis": distribution.kurtosis,
            "n_observations": distribution.n_observations,
        },
        chart_id=chart_id_histogram(distribution.column_name),
        score=abs(skewness),
    )


def _modality_insight(distribution: NumericDistribution) -> Insight | None:
    if distribution.modality != "multimodal":
        return None

    return Insight(
        title=f"{distribution.column_name} suggests multiple subpopulations",
        message=(
            f"The distribution of {distribution.column_name} shows "
            f"{distribution.n_modes} modes in its histogram, which often "
            f"indicates that several distinct groups are mixed in the "
            f"same column. Averaging the whole column may hide these groups."
        ),
        severity="info",
        category="distribution",
        columns=[distribution.column_name],
        evidence={
            "n_modes": distribution.n_modes,
            "n_observations": distribution.n_observations,
        },
        chart_id=chart_id_histogram(distribution.column_name),
        score=float(distribution.n_modes),
    )


def _concentration_insight(distribution: NumericDistribution) -> Insight | None:
    share = distribution.top_5pct_share

    if share is None or share < 40.0:
        return None

    return Insight(
        title=f"Top values dominate the total of {distribution.column_name}",
        message=(
            f"The top 5% of observations in {distribution.column_name} "
            f"concentrate {share:.1f}% of the total. Analyses based on "
            f"the overall mean may be driven by a small number of rows."
        ),
        severity="info",
        category="distribution",
        columns=[distribution.column_name],
        evidence={"top_5pct_share": share},
        chart_id=chart_id_histogram(distribution.column_name),
        score=share,
    )


def generate_distribution_insights(
    distributions: list[NumericDistribution],
) -> list[Insight]:
    insights: list[Insight] = []

    for distribution in distributions:
        for generator in (
            _skewness_insight,
            _modality_insight,
            _concentration_insight,
        ):
            insight = generator(distribution)

            if insight is not None:
                insights.append(insight)

    return insights


# ---------------------------------------------------------------------------
# Patrones (grupos y outliers)
# ---------------------------------------------------------------------------


def _group_difference_insight(
    pattern: GroupDifferencePattern,
) -> Insight | None:
    strength = max(
        abs(pattern.standardized_difference),
        abs(pattern.robust_standardized_difference),
    )

    if strength < 1.0:
        return None

    relative_pct = (
        abs(pattern.difference) / abs(pattern.overall_mean) * 100
        if pattern.overall_mean != 0
        else None
    )

    relative_text = (
        f" ({relative_pct:.0f}% away from the overall mean)"
        if relative_pct is not None
        else ""
    )

    return Insight(
        title=(
            f"Group '{pattern.group_value}' of {pattern.group_column} "
            f"stands out in {pattern.numeric_column}"
        ),
        message=(
            f"In {pattern.group_column}='{pattern.group_value}', the average "
            f"{pattern.numeric_column} is {pattern.group_mean:.2f} vs an "
            f"overall average of {pattern.overall_mean:.2f} "
            f"(difference: {pattern.difference:+.2f}{relative_text}, "
            f"standardized: {pattern.standardized_difference:+.2f}, "
            f"n={pattern.group_size})."
        ),
        severity="info",
        category="pattern",
        columns=[pattern.group_column, pattern.numeric_column],
        evidence={
            "group_mean": pattern.group_mean,
            "overall_mean": pattern.overall_mean,
            "difference": pattern.difference,
            "standardized_difference": pattern.standardized_difference,
            "robust_standardized_difference": pattern.robust_standardized_difference,
            "group_size": pattern.group_size,
        },
        chart_id=chart_id_categorical_numeric(
            pattern.group_column,
            pattern.numeric_column,
        ),
        score=strength,
        caveats=[ASSOCIATION_CAVEAT],
    )


def _numeric_outlier_insight(
    pattern: NumericOutlierPattern,
) -> Insight | None:
    if pattern.outlier_count == 0:
        return None

    percentage = pattern.outlier_count / pattern.n_observations * 100

    most_extreme = max(pattern.outlier_values, key=abs)

    return Insight(
        title=f"{pattern.column} contains {pattern.outlier_count} IQR outliers",
        message=(
            f"{pattern.column} has {pattern.outlier_count} values outside "
            f"the IQR range [{pattern.lower_bound:.2f}, "
            f"{pattern.upper_bound:.2f}] "
            f"({percentage:.1f}% of {pattern.n_observations} observations). "
            f"Most extreme value: {most_extreme:.2f}."
        ),
        severity="info",
        category="pattern",
        columns=[pattern.column],
        evidence={
            "outlier_count": pattern.outlier_count,
            "n_observations": pattern.n_observations,
            "outlier_percentage": percentage,
            "lower_bound": pattern.lower_bound,
            "upper_bound": pattern.upper_bound,
            "most_extreme": most_extreme,
        },
        chart_id=chart_id_histogram(pattern.column),
        score=percentage,
        caveats=[OUTLIER_CAVEAT],
    )


def generate_pattern_insights(
    group_difference_patterns: list[GroupDifferencePattern],
    numeric_outlier_patterns: list[NumericOutlierPattern],
    *,
    max_group_insights: int = 10,
    max_outlier_insights: int = 10,
) -> list[Insight]:
    insights: list[Insight] = []

    group_insights = [
        insight
        for pattern in group_difference_patterns
        if (insight := _group_difference_insight(pattern)) is not None
    ]

    group_insights.sort(key=lambda insight: insight.score, reverse=True)
    insights.extend(group_insights[:max_group_insights])

    outlier_insights = [
        insight
        for pattern in numeric_outlier_patterns
        if (insight := _numeric_outlier_insight(pattern)) is not None
    ]

    outlier_insights.sort(key=lambda insight: insight.score, reverse=True)
    insights.extend(outlier_insights[:max_outlier_insights])

    return insights


# ---------------------------------------------------------------------------
# Estructura del dataset
# ---------------------------------------------------------------------------


def _constant_column_insight(
    column: ColumnProfile,
) -> Insight | None:
    if column.role in {"identifier", "text", "unknown"}:
        return None

    if column.n_unique != 1:
        return None

    sample = column.sample_values[0] if column.sample_values else "?"

    return Insight(
        title=f"{column.name} is constant",
        message=(
            f"{column.name} has a single distinct value ({sample}) and "
            f"cannot differentiate observations. Consider removing it."
        ),
        severity="warning",
        category="structure",
        columns=[column.name],
        evidence={"n_unique": column.n_unique},
        score=50.0,
    )


def _near_constant_column_insight(
    column: ColumnProfile,
    n_rows: int,
) -> Insight | None:
    if column.role in {"identifier", "text", "unknown"}:
        return None

    if n_rows < 100 or column.n_unique > 1:
        return None

    uniqueness_pct = column.pct_unique

    if uniqueness_pct >= 1.0:
        return None

    return Insight(
        title=f"{column.name} is almost constant",
        message=(
            f"{column.name} has only {column.n_unique} distinct value(s) "
            f"across {n_rows} rows ({uniqueness_pct:.2f}% unique). It "
            f"carries very little analytical information."
        ),
        severity="info",
        category="structure",
        columns=[column.name],
        evidence={"n_unique": column.n_unique, "pct_unique": uniqueness_pct},
        score=25.0,
    )


def generate_structure_insights(
    profile: DatasetProfile,
) -> list[Insight]:
    insights: list[Insight] = []

    for column in profile.columns:
        constant_insight = _constant_column_insight(column)

        if constant_insight is not None:
            insights.append(constant_insight)

        near_constant_insight = _near_constant_column_insight(column, profile.n_rows)

        if near_constant_insight is not None:
            insights.append(near_constant_insight)

    identifier_columns = [
        column.name for column in profile.columns if column.role == "identifier"
    ]

    if identifier_columns:
        insights.append(
            Insight(
                title="Identifier columns excluded from analysis",
                message=(
                    f"The following columns were treated as identifiers and "
                    f"excluded from statistical analysis: "
                    f"{', '.join(identifier_columns)}."
                ),
                severity="info",
                category="structure",
                columns=identifier_columns,
                evidence={"n_identifiers": len(identifier_columns)},
                score=5.0,
            )
        )

    return insights


# ---------------------------------------------------------------------------
# Agregaciones
# ---------------------------------------------------------------------------


def _dominant_group_insight(
    aggregation: CategoricalNumericAggregation,
    roles: dict[str, str] | None = None,
) -> Insight | None:
    if not aggregation.groups:
        return None

    total_count = sum(group.count for group in aggregation.groups)

    if total_count == 0:
        return None

    dominant = max(aggregation.groups, key=lambda group: group.count)

    share = dominant.count / total_count * 100

    if share <= 50.0:
        return None

    column_categorical = aggregation.column_categorical
    is_boolean = bool(roles) and roles.get(column_categorical) == "boolean"
    label = humanize_group_value(
        column_categorical, dominant.group_value, roles
    )

    if is_boolean:
        raw = dominant.group_value.strip().lower()

        # La dominancia del estado negativo (0/no) es el caso esperado
        # en casi cualquier dataset: no aporta nada.
        if raw in _BOOLEAN_NEGATIVE_VALUES:
            return None

        # La dominancia del estado positivo sí es noticia (p.ej. '92%
        # de los pedidos están reembolsados'), pero solo si es fuerte.
        if share < BOOLEAN_DOMINANT_ALERT_SHARE:
            return None

        return Insight(
            title=f"'{label}' dominates {column_categorical}",
            message=(
                f"{dominant.count} of {total_count} rows "
                f"({share:.1f}%) are '{label}' in "
                f"{column_categorical}. Most rows fall in the positive "
                f"state of this flag — verify this distribution is "
                f"expected before drawing conclusions from it."
            ),
            severity="warning",
            category="aggregation",
            columns=[column_categorical],
            evidence={
                "dominant_group": dominant.group_value,
                "dominant_share": share,
                "n_observations": total_count,
            },
            score=share,
        )

    return Insight(
        title=(
            f"'{label}' dominates "
            f"{column_categorical}"
        ),
        message=(
            f"Group '{label}' concentrates {share:.1f}% of "
            f"the {total_count} rows in {column_categorical}. "
            f"Comparisons across groups may be dominated by this category."
        ),
        severity="info",
        category="aggregation",
        columns=[column_categorical],
        evidence={
            "dominant_group": dominant.group_value,
            "dominant_share": share,
            "n_observations": total_count,
        },
        score=share,
    )


def generate_aggregation_insights(
    aggregations: list[CategoricalNumericAggregation],
    roles: dict[str, str] | None = None,
) -> list[Insight]:
    insights: list[Insight] = []

    for aggregation in aggregations:
        insight = _dominant_group_insight(aggregation, roles)

        if insight is not None:
            insights.append(insight)

    # Una sola insight por columna categórica (la de mayor concentración).
    best_by_column: dict[str, Insight] = {}

    for insight in insights:
        column = insight.columns[0]

        if column not in best_by_column or insight.score > best_by_column[column].score:
            best_by_column[column] = insight

    return sorted(best_by_column.values(), key=lambda insight: insight.score, reverse=True)


# ---------------------------------------------------------------------------
# Calidad (entry point)
# ---------------------------------------------------------------------------


def generate_quality_insights(
    report: QualityReport,
) -> list[Insight]:
    insights: list[Insight] = []

    for column in report.columns:
        for generator in (
            _missing_values_insight,
            _type_mismatch_insight,
            _invalid_values_insight,
            _extreme_values_insight,
        ):
            insight = generator(column)

            if insight is not None:
                insights.append(insight)

    duplicate_insight = _duplicates_insight(report)

    if duplicate_insight is not None:
        insights.append(duplicate_insight)

    return insights


# ---------------------------------------------------------------------------
# Executive summary
# ---------------------------------------------------------------------------

_SEVERITY_WEIGHT = {"warning": 2.0, "info": 1.0}


def generate_executive_summary(
    insights: list[Insight],
    *,
    top_n: int = EXECUTIVE_SUMMARY_TOP_N,
    max_warnings: int = EXECUTIVE_SUMMARY_MAX_WARNINGS,
    max_per_category: int = EXECUTIVE_PER_CATEGORY_MAX,
) -> list[Insight]:
    """
    Selecciona los hallazgos más relevantes.

    Estrategia: como máximo `max_warnings` alerts de calidad
    (para que no ahoguen los insights analíticos) y el resto se
    completa con selección estratificada por categoría (round-robin
    en orden de score): los scores no son comparables entre familias
    (un group_separation 2.0 compite contra un "ranking score 100"),
    así cada familia aporta sus mejores hallazgos al resumen.
    """
    ranked = sorted(
        insights,
        key=lambda insight: (
            _SEVERITY_WEIGHT.get(insight.severity, 1.0) * 100.0 + insight.score
        ),
        reverse=True,
    )

    warnings = [insight for insight in ranked if insight.severity == "warning"]
    others = [insight for insight in ranked if insight.severity != "warning"]

    selected: list[Insight] = warnings[:max_warnings]

    by_category: dict[str, list[Insight]] = {}

    for insight in others:
        by_category.setdefault(insight.category, []).append(insight)

    quota = max(top_n - len(selected), 0)
    taken_per_category: dict[str, int] = {key: 0 for key in by_category}

    while quota > 0 and any(by_category.values()):
        progressed = False

        for category in sorted(by_category):
            queue = by_category[category]

            if not queue:
                continue

            if taken_per_category[category] >= max_per_category:
                continue

            selected.append(queue.pop(0))
            taken_per_category[category] += 1
            quota -= 1
            progressed = True

            if quota == 0:
                break

        if not progressed:
            break

    # Si el reparto estratificado no llenó los slots (pocos datos o
    # una sola categoría dominante), se completa por score puro: el
    # resumen nunca queda corto por culpa de la cuota.
    if quota > 0:
        leftovers = [
            insight
            for queue in by_category.values()
            for insight in queue
        ]
        leftovers.sort(
            key=lambda insight: (
                _SEVERITY_WEIGHT.get(insight.severity, 1.0) * 100.0
                + insight.score
            ),
            reverse=True,
        )
        selected.extend(leftovers[:quota])

    # Solo si faltan slots se rellenan con warnings adicionales.
    if len(selected) < top_n:
        selected.extend(
            warnings[max_warnings : top_n - len(selected) + max_warnings]
        )

    selected_ids = {id(insight) for insight in selected}

    return sorted(
        (insight for insight in ranked if id(insight) in selected_ids),
        key=lambda insight: (
            _SEVERITY_WEIGHT.get(insight.severity, 1.0) * 100.0 + insight.score
        ),
        reverse=True,
    )
