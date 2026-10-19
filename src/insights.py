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
    COUNT_SKIP_SKEW,
    COUNT_SKIP_TOP_PCT,
    EXECUTIVE_SUMMARY_TOP_N,
    INSIGHT_BUDGET_BY_FAMILY,
    INSIGHT_BUDGET_TOTAL,
    MIN_GROUP_SEPARATION,
    MIN_RELATIVE_DIFF,
    RELATIONSHIP_MAX_MISSING_PCT,
    RELATIONSHIP_MIN_GROUP_N,
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


def _coverage_caveats(coverage_pct: float) -> list[str]:
    """
    Califica una relación según qué fracción del dataset cubre.

    - ≥ el umbral: sin caveat (la relación describe el dataset entero).
    - ≥ la mitad: caveat, pero se emite (missing moderado).
    - < la mitad: no se emite (el "hallazgo" es sobre un subconjunto
      pequeño y su fuerza es engañosa).
    """
    if coverage_pct >= RELATIONSHIP_MAX_MISSING_PCT:
        return []

    if coverage_pct >= RELATIONSHIP_MAX_MISSING_PCT / 2.0:
        return [f"Based on {coverage_pct:.0f}% of rows (missing in the data)."]

    return [_COVERAGE_SKIP]


_COVERAGE_SKIP = "__skip_low_coverage__"


def _practically_meaningful(
    group_means: dict[str, float],
    group_separation: float,
) -> bool:
    """
    Significancia práctica (P3): una diferencia grande en desviaciones
    estándar puede ser artefacto de poca varianza global. Exige además
    que las medias de grupo se dispersen relativamente.

    `pclass across sex` (sep 0.28, ~10% relativo) queda fuera;
    `age across pclass` (sep 0.90, ~42% relativo) pasa.
    """
    if group_separation >= MIN_GROUP_SEPARATION * 2.0:
        return True

    if not group_means:
        return False

    values = list(group_means.values())
    mean_of_means = sum(values) / len(values)

    if mean_of_means == 0:
        return group_separation >= MIN_GROUP_SEPARATION * 2.0

    spread_pct = (max(values) - min(values)) / abs(mean_of_means) * 100.0

    return spread_pct >= MIN_RELATIVE_DIFF * 100.0


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

    # Validación contextual (P3): una diferencia estadísticamente
    # separada pero prácticamente nula no debe sonar sustancial.
    if not _practically_meaningful(
        relationship.group_means,
        relationship.group_separation,
    ):
        return None

    # Un grupo de n<30 no da base para afirmar nada sobre su media.
    # Se evalúa acá y no en el filtro de relaciones: inferencia tiene
    # sus propios umbrales (INFERENCE_MIN_GROUPS/OBSERVATIONS) y su
    # caveat "small sample", y vaciarla acá apagaría hallazgos
    # legítimos en datasets con grupos chicos.
    if not relationship.group_counts:
        return None

    if min(relationship.group_counts.values()) < RELATIONSHIP_MIN_GROUP_N:
        return None

    coverage_caveats = _coverage_caveats(relationship.coverage_pct)

    if _COVERAGE_SKIP in coverage_caveats:
        return None

    top_group = max(relationship.group_means, key=relationship.group_means.get)
    bottom_group = min(relationship.group_means, key=relationship.group_means.get)

    caveats = [ASSOCIATION_CAVEAT, *coverage_caveats]

    score = relationship.group_separation

    if coverage_caveats:
        # La fuerza bruta sobre un subconjunto no es comparable con una
        # relación sobre el dataset entero: se degrada para no encabezar
        # el ranking con un hallazgo parcial.
        score *= 0.5

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
            "coverage_pct": round(relationship.coverage_pct, 2),
        },
        chart_id=chart_id_categorical_numeric(
            relationship.column_categorical,
            relationship.column_numeric,
        ),
        score=score,
        caveats=caveats,
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

    # Una variable de conteo es asimétrica a la derecha casi por
    # definición (suelo en cero, cola larga de valores altos); emitir
    # esa forma como hallazgo es trivial. Solo interesa si la asimetría
    # es tan extrema que ya no es la forma esperada del conteo.
    if distribution.looks_like_count and abs(skewness) < COUNT_SKIP_SKEW:
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

    # En una numérica discreta de pocos valores, cada pico del
    # histograma es un valor posible; "varias subpoblaciones" no
    # aporta nada cuando casi todos los valores son modos.
    if distribution.is_low_cardinality_discrete:
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

    # Un conteo siempre concentra una parte notable de su total en la
    # cola superior (la masa se apila en los valores bajos). Emitir esa
    # concentración como hallazgo es trivial; solo interesa si es tan
    # desproporcionada que el total ya no representa al conjunto.
    if distribution.looks_like_count and share < COUNT_SKIP_TOP_PCT:
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
    coverage_pct: float | None = None,
) -> Insight | None:
    strength = max(
        abs(pattern.standardized_difference),
        abs(pattern.robust_standardized_difference),
    )

    if strength < 1.0:
        return None

    # Validación contextual (P3): una relación sobre un subconjunto
    # pequeño no debe emitirse con la misma fuerza que una sobre el
    # dataset entero. Sin cobertura calculada no se califica (compat).
    if coverage_pct is not None:
        coverage_caveats = _coverage_caveats(coverage_pct)

        if _COVERAGE_SKIP in coverage_caveats:
            return None
    else:
        coverage_caveats = []

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

    score = strength

    if coverage_caveats:
        score *= 0.5

    evidence: dict[str, float | int | str] = {
        "group_mean": pattern.group_mean,
        "overall_mean": pattern.overall_mean,
        "difference": pattern.difference,
        "standardized_difference": pattern.standardized_difference,
        "robust_standardized_difference": pattern.robust_standardized_difference,
        "group_size": pattern.group_size,
    }

    if coverage_pct is not None:
        evidence["coverage_pct"] = round(coverage_pct, 2)

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
        evidence=evidence,
        chart_id=chart_id_categorical_numeric(
            pattern.group_column,
            pattern.numeric_column,
        ),
        score=score,
        caveats=[ASSOCIATION_CAVEAT, *coverage_caveats],
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
    coverage_by_pair: dict[tuple[str, str], float] | None = None,
    max_group_insights: int = 10,
    max_outlier_insights: int = 10,
) -> list[Insight]:
    insights: list[Insight] = []
    coverage_by_pair = coverage_by_pair or {}

    group_insights = [
        insight
        for pattern in group_difference_patterns
        if (
            insight := _group_difference_insight(
                pattern,
                coverage_by_pair.get((pattern.group_column, pattern.numeric_column)),
            )
        )
        is not None
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


def apply_insight_budget(
    by_family: dict[str, list[Insight]],
) -> dict[str, list[Insight]]:
    """
    Presupuesto por familia (P4): recorta cada familia a su cupo,
    conservando los mejores por score dentro de la familia.

    Fase 1: por familia, los `quota[family]` mejores por score. No
    hay dedupe por `chart_id`: varios insights pueden prometer el
    mismo chart (la serie temporal completa, el histograma de una
    columna) y cada uno es un hallazgo distinto — el corte es
    siempre por score, nunca por chart compartido.

    Fase 2: si el total supera `INSIGHT_BUDGET_TOTAL` (guarda
    patológica; con los quotas actuales ningún dataset del corpus
    QA lo alcanza), se recorta por prioridad de familia: contexto
    primero (aggregation, comparative), sustancia después,
    calidad nunca.
    """
    quotas = INSIGHT_BUDGET_BY_FAMILY

    def priority(family: str) -> int:
        return _FAMILY_PRIORITY.get(family, len(_FAMILY_PRIORITY))

    budget_total = INSIGHT_BUDGET_TOTAL
    kept_total = 0
    result: dict[str, list[Insight]] = {}

    for family, insights in by_family.items():
        quota = quotas.get(family, 0)
        ordered = sorted(
            insights,
            key=lambda insight: insight.score,
            reverse=True,
        )
        result[family] = ordered[:quota]
        kept_total += len(result[family])

    # Fase 2: guarda patológica. Recorta empezando por la familia
    # de menor prioridad (contexto primero), y dentro de cada
    # familia por el peor score.
    if kept_total > budget_total:
        for family in sorted(
            (key for key in result if result[key]),
            key=lambda name: (-priority(name), name),
            reverse=True,
        ):
            queue = result[family]
            overflow = kept_total - budget_total

            if overflow <= 0:
                break

            queue.sort(key=lambda insight: insight.score)
            drop = min(overflow, len(queue))
            result[family] = queue[drop:]
            kept_total -= drop

    return {family: insights for family, insights in result.items() if insights}


_FAMILY_PRIORITY = {
    "quality_insights": 0,
    "relationship_insights": 1,
    "inference_insights": 2,
    "distribution_insights": 3,
    "pattern_insights": 4,
    "structure_insights": 5,
    "temporal_insights": 6,
    "ml_insights": 7,
    "comparative_insights": 8,
    "aggregation_insights": 9,
}


def generate_executive_summary(
    insights: list[Insight],
    *,
    n_rows: int | None = None,
    n_columns: int | None = None,
    top_n: int = EXECUTIVE_SUMMARY_TOP_N,
) -> list[Insight]:
    """
    Resumen ejecutivo narrativo (P4): 4-6 frases construidas por
    template, en vez de un vertedero de tarjetas.

    Cada frase referencia (por título corto) la tarjeta fuente; el
    resultado sigue siendo una lista de `Insight` (para no romper
    `_INSIGHT_LIST_FIELDS`) con `category="executive_summary"`.
    Las familias se buscan por la `category` real de los insights
    (`relationship`, `quality`, …), no por el nombre de familia del
    presupuesto (`relationship_insights`, …).
    """
    if not insights:
        return []

    ranked = sorted(
        insights,
        key=lambda insight: (
            _SEVERITY_WEIGHT.get(insight.severity, 1.0) * 100.0 + insight.score
        ),
        reverse=True,
    )

    by_category: dict[str, list[Insight]] = {}
    for insight in ranked:
        by_category.setdefault(insight.category, []).append(insight)

    def best(category: str) -> Insight | None:
        queue = by_category.get(category, [])
        return queue[0] if queue else None

    def phrase(
        title: str,
        message: str,
        source: Insight | None = None,
    ) -> Insight:
        return Insight(
            title=title,
            message=message,
            severity="info",
            category="executive_summary",
            columns=list(source.columns) if source is not None else [],
            evidence={},
        )

    def see(insight: Insight) -> str:
        return f"{insight.message.rstrip('.')} (see: {insight.title[:72]})"

    phrases: list[Insight] = []

    # 1. Dataset + calidad principal.
    shape = (
        f"Dataset with {n_rows} rows × {n_columns} columns"
        if n_rows is not None and n_columns is not None
        else "Dataset"
    )
    quality_warnings = [
        insight
        for insight in by_category.get("quality", [])
        if insight.severity == "warning"
    ]

    if quality_warnings:
        quality_msg = quality_warnings[0].message.rstrip(".")
        dataset_msg = f"{shape}. Main quality signal: {quality_msg}."
    else:
        dataset_msg = f"{shape}. No quality warnings."

    phrases.append(phrase("Executive summary — dataset", dataset_msg))

    # 2-7. Una frase por familia de sustancia, en orden narrativo.
    for category, title in (
        ("relationship", "Executive summary — strongest relationship"),
        ("distribution", "Executive summary — distribution"),
        ("temporal", "Executive summary — temporal"),
        ("ml_recommendations", "Executive summary — ML target"),
        ("comparative", "Executive summary — comparative"),
    ):
        source = best(category)
        if source is not None:
            phrases.append(phrase(title, see(source), source))

    # Patrón o inferencia comparten slot: son dos rutas al mismo
    # hallazgo ("se separan" vs "con qué confianza").
    pattern = best("pattern") or best("inference")
    if pattern is not None:
        phrases.append(
            phrase("Executive summary — pattern", see(pattern), pattern),
        )

    return phrases[:top_n]
