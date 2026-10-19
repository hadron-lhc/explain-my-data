"""
comparative_insights.py

Insights de segundo nivel generados por reglas:

A. Ranking de grupo por total (con contraste volumen/valor).
B. Deconposición de drivers: ¿el total lo mueve el volumen o el valor?
C. Consistencia entre métricas: el mismo grupo lidera dos medidas.
D. Distribuciones categóricas balanceadas (sin valor dominante).
E. Hallazgos nulos: comparaciones binarias casi iguales, correlaciones
   débiles y pares categóricos independientes.

Todo es determinista: plantillas + evidencia numérica del pipeline.
No se usa ningún modelo de lenguaje.
"""

from __future__ import annotations

import re

from src.aggregations import CategoricalNumericAggregation, GroupAggregation
from src.config import (
    BALANCED_MAX_SHARE,
    BALANCED_MAX_VALUES,
    BALANCED_MIN_VALUES,
    BALANCED_SKEW,
    COMPARATIVE_BALANCED_MAX,
    COMPARATIVE_CONSISTENCY_MAX,
    COMPARATIVE_CONSISTENCY_PER_METRIC_MAX,
    COMPARATIVE_DRIVER_MAX,
    COMPARATIVE_DRIVER_PER_METRIC_MAX,
    COMPARATIVE_NULL_MAX,
    COMPARATIVE_RANK_MAX,
    COMPARATIVE_RANK_PER_METRIC_MAX,
    CONSISTENCY_GAP,
    COVERAGE_FULL_PCT,
    DRIVER_SPREAD_RATIO,
    DRIVER_VOLUME_MAX,
    NULL_CATNUM_MAX_SEPARATION,
    NULL_CORR_MAX,
    NULL_MIN_OBSERVATIONS,
    NULL_P_MIN,
    NULL_V_MAX,
    RANK_TOP_GAP,
)
from src.insights import (
    Insight,
    chart_id_categorical_categorical,
    chart_id_categorical_numeric,
    chart_id_scatter,
    humanize_group_value,
)
from src.relationships import (
    CategoricalCategoricalRelationship,
    CategoricalNumericRelationship,
    NumericRelationship,
)
from src.statistics import CategoricalStatistics

_EPS = 1e-9

# Nombres donde sumar la columna no tiene sentido analítico
# (tasas, edades, fechas): se excluyen de los insights basados en totales.
_SUM_NAME_SKIP = re.compile(
    r"(rate|ratio|pct|percent|(^|_)age($|_)|date|(^|_)id($|_))",
    re.IGNORECASE,
)


def _sum_meaningful(column_name: str) -> bool:
    return _SUM_NAME_SKIP.search(column_name) is None


def _totals_phrase(column_name: str) -> str:
    """'total price' / 'total_sales' — evita 'total total_sales'."""
    lowered = column_name.lower()

    if lowered.startswith("total_") or lowered == "total":
        return column_name

    return f"total {column_name}"


# ---------------------------------------------------------------------------
# Formato de números
# ---------------------------------------------------------------------------


def format_number(value: float) -> str:
    """
    Formato compacto legible sin símbolo de moneda.

    >= 1e6 -> "5.02M"; >= 1e4 -> separadores de miles; enteros ->
    separadores de miles; resto -> 2 decimales.
    """
    magnitude = abs(value)

    if magnitude >= 1e6:
        scaled = value / 1e6
        text = f"{scaled:.2f}".rstrip("0").rstrip(".")
        return f"{text}M"

    if magnitude >= 1e4:
        return f"{value:,.0f}"

    if float(value).is_integer():
        return f"{int(value):,}"

    return f"{value:,.2f}"


def _format_decimal(value: float) -> str:
    return f"{value:,.2f}"


# ---------------------------------------------------------------------------
# Helpers sobre agregaciones
# ---------------------------------------------------------------------------


def _group_total(group: GroupAggregation) -> float:
    return group.mean * group.count


def _ranked_totals(
    aggregation: CategoricalNumericAggregation,
) -> list[tuple[GroupAggregation, float]]:
    totals = [
        (group, _group_total(group))
        for group in aggregation.groups
        if group.count > 0
    ]
    totals.sort(key=lambda item: item[1], reverse=True)
    return totals


def _relative_spread(values: list[float], denominator: float) -> float:
    if abs(denominator) < _EPS or not values:
        return 0.0

    return (max(values) - min(values)) / abs(denominator)


def _share_qualifier(
    share: float,
    column_numeric: str,
    column_categorical: str,
    coverage_pct: float,
    *,
    all_non_negative: bool,
) -> str:
    """
    Texto de participación honesto.

    La agregación solo incluye filas donde par (grupo, valor) no es nulo.
    Si eso no cubre casi todo el dataset, el share es de subconjunto y hay
    que decirlo en lugar de afirmar 'of all X'.
    """
    if not all_non_negative:
        return ""

    if coverage_pct >= COVERAGE_FULL_PCT:
        return f" ({share:.1f}% of all {column_numeric})"

    return (
        f" ({share:.1f}% of {column_numeric} where "
        f"{column_categorical} is set; covers {coverage_pct:.0f}% of rows)"
    )


# ---------------------------------------------------------------------------
# A. Ranking de grupo por total
# ---------------------------------------------------------------------------


def _ranking_insight(
    aggregation: CategoricalNumericAggregation,
    roles: dict[str, str] | None = None,
) -> Insight | None:
    if roles and roles.get(aggregation.column_categorical) == "boolean":
        # '0' lidera el total es trivialmente cierto en cualquier flag:
        # no es un hallazgo.
        return None

    if not _sum_meaningful(aggregation.column_numeric):
        return None

    if aggregation.n_observations < 20:
        return None

    totals = _ranked_totals(aggregation)

    if len(totals) < 2:
        return None

    (top, top_total), (second, second_total) = totals[0], totals[1]

    if top_total <= 0 or second_total <= 0:
        return None

    ratio = top_total / second_total

    if ratio < RANK_TOP_GAP:
        return None

    overall_total = sum(total for _, total in totals)

    if overall_total <= 0:
        return None

    share = top_total / overall_total * 100

    all_non_negative = all(total >= 0 for _, total in totals)
    share_text = _share_qualifier(
        share,
        aggregation.column_numeric,
        aggregation.column_categorical,
        aggregation.coverage_pct,
        all_non_negative=all_non_negative,
    )

    # Contraste volumen vs valor: ¿el líder lo es por tamaño o por valor?
    counts = [group.count for group in aggregation.groups]
    count_spread = _relative_spread(
        [float(count) for count in counts],
        float(sum(counts)),
    )
    biggest = max(aggregation.groups, key=lambda group: group.count)

    if count_spread <= DRIVER_VOLUME_MAX and len(totals) > 2:
        contrast = (
            f"despite group sizes being within {count_spread:.0%} of one "
            f"another — the gap comes from the average "
            f"({_format_decimal(top.mean)} vs "
            f"{_format_decimal(second.mean)} for {second.group_value})."
        )
    elif biggest.group_value != top.group_value:
        contrast = (
            f"even though the largest group by row count is "
            f"'{biggest.group_value}' ({biggest.count} rows)."
        )
    else:
        contrast = ""

    message = (
        f"Combined {aggregation.column_numeric} for "
        f"'{top.group_value}' is "
        f"{format_number(top_total)}{share_text}, "
        f"{ratio:.1f}\u00d7 the next group ('{second.group_value}' at "
        f"{format_number(second_total)})"
    )

    if contrast:
        message += f", {contrast}"
    else:
        message += "."

    message += (
        f" This suggests '{top.group_value}' stands out mainly through "
        f"its per-row value rather than a larger number of rows."
        if count_spread <= DRIVER_VOLUME_MAX
        else " This makes it the clear leader by cumulative magnitude."
    )

    score = min(100.0, (ratio - 1.0) * 50.0 + share * 0.2)

    return Insight(
        title=(
            f"{top.group_value} accounts for the highest "
            f"{_totals_phrase(aggregation.column_numeric)}"
        ),
        message=message,
        severity="info",
        category="comparative",
        columns=[aggregation.column_categorical, aggregation.column_numeric],
        evidence={
            "top_total": top_total,
            "second_total": second_total,
            "gap_ratio": ratio,
            "top_share_pct": share,
            "top_group": top.group_value,
            "coverage_pct": aggregation.coverage_pct,
        },
        chart_id=chart_id_categorical_numeric(
            aggregation.column_categorical,
            aggregation.column_numeric,
        ),
        score=score,
    )


def generate_ranking_insights(
    aggregations: list[CategoricalNumericAggregation],
    roles: dict[str, str] | None = None,
) -> list[Insight]:
    insights = [
        insight
        for aggregation in aggregations
        if (insight := _ranking_insight(aggregation, roles)) is not None
    ]

    # Un solo ranking por columna categórica: el de mayor gap.
    # Las demás métricas donde gana el mismo grupo se cubren en la
    # familia de consistencia ("also leads in ...").
    best_by_category: dict[str, Insight] = {}

    for insight in sorted(
        insights,
        key=lambda item: item.evidence.get("gap_ratio", 0.0),
        reverse=True,
    ):
        category_column = insight.columns[0]

        if category_column not in best_by_category:
            best_by_category[category_column] = insight

    # Además, a lo sumo COMPARATIVE_RANK_PER_METRIC_MAX rankings por
    # métrica: una métrica concentrada no debe generar un ranking por
    # cada columna categórica del dataset (P4).
    by_metric: dict[str, list[Insight]] = {}

    for insight in sorted(
        best_by_category.values(),
        key=lambda item: item.score,
        reverse=True,
    ):
        metric_column = insight.columns[1]
        queue = by_metric.setdefault(metric_column, [])

        if len(queue) < COMPARATIVE_RANK_PER_METRIC_MAX:
            queue.append(insight)

    capped = [
        insight
        for queue in by_metric.values()
        for insight in queue
    ]

    ranked = sorted(
        capped,
        key=lambda item: item.score,
        reverse=True,
    )
    return ranked[:COMPARATIVE_RANK_MAX]


# ---------------------------------------------------------------------------
# B. Driver: volumen vs valor
# ---------------------------------------------------------------------------


def _driver_insight(
    aggregation: CategoricalNumericAggregation,
) -> Insight | None:
    if len(aggregation.groups) < 3 or aggregation.n_observations < 30:
        return None

    groups = aggregation.groups
    means = [group.mean for group in groups]
    counts = [float(group.count) for group in groups]

    weighted_mean = (
        sum(group.mean * group.count for group in groups) / sum(counts)
    )

    if abs(weighted_mean) < _EPS:
        return None

    spread_means = _relative_spread(means, weighted_mean)
    spread_counts = _relative_spread(counts, sum(counts))

    numeric_column = aggregation.column_numeric
    categorical_column = aggregation.column_categorical

    if (
        spread_counts <= DRIVER_VOLUME_MAX
        and spread_means
        >= DRIVER_SPREAD_RATIO * max(spread_counts, 0.05)
    ):
        abs_means = [abs(mean) for mean in means]
        ratio = max(abs_means) / max(min(abs_means), _EPS)

        return Insight(
            title=(
                f"Differences in {numeric_column} across "
                f"{categorical_column} are driven by average values "
                f"rather than group sizes"
            ),
            message=(
                f"Row counts are relatively similar across "
                f"{categorical_column} groups (spread: "
                f"{spread_counts:.0%}), while average "
                f"{numeric_column} varies substantially (from "
                f"{_format_decimal(min(means))} to "
                f"{_format_decimal(max(means))}, "
                f"{ratio:.1f}\u00d7). This suggests the variation in "
                f"{numeric_column} comes mainly from per-row value "
                f"rather than volume."
            ),
            severity="info",
            category="comparative",
            columns=[categorical_column, numeric_column],
            evidence={
                "spread_means": spread_means,
                "spread_counts": spread_counts,
                "min_mean": min(means),
                "max_mean": max(means),
                "n_groups": len(groups),
            },
            chart_id=chart_id_categorical_numeric(
                categorical_column,
                numeric_column,
            ),
            score=min(100.0, (spread_means / max(spread_counts, 0.05)) * 20.0),
        )

    if (
        spread_means <= DRIVER_VOLUME_MAX
        and spread_counts
        >= DRIVER_SPREAD_RATIO * max(spread_means, 0.05)
    ):
        ratio = max(counts) / max(min(counts), _EPS)

        return Insight(
            title=(
                f"Differences in {numeric_column} across "
                f"{categorical_column} are driven by group sizes "
                f"rather than averages"
            ),
            message=(
                f"Average {numeric_column} is nearly flat across "
                f"{categorical_column} groups (from "
                f"{_format_decimal(min(means))} to "
                f"{_format_decimal(max(means))}), while row counts "
                f"vary substantially (spread: {spread_counts:.0%}, "
                f"{ratio:.1f}\u00d7 difference in group size). This "
                f"suggests totals differ mainly because some groups "
                f"contain many more rows, not because rows differ."
            ),
            severity="info",
            category="comparative",
            columns=[categorical_column, numeric_column],
            evidence={
                "spread_means": spread_means,
                "spread_counts": spread_counts,
                "min_count": int(min(counts)),
                "max_count": int(max(counts)),
                "n_groups": len(groups),
            },
            chart_id=chart_id_categorical_numeric(
                categorical_column,
                numeric_column,
            ),
            score=min(
                100.0, (spread_counts / max(spread_means, 0.05)) * 20.0
            ),
        )

    return None


def generate_driver_insights(
    aggregations: list[CategoricalNumericAggregation],
) -> list[Insight]:
    insights = [
        insight
        for aggregation in aggregations
        if (insight := _driver_insight(aggregation)) is not None
    ]

    # Un driver por columna categórica (el de mayor score).
    best_by_category: dict[str, Insight] = {}

    for insight in sorted(insights, key=lambda item: item.score, reverse=True):
        category_column = insight.columns[0]

        if category_column not in best_by_category:
            best_by_category[category_column] = insight

    # A lo sumo COMPARATIVE_DRIVER_PER_METRIC_MAX drivers por métrica
    # (P4): "capital_gain driven by occupation" es el hallazgo; el
    # mismo hallazgo repetido por workclass/relationship/race no añade
    # nada.
    by_metric: dict[str, list[Insight]] = {}

    for insight in sorted(
        best_by_category.values(),
        key=lambda item: item.score,
        reverse=True,
    ):
        metric_column = insight.columns[1]
        queue = by_metric.setdefault(metric_column, [])

        if len(queue) < COMPARATIVE_DRIVER_PER_METRIC_MAX:
            queue.append(insight)

    ranked = sorted(
        (
            insight
            for queue in by_metric.values()
            for insight in queue
        ),
        key=lambda item: item.score,
        reverse=True,
    )
    return ranked[:COMPARATIVE_DRIVER_MAX]


# ---------------------------------------------------------------------------
# C. Consistencia entre métricas
# ---------------------------------------------------------------------------


def _consistency_insights(
    aggregations: list[CategoricalNumericAggregation],
    roles: dict[str, str] | None = None,
) -> list[Insight]:
    by_category: dict[str, list[CategoricalNumericAggregation]] = {}

    for aggregation in aggregations:
        if roles and roles.get(aggregation.column_categorical) == "boolean":
            # Mismo criterio que el ranking: 'el flag lidera X' es
            # trivial, no un hallazgo de consistencia.
            continue

        by_category.setdefault(aggregation.column_categorical, []).append(
            aggregation
        )

    candidates: list[tuple[float, Insight]] = []

    for categorical_column, group in by_category.items():
        usable: list[
            tuple[CategoricalNumericAggregation, list[tuple[GroupAggregation, float]]]
        ] = []

        for aggregation in group:
            if not _sum_meaningful(aggregation.column_numeric):
                continue

            totals = _ranked_totals(aggregation)

            if len(totals) < 2:
                continue

            if any(total <= 0 for _, total in totals):
                continue

            usable.append((aggregation, totals))

        if len(usable) < 2:
            continue

        def _gap(
            item: tuple[CategoricalNumericAggregation, list[tuple[GroupAggregation, float]]],
        ) -> float:
            totals = item[1]
            return totals[0][1] / max(totals[1][1], _EPS)

        # La métrica primaria es la de mayor gap; las demás se
        # reportan como "también lidera ..." solo si el mismo grupo
        # gana también ahí.
        primary, primary_totals = max(usable, key=_gap)
        leader = primary_totals[0][0].group_value

        category_candidates: list[tuple[float, Insight]] = []

        for aggregation, totals in usable:
            if aggregation is primary:
                continue

            if totals[0][0].group_value != leader:
                continue

            gap = totals[0][1] / max(totals[1][1], _EPS)

            if gap < CONSISTENCY_GAP:
                continue

            runner_up = totals[1][0].group_value

            insight = Insight(
                title=f"{leader} also leads in {aggregation.column_numeric}",
                message=(
                    f"Beyond {primary.column_numeric}, '{leader}' also has "
                    f"the highest {_totals_phrase(aggregation.column_numeric)} "
                    f"({format_number(totals[0][1])}), "
                    f"{gap:.1f}\u00d7 the next group ('{runner_up}' at "
                    f"{format_number(totals[1][1])}). This suggests the "
                    f"same groups stand out across different measures of "
                    f"{categorical_column}."
                ),
                severity="info",
                category="comparative",
                columns=[categorical_column, aggregation.column_numeric],
                evidence={
                    "leader": leader,
                    "second_total": totals[0][1],
                    "gap_ratio": gap,
                    "primary_numeric": primary.column_numeric,
                },
                chart_id=chart_id_categorical_numeric(
                    categorical_column,
                    aggregation.column_numeric,
                ),
                score=min(100.0, gap * 40.0),
            )

            category_candidates.append((gap, insight))

        # Un solo "also leads" por columna categórica: el de mayor gap.
        if category_candidates:
            candidates.append(max(category_candidates, key=lambda item: item[0]))

    candidates.sort(key=lambda item: item[0], reverse=True)

    # A lo sumo COMPARATIVE_CONSISTENCY_PER_METRIC_MAX por métrica
    # (P4): "White también lidera capital_loss" y "Male también
    # lidera capital_loss" son el mismo hallazgo.
    by_metric: dict[str, list[Insight]] = {}

    for _, insight in candidates:
        metric_column = insight.columns[1]
        queue = by_metric.setdefault(metric_column, [])

        if len(queue) < COMPARATIVE_CONSISTENCY_PER_METRIC_MAX:
            queue.append(insight)

    capped = sorted(
        (
            insight
            for queue in by_metric.values()
            for insight in queue
        ),
        key=lambda item: item.score,
        reverse=True,
    )
    return capped[:COMPARATIVE_CONSISTENCY_MAX]


def generate_consistency_insights(
    aggregations: list[CategoricalNumericAggregation],
    roles: dict[str, str] | None = None,
) -> list[Insight]:
    return _consistency_insights(aggregations, roles)


# ---------------------------------------------------------------------------
# D. Distribuciones categóricas balanceadas
# ---------------------------------------------------------------------------


def _balanced_insight(
    statistics: CategoricalStatistics,
) -> Insight | None:
    if not (
        BALANCED_MIN_VALUES
        <= statistics.n_unique
        <= BALANCED_MAX_VALUES
    ):
        return None

    if len(statistics.value_counts) != statistics.n_unique:
        return None

    if statistics.count <= 0:
        return None

    counts = [count for _, count in statistics.value_counts]

    if min(counts) <= 0:
        return None

    shares = [
        (value, count / statistics.count * 100)
        for value, count in statistics.value_counts
    ]

    top_share = shares[0][1]
    min_share = min(share for _, share in shares)

    if top_share > BALANCED_MAX_SHARE:
        return None

    if top_share / max(min_share, _EPS) > BALANCED_SKEW:
        return None

    even_share = 100.0 / statistics.n_unique
    smallest_value = min(shares, key=lambda item: item[1])[0]

    return Insight(
        title=f"{statistics.column_name} shows no dominant value",
        message=(
            f"The {statistics.n_unique} values of "
            f"{statistics.column_name} are spread almost evenly: "
            f"'{shares[0][0]}' leads with only {top_share:.1f}% and "
            f"'{smallest_value}' has {min_share:.1f}%. Each value "
            f"accounts for roughly {even_share:.0f}% of the "
            f"{statistics.count} rows, suggesting there is no single "
            f"dominant segment in {statistics.column_name}."
        ),
        severity="info",
        category="comparative",
        columns=[statistics.column_name],
        evidence={
            "n_values": statistics.n_unique,
            "top_share_pct": top_share,
            "min_share_pct": min_share,
            "even_share_pct": even_share,
            "n_observations": statistics.count,
        },
        score=100.0 - top_share,
    )


def generate_balanced_insights(
    categorical_statistics: list[CategoricalStatistics],
) -> list[Insight]:
    insights = [
        insight
        for statistics in categorical_statistics
        if (insight := _balanced_insight(statistics)) is not None
    ]

    insights.sort(key=lambda insight: insight.score, reverse=True)
    return insights[:COMPARATIVE_BALANCED_MAX]


# ---------------------------------------------------------------------------
# E. Hallazgos nulos
# ---------------------------------------------------------------------------


def _binary_null_insight(
    relationship: CategoricalNumericRelationship,
    roles: dict[str, str] | None = None,
) -> Insight | None:
    if relationship.n_groups != 2:
        return None

    if relationship.n_observations < NULL_MIN_OBSERVATIONS:
        return None

    # El disparador es la separación en unidades de desviación:
    # una diferencia relativa chica puede seguir siendo informativa
    # si la varianza intra-grupo es mínima, y viceversa.
    if relationship.group_separation > NULL_CATNUM_MAX_SEPARATION:
        return None

    (group_a, mean_a), (group_b, mean_b) = sorted(
        relationship.group_means.items()
    )

    weighted_mean = (
        sum(
            mean * relationship.group_counts.get(group, 0)
            for group, mean in relationship.group_means.items()
        )
        / max(relationship.n_observations, 1)
    )

    relative_diff = (
        abs(mean_a - mean_b) / abs(weighted_mean)
        if abs(weighted_mean) > _EPS
        else 0.0
    )

    count_a = relationship.group_counts.get(group_a, 0)
    count_b = relationship.group_counts.get(group_b, 0)

    closeness = (
        1.0 - relationship.group_separation / NULL_CATNUM_MAX_SEPARATION
    )

    label_a = humanize_group_value(
        relationship.column_categorical, group_a, roles
    )
    label_b = humanize_group_value(
        relationship.column_categorical, group_b, roles
    )

    return Insight(
        title=(
            f"{relationship.column_numeric} is nearly identical "
            f"across {relationship.column_categorical} groups"
        ),
        message=(
            f"Average {relationship.column_numeric} is "
            f"{_format_decimal(mean_a)} for '{label_a}' "
            f"({count_a} rows) and {_format_decimal(mean_b)} for "
            f"'{label_b}' ({count_b} rows) \u2014 a difference of "
            f"only {relative_diff:.1%} of the overall average. This "
            f"suggests {relationship.column_categorical} is not "
            f"strongly associated with "
            f"{relationship.column_numeric} in this dataset."
        ),
        severity="info",
        category="comparative",
        columns=[
            relationship.column_categorical,
            relationship.column_numeric,
        ],
        evidence={
            "mean_group_a": mean_a,
            "mean_group_b": mean_b,
            "relative_difference": relative_diff,
            "group_separation": relationship.group_separation,
            "n_observations": relationship.n_observations,
        },
        chart_id=chart_id_categorical_numeric(
            relationship.column_categorical,
            relationship.column_numeric,
        ),
        # Base más alta que los otros nulos: es el más concreto
        # (compara dos medias observadas directamente).
        score=50.0 + 40.0 * max(closeness, 0.0),
    )


def _weak_correlation_insight(
    relationship: NumericRelationship,
) -> Insight | None:
    if relationship.n_observations < NULL_MIN_OBSERVATIONS:
        return None

    strength = max(abs(relationship.pearson), abs(relationship.spearman))

    if strength > NULL_CORR_MAX:
        return None

    closeness = 1.0 - strength / NULL_CORR_MAX

    return Insight(
        title=(
            f"No meaningful association between "
            f"{relationship.column_x} and {relationship.column_y}"
        ),
        message=(
            f"Correlation between {relationship.column_x} and "
            f"{relationship.column_y} is weak "
            f"(Pearson: {relationship.pearson:.2f}, "
            f"Spearman: {relationship.spearman:.2f}, "
            f"n: {relationship.n_observations}), suggesting these "
            f"columns vary largely independently in this dataset."
        ),
        severity="info",
        category="comparative",
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
        score=40.0 + 40.0 * max(closeness, 0.0),
    )


def _independent_pair_insight(
    relationship: CategoricalCategoricalRelationship,
) -> Insight | None:
    if relationship.n_observations < NULL_MIN_OBSERVATIONS:
        return None

    if relationship.cramers_v > NULL_V_MAX:
        return None

    if relationship.p_value < NULL_P_MIN:
        return None

    closeness = 1.0 - relationship.cramers_v / NULL_V_MAX

    return Insight(
        title=(
            f"{relationship.column_x} and {relationship.column_y} "
            f"appear independent"
        ),
        message=(
            f"Cram\u00e9r's V is {relationship.cramers_v:.2f} with a "
            f"chi2 p-value of {relationship.p_value:.2f} "
            f"(n: {relationship.n_observations}), suggesting no "
            f"detectable association between "
            f"{relationship.column_x} and {relationship.column_y} "
            f"in this dataset."
        ),
        severity="info",
        category="comparative",
        columns=[relationship.column_x, relationship.column_y],
        evidence={
            "cramers_v": relationship.cramers_v,
            "p_value": relationship.p_value,
            "n_observations": relationship.n_observations,
        },
        chart_id=chart_id_categorical_categorical(
            relationship.column_x,
            relationship.column_y,
        ),
        score=40.0 + 40.0 * max(closeness, 0.0),
    )


def generate_null_insights(
    numeric_relationships: list[NumericRelationship],
    categorical_numeric_relationships: list[CategoricalNumericRelationship],
    categorical_categorical_relationships: list[
        CategoricalCategoricalRelationship
    ],
    roles: dict[str, str] | None = None,
) -> list[Insight]:
    insights: list[Insight] = [
        insight
        for relationship in categorical_numeric_relationships
        if (insight := _binary_null_insight(relationship, roles)) is not None
    ]

    insights.extend(
        insight
        for relationship in numeric_relationships
        if (insight := _weak_correlation_insight(relationship)) is not None
    )

    insights.extend(
        insight
        for relationship in categorical_categorical_relationships
        if (insight := _independent_pair_insight(relationship)) is not None
    )

    insights.sort(key=lambda insight: insight.score, reverse=True)
    return insights[:COMPARATIVE_NULL_MAX]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def generate_comparative_insights(
    aggregations: list[CategoricalNumericAggregation],
    categorical_statistics: list[CategoricalStatistics],
    numeric_relationships: list[NumericRelationship],
    categorical_numeric_relationships: list[CategoricalNumericRelationship],
    categorical_categorical_relationships: list[
        CategoricalCategoricalRelationship
    ],
    roles: dict[str, str] | None = None,
) -> list[Insight]:
    """
    Genera los insights comparativos del reporte: rankings,
    drivers, consistencia, balance y hallazgos nulos.

    `roles` mapea columna -> role inferido por el perfilado; se usa
    para descartar hallazgos triviales en booleanos y para
    humanizar sus etiquetas ('0' -> 'not refunded').
    """
    return (
        generate_ranking_insights(aggregations, roles)
        + generate_driver_insights(aggregations)
        + generate_consistency_insights(aggregations, roles)
        + generate_balanced_insights(categorical_statistics)
        + generate_null_insights(
            numeric_relationships,
            categorical_numeric_relationships,
            categorical_categorical_relationships,
            roles,
        )
    )
