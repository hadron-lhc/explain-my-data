"""
Curación de insights: un hallazgo = una tarjeta.

Reglas deterministas y solo-aditivas en información:

1. Cap de patrones por par (PATTERN_PER_PAIR_CAP): un par (grupo,
   numérico) con varios grupos ≥1σ genera un insight por grupo; el
   usuario recibe los `cap` más estandarizados (los extremos) y el
   resto se sintetiza en la tarjeta de relationship si existe.

2. Dedupe relationship↔pattern: "X differs across Y groups" y
   "Group 'G' of Y stands out in X" cuentan el mismo hallazgo por dos
   rutas. Cuando el par ya tiene un patrón sobreviviente, la tarjeta
   relationship se pliega: su group_separation pasa a la evidencia y
   a un caveat del patrón primario (el de mayor score).

3. Fold relationship/pattern↔inference (F1): cuando el par tiene
   tarjeta de inferencia (p, η², grupo tope), las tarjetas
   "differs across" y "stands out" del mismo par se pliegan en ella.

4. Fold de cola (F1): "Potentially extreme values in X" (3×IQR),
   "X contains N IQR outliers" (1.5×IQR) y "X is right/left-skewed"
   describen la misma cola; una sola tarjeta primaria por columna.
"""

import re

from src.config import PATTERN_PER_PAIR_CAP
from src.insights import Insight


def _pair_key(insight: Insight) -> tuple[str, str] | None:
    if len(insight.columns) == 2:
        return tuple(sorted(column.lower() for column in insight.columns))
    return None


def _is_standout(insight: Insight) -> bool:
    return insight.category == "pattern" and "stands out" in insight.title


def _is_group_differs(insight: Insight) -> bool:
    return (
        insight.category == "relationship"
        and "differs across" in insight.title
    )


def _standout_signed(insight: Insight) -> float:
    value = insight.evidence.get("standardized_difference")

    if isinstance(value, (int, float)):
        return float(value)

    return insight.score


def _standout_strength(insight: Insight) -> float:
    return abs(_standout_signed(insight))


def _select_pair_representatives(
    group: list[Insight],
    cap: int,
) -> list[Insight]:
    """
    Para un par (grupo, numérico), conserva los extremos firmados
    (el grupo más alto y el más bajo, que son la historia real:
    "Oceania" y "Africa" en lifeExp) y rellena el resto de slots
    por fuerza absoluta.
    """
    if len(group) <= cap:
        return list(group)

    ordered = sorted(group, key=_standout_signed, reverse=True)
    extremes = [ordered[0], ordered[-1]]

    if extremes[0] is extremes[-1]:
        extremes = [ordered[0]]

    extremes = sorted(extremes, key=_standout_strength, reverse=True)[:cap]
    selected_ids = {id(insight) for insight in extremes}

    fillers = sorted(
        (insight for insight in group if id(insight) not in selected_ids),
        key=_standout_strength,
        reverse=True,
    )

    for insight in fillers:
        if len(selected_ids) >= cap:
            break

        selected_ids.add(id(insight))

    return [insight for insight in group if id(insight) in selected_ids]


def cap_pattern_insights(
    pattern_insights: list[Insight],
    cap: int = PATTERN_PER_PAIR_CAP,
) -> list[Insight]:
    """Conserva `cap` patrones por par (grupo, num): extremos + fillers."""
    standouts_by_pair: dict[tuple[str, str], list[Insight]] = {}

    for insight in pattern_insights:
        if not _is_standout(insight):
            continue
        key = _pair_key(insight)

        if key is not None:
            standouts_by_pair.setdefault(key, []).append(insight)

    kept_ids = {
        id(insight)
        for group in standouts_by_pair.values()
        for insight in _select_pair_representatives(group, cap)
    }

    return [
        insight
        for insight in pattern_insights
        if not _is_standout(insight) or id(insight) in kept_ids
    ]


def fold_group_differs_insights(
    relationship_insights: list[Insight],
    pattern_insights: list[Insight],
) -> list[Insight]:
    """
    Pliega las tarjetas "differs across" cuyo par ya tiene un patrón.

    El group_separation del relationship pliegado se agrega a la
    evidencia y a un caveat del patrón primario (mayor score), así la
    información no se pierde: solo cambia de tarjeta.
    """
    pairs_with_pattern = {
        key
        for insight in pattern_insights
        if _is_standout(insight)
        for key in [_pair_key(insight)]
        if key is not None
    }

    folded_by_pair: dict[tuple[str, str], Insight] = {}
    kept: list[Insight] = []

    for insight in relationship_insights:
        key = _pair_key(insight)

        if _is_group_differs(insight) and key in pairs_with_pattern:
            folded_by_pair[key] = insight
            continue

        kept.append(insight)

    if not folded_by_pair:
        return kept

    primary_by_pair: dict[tuple[str, str], Insight] = {}

    for insight in pattern_insights:
        if not _is_standout(insight):
            continue

        key = _pair_key(insight)

        if key is None:
            continue

        current = primary_by_pair.get(key)

        if current is None or insight.score > current.score:
            primary_by_pair[key] = insight

    for key, folded in folded_by_pair.items():
        primary = primary_by_pair.get(key)

        if primary is None:
            continue

        separation = folded.evidence.get("group_separation")

        if isinstance(separation, (int, float)):
            primary.evidence["group_separation"] = float(separation)
            primary.caveats.append(
                f"group separation across all groups: {float(separation):.2f}"
            )

        # La primaria representa el hallazgo del par completo: hereda
        # la severidad del relationship plegado (una separación
        # significativa pesa más que un standout aislado).
        if folded.severity == "warning":
            primary.severity = "warning"

        n_groups = folded.evidence.get("n_groups")

        if isinstance(n_groups, (int, float)):
            primary.evidence["n_groups"] = int(n_groups)

    return kept


def fold_inference_redundancy(
    relationship_insights: list[Insight],
    pattern_insights: list[Insight],
    inference_insights: list[Insight],
) -> tuple[list[Insight], list[Insight], list[Insight]]:
    """
    F1: una relación cat-num = una tarjeta. La tarjeta de inferencia
    del par (p, η², grupo tope) es la primaria; las tarjetas
    "differs across" (relationship) y "stands out" (pattern) del
    mismo par se pliegan en ella: su evidencia y un resumen compacto
    pasan al mensaje, así el hallazgo no se pierde, solo cambia de
    tarjeta.
    """
    primary_by_pair: dict[tuple[str, str], Insight] = {}

    for insight in inference_insights:
        key = _pair_key(insight)

        if key is None:
            continue

        current = primary_by_pair.get(key)

        if current is None or insight.score > current.score:
            primary_by_pair[key] = insight

    if not primary_by_pair:
        return relationship_insights, pattern_insights, inference_insights

    kept_relationships: list[Insight] = []
    kept_patterns: list[Insight] = []

    for insight in relationship_insights:
        key = _pair_key(insight)

        if _is_group_differs(insight) and key in primary_by_pair:
            _merge_relationship_into_inference(primary_by_pair[key], insight)
        else:
            kept_relationships.append(insight)

    for insight in pattern_insights:
        key = _pair_key(insight)

        if _is_standout(insight) and key in primary_by_pair:
            _merge_pattern_into_inference(primary_by_pair[key], insight)
        else:
            kept_patterns.append(insight)

    return kept_relationships, kept_patterns, inference_insights


def _merge_relationship_into_inference(
    primary: Insight,
    folded: Insight,
) -> None:
    separation = folded.evidence.get("group_separation")

    if isinstance(separation, (int, float)):
        primary.evidence["group_separation"] = float(separation)
        primary.message += (
            f" Group separation across all groups: {float(separation):.2f}."
        )

    n_groups = folded.evidence.get("n_groups")

    if isinstance(n_groups, (int, float)):
        primary.evidence["n_groups"] = int(n_groups)

    # "Highest average: A (a); lowest: B (b)" (+ ratio y n): la
    # inferencia solo nombra al grupo tope, y el grupo mínimo con su
    # media es parte del hallazgo (p.ej. "lowest: USA (75.80)").
    marker = "Highest average: "
    start = folded.message.find(marker)

    if start >= 0:
        primary.message += f" {folded.message[start:]}"

    for key in ("top_group_mean", "bottom_group_mean", "coverage_pct"):
        if key in folded.evidence and key not in primary.evidence:
            primary.evidence[key] = folded.evidence[key]


def _merge_pattern_into_inference(
    primary: Insight,
    folded: Insight,
) -> None:
    group_value = folded.evidence.get("group_value")

    if group_value is None and "Group '" in folded.title:
        group_value = folded.title.split("'", 2)[1]

    group_mean = folded.evidence.get("group_mean")

    if group_value is not None and isinstance(group_mean, (int, float)):
        detail = f"{group_mean:.2f}"
        overall_mean = folded.evidence.get("overall_mean")

        if isinstance(overall_mean, (int, float)):
            detail += f" vs overall {overall_mean:.2f}"

        group_column = folded.columns[0] if folded.columns else ""
        primary.message += (
            f" Group '{group_value}' of {group_column} stands out ({detail})."
        )

    # La tarjeta relationship pudo haberse plegado antes en este
    # patrón: hereda su separación para no perderla en el traslado.
    for key in ("group_separation", "n_groups"):
        if key in folded.evidence and key not in primary.evidence:
            primary.evidence[key] = folded.evidence[key]


def _is_tail_member(insight: Insight, kind: str) -> bool:
    if kind == "extreme":
        return (
            insight.category == "quality"
            and insight.title.startswith("Potentially extreme values in")
        )

    if kind == "outliers":
        return insight.category == "pattern" and "IQR outliers" in insight.title

    return insight.category == "distribution" and "skewed" in insight.title


def _merge_tail_into_primary(
    primary: Insight,
    kind: str,
    folded: Insight,
) -> None:
    column = primary.columns[0] if primary.columns else folded.title

    if kind == "outliers":
        n = folded.evidence.get("outlier_count")
        lower = folded.evidence.get("lower_bound")
        upper = folded.evidence.get("upper_bound")
        most_extreme = folded.evidence.get("most_extreme")

        if (
            isinstance(n, (int, float))
            and isinstance(lower, (int, float))
            and isinstance(upper, (int, float))
            and isinstance(most_extreme, (int, float))
        ):
            primary.message += (
                f" {column} also has {int(n)} IQR outliers outside "
                f"[{lower:.2f}, {upper:.2f}]; most extreme value: "
                f"{most_extreme:.2f}."
            )
    else:  # extreme
        n = folded.evidence.get("n_extreme_values")
        pct = folded.evidence.get("extreme_percentage")

        if isinstance(n, (int, float)) and isinstance(pct, (int, float)):
            primary.message += (
                f" {column} also flags {int(n)} potentially extreme "
                f"values ({pct:.1f}% of the column)."
            )

    for key, value in folded.evidence.items():
        primary.evidence.setdefault(f"folded_{kind}_{key}", value)


def fold_tail_insights(
    quality_insights: list[Insight],
    pattern_insights: list[Insight],
    distribution_insights: list[Insight],
) -> tuple[list[Insight], list[Insight], list[Insight]]:
    """
    F1: una cola = una tarjeta por columna. "Potentially extreme
    values in X" (3×IQR), "X contains N IQR outliers" (1.5×IQR) y
    "X is right/left-skewed" describen la misma cola larga; se
    conserva la primaria (skew > outliers > extreme) y la evidencia
    de las otras se fusiona en su mensaje.
    """
    buckets: dict[str, dict[str, Insight]] = {}

    for kind, insights in (
        ("skew", distribution_insights),
        ("outliers", pattern_insights),
        ("extreme", quality_insights),
    ):
        for insight in insights:
            if len(insight.columns) != 1 or not _is_tail_member(insight, kind):
                continue

            buckets.setdefault(insight.columns[0].lower(), {})[kind] = insight

    folded_ids: set[int] = set()

    for members in buckets.values():
        if len(members) < 2:
            continue

        primary = members.get("skew") or members.get("outliers") or members["extreme"]

        for kind, insight in members.items():
            if insight is primary:
                continue

            _merge_tail_into_primary(primary, kind, insight)
            folded_ids.add(id(insight))

    return (
        [insight for insight in quality_insights if id(insight) not in folded_ids],
        [insight for insight in pattern_insights if id(insight) not in folded_ids],
        [
            insight
            for insight in distribution_insights
            if id(insight) not in folded_ids
        ],
    )


def curate_insights(
    relationship_insights: list[Insight],
    pattern_insights: list[Insight],
) -> tuple[list[Insight], list[Insight]]:
    """Aplica el cap de patrones y el dedupe relationship↔pattern."""
    patterns = cap_pattern_insights(pattern_insights)
    relationships = fold_group_differs_insights(relationship_insights, patterns)

    return relationships, patterns


# --- F4: folds de ruido (outliers, consistencia, derivadas) ----------------

_IQR_TITLE_RE = re.compile(r"contains \d+ IQR outliers", re.IGNORECASE)


def _unique_caveats(insights: list[Insight]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []

    for insight in insights:
        for caveat in insight.caveats:
            if caveat not in seen:
                seen.add(caveat)
                result.append(caveat)

    return result


def fold_outlier_cards(pattern_insights: list[Insight]) -> list[Insight]:
    """
    F4: N tarjetas "X contains N IQR outliers" → una sola tarjeta
    por dataset. Cada columna conserva su conteo y su valor más
    extremo en el mensaje, así los hechos verificados del corpus
    (p. ej. 120.00 en cars) siguen presentes; solo cambia el
    empaquetado.
    """
    members = [
        insight
        for insight in pattern_insights
        if _IQR_TITLE_RE.search(insight.title)
    ]

    if len(members) < 2:
        return pattern_insights

    parts: list[str] = []
    evidence: dict[str, float | int | str] = {
        "folded_outlier_columns": len(members),
    }
    total_outliers = 0.0

    for insight in members:
        column = insight.columns[0] if insight.columns else insight.title
        count = insight.evidence.get("outlier_count")
        pct = insight.evidence.get("outlier_percentage")
        most_extreme = insight.evidence.get("most_extreme")

        if (
            isinstance(count, (int, float))
            and isinstance(pct, (int, float))
            and isinstance(most_extreme, (int, float))
        ):
            parts.append(
                f"{column} contains {int(count)} IQR outliers "
                f"({pct:.1f}% of observations); most extreme value: "
                f"{float(most_extreme):.2f}."
            )
            total_outliers += float(count)
            evidence[f"outliers_{column}"] = int(count)
            evidence[f"most_extreme_{column}"] = float(most_extreme)
        else:
            # Fallback: conservar el texto original si falta evidencia.
            parts.append(insight.message)

    primary = max(members, key=lambda insight: (insight.score, insight.title))
    folded = Insight(
        title=f"IQR outliers across {len(members)} numeric columns",
        message=" ".join(parts),
        severity=(
            "warning"
            if any(insight.severity == "warning" for insight in members)
            else "info"
        ),
        category="pattern",
        columns=[
            column
            for insight in members
            for column in insight.columns
        ],
        evidence={
            **evidence,
            "total_outliers": int(total_outliers),
            "max_outlier_percentage": max(
                insight.score for insight in members
            ),
        },
        chart_id=primary.chart_id,
        score=max(insight.score for insight in members),
        caveats=_unique_caveats(members),
    )

    return [
        insight
        for insight in pattern_insights
        if id(insight) not in {id(member) for member in members}
    ] + [folded]


def _leader_key(insight: Insight, marker: str) -> str | None:
    lowered = insight.title.lower()

    if marker not in lowered:
        return None

    return lowered.split(marker, 1)[0].strip()


def fold_leader_cards(comparative_insights: list[Insight]) -> list[Insight]:
    """
    F4: "X also leads in Y" se pliega en la tarjeta primaria del
    mismo grupo ("X accounts for the highest total Z"). El mensaje
    conserva el título de la secundaria ("Dinner also leads in tip")
    y su estadística, así el hallazgo de consistencia no se pierde.
    """
    primaries: dict[str, Insight] = {}

    for insight in comparative_insights:
        key = _leader_key(insight, " accounts for the highest total")

        if key is None:
            continue

        current = primaries.get(key)

        if current is None or insight.score > current.score:
            primaries[key] = insight

    if not primaries:
        return comparative_insights

    folded_ids: set[int] = set()

    for insight in comparative_insights:
        key = _leader_key(insight, " also leads in")

        if key is None:
            continue

        primary = primaries.get(key)

        if primary is None or primary is insight:
            continue

        primary.columns = list(
            dict.fromkeys(primary.columns + insight.columns)
        )
        primary.score = max(primary.score, insight.score)
        primary.caveats = _unique_caveats([primary, insight])
        primary.message += f" {insight.title}. {insight.message}"

        for evidence_key, value in insight.evidence.items():
            primary.evidence.setdefault(f"also_leads_{evidence_key}", value)

        folded_ids.add(id(insight))

    return [
        insight
        for insight in comparative_insights
        if id(insight) not in folded_ids
    ]


def fold_derived_cards(structure_insights: list[Insight]) -> list[Insight]:
    """
    F4: varias "X looks derived from other columns" → una tarjeta de
    estructura por dataset, con el detalle de cada columna. Conserva
    las frases "derived from ..." que el corpus verifica.
    """
    members = [
        insight
        for insight in structure_insights
        if insight.title.lower().endswith("looks derived from other columns")
    ]

    if len(members) < 2:
        return structure_insights

    folded = Insight(
        title=f"{len(members)} columns look derived from other columns",
        message=" ".join(insight.message for insight in members),
        severity=(
            "warning"
            if any(insight.severity == "warning" for insight in members)
            else "info"
        ),
        category="structure",
        columns=[
            column
            for insight in members
            for column in insight.columns
        ],
        evidence={
            "folded_derived_columns": len(members),
            "derived_columns": ", ".join(
                insight.columns[0]
                for insight in members
                if insight.columns
            ),
        },
        score=max(
            float(len(members)),
            max(insight.score for insight in members),
        ),
        caveats=_unique_caveats(members),
    )

    return [
        insight
        for insight in structure_insights
        if id(insight) not in {id(member) for member in members}
    ] + [folded]
