"""
Curación de insights: un hallazgo = una tarjeta.

Dos reglas, ambas deterministas y solo-aditivas en información:

1. Cap de patrones por par (PATTERN_PER_PAIR_CAP): un par (grupo,
   numérico) con varios grupos ≥1σ genera un insight por grupo; el
   usuario recibe los `cap` más estandarizados (los extremos) y el
   resto se sintetiza en la tarjeta de relationship si existe.

2. Dedupe relationship↔pattern: "X differs across Y groups" y
   "Group 'G' of Y stands out in X" cuentan el mismo hallazgo por dos
   rutas. Cuando el par ya tiene un patrón sobreviviente, la tarjeta
   relationship se pliega: su group_separation pasa a la evidencia y
   a un caveat del patrón primario (el de mayor score).
"""

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

        n_groups = folded.evidence.get("n_groups")

        if isinstance(n_groups, (int, float)):
            primary.evidence["n_groups"] = int(n_groups)

    return kept


def curate_insights(
    relationship_insights: list[Insight],
    pattern_insights: list[Insight],
) -> tuple[list[Insight], list[Insight]]:
    """Aplica el cap de patrones y el dedupe relationship↔pattern."""
    patterns = cap_pattern_insights(pattern_insights)
    relationships = fold_group_differs_insights(relationship_insights, patterns)

    return relationships, patterns
