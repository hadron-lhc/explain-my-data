"""
equivalence.py

Detección de columnas redundantes (P1 de PLAN_V3).

Un dataset puede traer la misma información escrita dos veces:

- equivalencia exacta: dos columnas que renombran lo mismo
  (`survived`/`alive`, `embarked`/`embark_town`, `price`/`price_copy`);
- columna derivada: una categórica sintética construida a partir de
  otras (`who` a partir de sex + age, `adult_male` a partir de
  sex + age-adult).

Emitir una tarjeta por cada par de equivalentes multiplica el ruido
sin añadir sustancia: el fold conserva UNA tarjeta de equivalencia y
pliega el resto de insights del grupo, además de preferir la columna
"fuente" (sin missing, cardinalidad natural) al generar relaciones.

Todo es determinista y solo-aditivo: no modifica el DataFrame ni
altera las relaciones subyacentes, solo decide qué se emite.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from src.config import (
    DERIVED_MAX_CARDINALITY,
    EQUIVALENCE_MIN_CORR,
    EQUIVALENCE_MIN_CRAMERS_V,
)
from src.insights import Insight
from src.relationships import (
    CategoricalCategoricalRelationship,
    CategoricalNumericRelationship,
)

# Cramers V por debajo del piso de equivalencia pero por encima de
# este valor, combinado con evidencia funcional, sugiere una columna
# derivada de otra (p.ej. who ~ f(sex, age)).
DERIVED_HINT_MIN_CRAMERS_V = 0.9

# Filas mínimas (con los tres valores presentes) para aceptar una
# derivación candidata.
DERIVED_MIN_ROWS = 30

# Filas minimas por grupo del socio categorico para evaluar el test
# funcional. El minimo es estricto (>=) para descartar grupos chicos:
# un grupo minusculo no da evidencia suficiente de derivacion, y de
# paso evita falsos positivos cuando el "resto" es casi constante
# (p.ej. sex dentro de un socio de 3 niveles donde 2 son triviales).
DERIVED_MIN_GROUP_ROWS = 40

# Preferencia de "fuente" frente a "derivada" (mayor es mejor).
_SOURCE_PREFERENCE_MISSING = 2.0
_SOURCE_PREFERENCE_UNIQUE = 1.0


@dataclass(frozen=True)
class EquivalenceRelation:
    """Un par de columnas que contienen la misma informacion."""

    column_source: str
    column_duplicate: str
    kind: str  # cat_cat | num_num | cat_num
    strength: float
    n_shared: int
    label: str  # "contains the same information as"


@dataclass(frozen=True)
class DerivedColumn:
    """Una categorica que parece derivada de un grupo de columnas."""

    column: str
    derived_from: tuple[str, ...]
    evidence: str


@dataclass
class EquivalenceMap:
    """Resultado de la deteccion de redundancia sobre un dataset."""

    relations: list[EquivalenceRelation] = field(default_factory=list)
    derived: list[DerivedColumn] = field(default_factory=list)

    @property
    def duplicate_columns(self) -> set[str]:
        """Columnas plegadas: no son fuente de nada nuevo."""
        duplicates = {relation.column_duplicate for relation in self.relations}
        duplicates.update(item.column for item in self.derived)
        return duplicates

    @property
    def source_by_duplicate(self) -> dict[str, str]:
        return {
            relation.column_duplicate: relation.column_source
            for relation in self.relations
        }

    @property
    def derived_by_column(self) -> dict[str, DerivedColumn]:
        return {item.column: item for item in self.derived}

    def is_folded(self, column_a: str, column_b: str) -> bool:
        """True si el par completo esta cubierto por un fold."""
        source_by_duplicate = self.source_by_duplicate
        derived_by_column = self.derived_by_column

        def root(column: str) -> str:
            if column in source_by_duplicate:
                return source_by_duplicate[column]
            if column in derived_by_column:
                return derived_by_column[column].derived_from[0]
            return column

        return root(column_a) == root(column_b)


def _prepare(series: pd.Series, role: str) -> pd.Series | None:
    if role == "numeric":
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().sum() < 2 or numeric.nunique(dropna=True) < 2:
            return None
        return numeric

    # Categorica: recorta pero conserva los nulos, para que la
    # preferencia de fuente (menos missing) pueda puntuarlos. Las
    # comparaciones descartan nulos con dropna/shared_mask aparte.
    cleaned = series.astype("string").str.strip()
    if cleaned.notna().sum() < 2 or cleaned.nunique(dropna=True) < 2:
        return None
    return cleaned


def _shared_mask(series: pd.Series, other: pd.Series) -> pd.Series:
    """Mascara booleana (aligneada) donde ambas columnas tienen valor."""
    left = series.notna()
    right = other.reindex(series.index).notna()
    return left & right


def _cramers_v(x: pd.Series, y: pd.Series) -> tuple[float, int] | None:
    data = pd.DataFrame({"x": x, "y": y}).dropna()

    if len(data) < 2:
        return None

    contingency = pd.crosstab(data["x"], data["y"])
    n_rows, n_cols = contingency.shape

    if n_rows < 2 or n_cols < 2:
        return None

    try:
        # Sin correccion de Yates: para deteccion de equivalencia
        # interesa el V estandar sin decaer con n chico.
        chi2, _, _, _ = chi2_contingency(contingency, correction=False)
    except ValueError:
        return None

    n = int(contingency.to_numpy().sum())

    if n == 0:
        return None

    min_dim = min(n_rows, n_cols)
    value = float(np.sqrt(chi2 / (n * (min_dim - 1))))
    return value, n


def _one_to_one(x: pd.Series, y: pd.Series) -> bool:
    """True si cada valor de x corresponde a un unico valor de y y
    viceversa (renombre puro a pesar de codigos distintos)."""
    data = pd.DataFrame({"x": x, "y": y}).dropna()

    if data.empty:
        return False

    return data.groupby("x")["y"].nunique().max() == 1 and (
        data.groupby("y")["x"].nunique().max() == 1
    )


def _detect_cat_cat(
    profiles: dict[str, str],
    prepared: dict[str, pd.Series],
) -> list[EquivalenceRelation]:
    categoricals = [
        name
        for name, role in profiles.items()
        if role in {"categorical", "boolean"} and name in prepared
    ]

    relations: list[EquivalenceRelation] = []

    for i, column_a in enumerate(categoricals):
        for column_b in categoricals[i + 1 :]:
            series_a = prepared[column_a]
            series_b = prepared[column_b]

            shared = _shared_mask(series_a, series_b)
            shared_series_a = series_a[shared]
            shared_series_b = series_b.reindex(shared_series_a.index)

            result = _cramers_v(shared_series_a, shared_series_b)

            if result is None:
                continue

            value, n = result

            if value < EQUIVALENCE_MIN_CRAMERS_V:
                continue

            if not _one_to_one(shared_series_a, shared_series_b):
                continue

            source, duplicate = _prefer_source(
                column_a,
                column_b,
                profiles,
                prepared,
            )

            relations.append(
                EquivalenceRelation(
                    column_source=source,
                    column_duplicate=duplicate,
                    kind="cat_cat",
                    strength=value,
                    n_shared=n,
                    label="contains the same information as",
                )
            )

    return relations


def _detect_num_num(
    profiles: dict[str, str],
    prepared: dict[str, pd.Series],
) -> list[EquivalenceRelation]:
    numerics = [
        name for name, role in profiles.items() if role == "numeric" and name in prepared
    ]

    relations: list[EquivalenceRelation] = []

    for i, column_a in enumerate(numerics):
        for column_b in numerics[i + 1 :]:
            data = pd.DataFrame(
                {
                    "x": pd.to_numeric(prepared[column_a], errors="coerce"),
                    "y": pd.to_numeric(prepared[column_b], errors="coerce"),
                }
            ).dropna()

            if len(data) < 2:
                continue

            if data["x"].nunique() < 2 or data["y"].nunique() < 2:
                continue

            pearson = data["x"].corr(data["y"], method="pearson")

            if pd.isna(pearson) or abs(float(pearson)) < EQUIVALENCE_MIN_CORR:
                continue

            # Una copia debe ser biyectiva a nivel de valor: dos
            # columnas casi proporcionales pero no biyectivas (p.ej.
            # `orders ≈ revenue/10` con ruido) no son equivalencia.
            if not _one_to_one(data["x"], data["y"]):
                continue

            source, duplicate = _prefer_source(
                column_a,
                column_b,
                profiles,
                prepared,
            )

            relations.append(
                EquivalenceRelation(
                    column_source=source,
                    column_duplicate=duplicate,
                    kind="num_num",
                    strength=abs(float(pearson)),
                    n_shared=len(data),
                    label="is a linear copy of",
                )
            )

    return relations


def _detect_cat_num(
    profiles: dict[str, str],
    prepared: dict[str, pd.Series],
    relationships: list[CategoricalNumericRelationship],
    min_groups: int = 2,
    max_groups: int = 20,
) -> list[EquivalenceRelation]:
    """
    Una categórica que codifica exactamente una numerica discreta
    (p.ej. `class` etiqueta a `pclass`) es equivalente: los valores
    de la numerica son función de los grupos de la categórica.
    """
    relations: list[EquivalenceRelation] = []

    for relationship in relationships:
        column_cat = relationship.column_categorical
        column_num = relationship.column_numeric

        if column_cat not in prepared or column_num not in prepared:
            continue

        if profiles.get(column_cat) not in {"categorical", "boolean"}:
            continue

        if profiles.get(column_num) != "numeric":
            continue

        n_groups = relationship.n_groups

        if n_groups < min_groups or n_groups > max_groups:
            continue

        data = pd.DataFrame(
            {
                "cat": prepared[column_cat].reindex(
                    prepared[column_num].index
                ),
                "num": pd.to_numeric(prepared[column_num], errors="coerce"),
            }
        ).dropna()

        if len(data) < 2:
            continue

        # La numerica debe ser discreta y tener exactamente un valor
        # por grupo (sin solapes): es un codificado de la categórica.
        num_values = data["num"].nunique()

        if num_values != n_groups:
            continue

        if data.groupby("cat")["num"].nunique().max() != 1:
            continue

        if data.groupby("num")["cat"].nunique().max() != 1:
            continue

        # `class` etiqueta a `pclass` (p.ej. First/1.0): la numérica es
        # el código y la categórica la etiqueta. Se prefiere la numérica
        # como fuente porque conserva la señal analítica (correlaciones,
        # distribuciones, inferencia) que la etiqueta no aporta.
        relations.append(
            EquivalenceRelation(
                column_source=column_num,
                column_duplicate=column_cat,
                kind="cat_num",
                strength=1.0,
                n_shared=len(data),
                label="encodes the same information as",
            )
        )

    return relations


def _prefer_source(
    column_a: str,
    column_b: str,
    profiles: dict[str, str],
    prepared: dict[str, pd.Series],
) -> tuple[str, str]:
    """Elige la columna fuente: menos missing y cardinalidad natural."""
    score_a = _source_preference(column_a, profiles, prepared)
    score_b = _source_preference(column_b, profiles, prepared)

    if score_a != score_b:
        return (
            (column_a, column_b) if score_a > score_b else (column_b, column_a)
        )

    # Empate: gana el nombre alfabéticamente (determinista).
    return tuple(sorted((column_a, column_b)))  # type: ignore[return-value]


def _source_preference(
    column: str,
    profiles: dict[str, str],
    prepared: dict[str, pd.Series],
) -> float:
    series = prepared.get(column)

    if series is None:
        return 0.0

    total = len(series)

    if total == 0:
        return 0.0

    missing_ratio = float(series.isna().sum()) / total
    missing_score = (1.0 - missing_ratio) * _SOURCE_PREFERENCE_MISSING

    unique = int(series.nunique(dropna=True))
    unique_score = (
        _SOURCE_PREFERENCE_UNIQUE
        if unique <= DERIVED_MAX_CARDINALITY
        else 0.0
    )

    return missing_score + unique_score


def _is_step_function(numeric: pd.Series, labels: pd.Series, max_runs: int) -> bool:
    """
    True si `labels` es una función escalonada de `numeric` ordenado:
    al ordenar por la numerica, las etiquetas forman a lo sumo
    `max_runs` tramos contiguos (p.ej. who: niños al principio,
    adultos después, dentro de cada sexo).
    """
    data = pd.DataFrame({"n": numeric, "label": labels}).dropna()

    if len(data) < 2:
        return False

    ordered = data.sort_values("n", kind="stable")
    changes = int((ordered["label"] != ordered["label"].shift()).sum())

    return 1 <= changes <= max_runs


def _detect_derived(
    profiles: dict[str, str],
    prepared: dict[str, pd.Series],
    cat_cat: list[CategoricalCategoricalRelationship],
    equivalence_columns: set[str],
) -> list[DerivedColumn]:
    """
    Heurística sin LLM y determinista: una categórica de cardinalidad
    baja con V alto con OTRA categórica es candidata a derivada si
    existe una numérica tal que, dentro de cada valor del socio, la
    candidata es una función escalonada de la numérica
    (p.ej. who = f(sex, age)). El test es direccional: who pasa con
    (sex, age), pero sex NO pasa con (who, age) — así la fuente se
    conserva y la derivada se pliega.
    """
    strong_partners: dict[str, list[tuple[str, float]]] = {}

    for relationship in cat_cat:
        if relationship.cramers_v < DERIVED_HINT_MIN_CRAMERS_V:
            continue

        for column, partner in (
            (relationship.column_x, relationship.column_y),
            (relationship.column_y, relationship.column_x),
        ):
            if column in equivalence_columns or partner in equivalence_columns:
                continue

            if profiles.get(column) not in {"categorical", "boolean"}:
                continue

            strong_partners.setdefault(column, []).append(
                (partner, relationship.cramers_v)
            )

    numerics = sorted(
        name
        for name, role in profiles.items()
        if role == "numeric" and name in prepared
    )

    derived: list[DerivedColumn] = []

    for column in sorted(strong_partners):
        series = prepared.get(column)

        if series is None:
            continue

        levels = int(series.nunique(dropna=True))

        if levels < 2 or levels > DERIVED_MAX_CARDINALITY:
            continue

        # El socio con el vínculo más débil (pero ≥ piso) es el más
        # "fuente"-like del clúster: who/sex/adult_male son un clúster
        # tautológico donde el eslabón más flojo apunta al original.
        candidates = sorted(
            set(strong_partners[column]),
            key=lambda item: (item[1], item[0]),
        )

        found: tuple[str, str] | None = None

        for partner, _v in candidates:
            partner_series = prepared.get(partner)

            if partner_series is None:
                continue

            for numeric_name in numerics:
                numeric_series = pd.to_numeric(
                    prepared[numeric_name],
                    errors="coerce",
                )

                frame = pd.DataFrame(
                    {
                        "a": partner_series,
                        "c": series,
                        "n": numeric_series,
                    }
                ).dropna()

                if len(frame) < DERIVED_MIN_ROWS:
                    continue

                evaluated = 0
                ok = True

                for _, group in frame.groupby("a"):
                    if len(group) < DERIVED_MIN_GROUP_ROWS:
                        continue

                    evaluated += 1

                    if not _is_step_function(group["n"], group["c"], levels):
                        ok = False
                        break

                if ok and evaluated > 0:
                    found = (partner, numeric_name)
                    break

            if found is not None:
                break

        if found is not None:
            partner, numeric_name = found
            derived.append(
                DerivedColumn(
                    column=column,
                    derived_from=(partner, numeric_name),
                    evidence=(
                        f"values follow {numeric_name} within each "
                        f"{partner} group (tied to {partner} with "
                        f"high association)"
                    ),
                )
            )

    # Ambigüedad direccional: si A parece derivada de B y B de A,
    # ninguna es claramente la fuente. Se descartan ambas (quedan
    # como columnas independientes) en vez de plegar una al azar.
    mutual = {
        item.column
        for item in derived
        if any(
            other.column == item.derived_from[0]
            and other.derived_from[0] == item.column
            for other in derived
        )
    }

    if mutual:
        derived = [item for item in derived if item.column not in mutual]

    return derived


def detect_equivalences(
    profiles: dict[str, str],
    prepared_columns: dict[str, pd.Series],
    cat_cat: list[CategoricalCategoricalRelationship] | None = None,
    cat_num: list[CategoricalNumericRelationship] | None = None,
) -> EquivalenceMap:
    """
    Detecta columnas equivalentes y derivadas de un dataset.

    `profiles` mapea nombre → rol. `prepared_columns` mapea nombre →
    serie preparada. Las relaciones (si se pasan) alimentan la
    detección cat↔num y de derivadas; no son obligatorias.
    """
    cat_cat = cat_cat or []
    cat_num = cat_num or []

    cat_cat_relations = _detect_cat_cat(profiles, prepared_columns)
    num_num_relations = _detect_num_num(profiles, prepared_columns)
    cat_num_relations = _detect_cat_num(
        profiles,
        prepared_columns,
        cat_num,
    )

    already_equivalent = {
        name
        for relation in cat_cat_relations + num_num_relations + cat_num_relations
        for name in (relation.column_source, relation.column_duplicate)
    }

    derived = _detect_derived(
        profiles,
        prepared_columns,
        cat_cat,
        already_equivalent,
    )

    relations = cat_cat_relations + num_num_relations + cat_num_relations

    # Orden determinista: fuente, duplicado, tipo.
    relations.sort(
        key=lambda relation: (
            relation.column_source,
            relation.column_duplicate,
            relation.kind,
        )
    )
    derived.sort(key=lambda item: item.column)

    return EquivalenceMap(relations=relations, derived=derived)


def equivalence_card(
    relations: list[EquivalenceRelation],
) -> Insight | None:
    """Una tarjeta "X e Y contienen la misma información" por fold."""
    if not relations:
        return None

    primary = max(relations, key=lambda relation: relation.strength)
    extras = [
        f"{relation.column_source} and {relation.column_duplicate}"
        for relation in relations
        if relation is not primary
    ]

    message = (
        f"{primary.column_source} {primary.label} "
        f"{primary.column_duplicate}."
    )

    if extras:
        message += f" Also equivalent: {'; '.join(extras)}."

    message += " Redundant columns were folded into one card."

    return Insight(
        title="Redundant columns carry the same information",
        message=message,
        severity="info",
        category="structure",
        columns=sorted(
            {primary.column_source, primary.column_duplicate}
        ),
        evidence={
            "n_equivalences": len(relations),
            "primary_strength": primary.strength,
        },
        score=primary.strength,
        caveats=["Redundant columns do not add independent signal."],
    )


def filter_insights(
    insights: list[Insight],
    equivalence_map: EquivalenceMap,
) -> list[Insight]:
    """
    Pliega los insights que solo repiten información de una columna
    redundante. Conserva la tarjeta de equivalencia y descarta las
    que nacen de una columna derivada o de un par ya cubierto por un
    fold de equivalencia exacta.
    """
    if not equivalence_map.relations and not equivalence_map.derived:
        return list(insights)

    duplicates = equivalence_map.duplicate_columns
    kept: list[Insight] = []

    for insight in insights:
        columns = [column.lower() for column in insight.columns]

        if not columns:
            kept.append(insight)
            continue

        # Una tarjeta que toca una columna redundante (duplicada o
        # derivada) repite información de la fuente: se pliega. La
        # tarjeta de equivalencia/derivación se emite aparte.
        if any(column in duplicates for column in columns):
            continue

        # Una tarjeta de dos+ columnas cuyo par comparte raíz (una es
        # la fuente de la otra) también es redundante.
        if len(columns) >= 2 and equivalence_map.is_folded(
            columns[0],
            columns[1],
        ):
            continue

        kept.append(insight)

    return kept


def filter_categorical_numeric_pairs(
    pairs: list[tuple[str, str]],
    equivalence_map: EquivalenceMap,
) -> list[tuple[str, str]]:
    """
    Excluye los pares (cat, num) donde alguna columna es redundante
    (duplicada o derivada): sus insights se pliegan en la fuente
    (preferencia fuente > derivada al generar).
    """
    if not equivalence_map.relations and not equivalence_map.derived:
        return list(pairs)

    duplicates = equivalence_map.duplicate_columns

    return [
        (column_cat, column_num)
        for column_cat, column_num in pairs
        if column_cat not in duplicates and column_num not in duplicates
    ]


def filter_relationships(
    relationships: list[CategoricalNumericRelationship],
    equivalence_map: EquivalenceMap,
) -> list[CategoricalNumericRelationship]:
    """Descarta relaciones donde alguna columna es redundante."""
    if not equivalence_map.relations and not equivalence_map.derived:
        return list(relationships)

    duplicates = equivalence_map.duplicate_columns

    return [
        relationship
        for relationship in relationships
        if relationship.column_categorical not in duplicates
        and relationship.column_numeric not in duplicates
    ]
