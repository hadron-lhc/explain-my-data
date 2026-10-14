"""
profiling.py

Responde una sola pregunta, sin conocer el dominio del dataset:

    "¿Qué tengo delante?"

No decide qué análisis hacer (eso es responsabilidad de otra capa,
por ejemplo analysis_plan.py). Solo observa y describe.

Distinción central: dtype != role analítico.
Una columna puede ser int64 y aun así no tener sentido promediarla
(ej: customer_id).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

# Roles que se analizan como variables categóricas de baja cardinalidad.
CATEGORICAL_LIKE_ROLES = frozenset({"categorical", "boolean"})

# ---------------------------------------------------------------------------
# Estructura de "hechos" (Facts)
# ---------------------------------------------------------------------------


@dataclass
class ColumnProfile:
    name: str
    dtype: str  # dtype crudo de pandas (lo que YA sabemos)
    role: str  # rol analítico inferido (lo que DEDUCIMOS)
    n_missing: int
    pct_missing: float
    n_unique: int
    pct_unique: float
    sample_values: list
    notes: list[str] = field(default_factory=list)
    is_discrete: bool = False


@dataclass
class DatasetProfile:
    n_rows: int
    n_columns: int
    n_duplicates: int
    columns: list[ColumnProfile]


# ---------------------------------------------------------------------------
# Heurísticas de inferencia de rol
# ---------------------------------------------------------------------------

_ID_NAME_PATTERN = re.compile(
    r"(^id$|_id$|^id_|uuid|guid|^index$|^idx$|^key$|_key$|^code$|_code$)",
    re.IGNORECASE,
)

_BOOLEAN_LIKE_SETS = [
    {"0", "1"},
    {"true", "false"},
    {"yes", "no"},
    {"y", "n"},
    {"t", "f"},
    {"si", "no"},
]

_IDENTIFIER_UNIQUENESS_THRESHOLD = 0.95


def _is_textual(series: pd.Series) -> bool:
    if pd.api.types.is_string_dtype(series):
        return True
    if not pd.api.types.is_object_dtype(series):
        return False

    non_null = series.dropna()

    if non_null.empty:
        return False

    return non_null.map(lambda value: isinstance(value, str)).all()


def _looks_like_identifier_by_name(name: str) -> bool:
    return bool(_ID_NAME_PATTERN.search(name))


def _looks_like_identifier_by_uniqueness(n_unique: int, n: int) -> bool:
    """Solo unicidad casi total, sin considerar el nombre. Se usa como
    fallback y debe evaluarse después de descartar fechas: una columna
    de fechas también puede tener unicidad ~100% sin ser un identificador
    """
    if n <= 20:
        return False

    return n_unique / n > _IDENTIFIER_UNIQUENESS_THRESHOLD


def _looks_like_numeric(series: pd.Series, sample_size: int = 50) -> bool:
    sample = series.dropna().astype(str).head(sample_size)

    if sample.empty:
        return False

    parsed = pd.to_numeric(sample, errors="coerce")

    return parsed.notna().mean() > 0.9


def _is_integer_like(series: pd.Series) -> bool:
    """Detecta si los valores numéricos de la
    columna son enteros aunque el dtype no lo sea."""
    if pd.api.types.is_integer_dtype(series):
        return True
    if pd.api.types.is_float_dtype(series):
        non_null = series.dropna()
        return len(non_null) > 0 and (non_null % 1 == 0).all()
    return False


def _looks_like_boolean(series: pd.Series) -> bool:
    uniques = series.dropna().unique()

    if len(uniques) != 2:
        return False

    lowered = {str(v).strip().lower() for v in uniques}

    return lowered in _BOOLEAN_LIKE_SETS


def _looks_like_datetime(series: pd.Series, sample_size: int = 50) -> bool:
    sample = series.dropna().astype(str).head(sample_size)

    if sample.empty:
        return False

    if sample.str.fullmatch(r"\d+").mean() > 0.8:
        return False

    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    return parsed.notna().mean() > 0.9


def _is_discrete_numeric(series: pd.Series) -> bool:
    numeric = pd.to_numeric(series, errors="coerce").dropna()

    if numeric.empty:
        return False

    values_are_integer_like = (numeric % 1 == 0).all()

    if not values_are_integer_like:
        return False

    return numeric.nunique() <= 20


def infer_role(series: pd.Series, name: str) -> tuple[str, list[str]]:
    """
    Orden:
            id_name
                ↓
            boolean
                ↓
            datetime
                ↓
            numeric
                ↓
            id_uniqueness
                ↓
            categorical/textual
    """

    notes: list[str] = []
    n = len(series)
    n_unique = int(series.nunique(dropna=True))

    # 1. Identificador por nombre
    if _looks_like_identifier_by_name(name):
        notes.append(
            "El nombre sugiere que la variable es un identificador (ej. id, uuid, code)."
        )
        return "identifier", notes

    # 2. Booleano
    if _looks_like_boolean(series):
        notes.append("Exactamente dos valores distintos: variable booleana.")
        return "boolean", notes

    # 3. Numérico real
    if pd.api.types.is_numeric_dtype(series):
        return "numeric", notes

    # 4. DateTime ya tipado por pandas
    if pd.api.types.is_datetime64_dtype(series):
        return "datetime", notes

    # 5. Fecha disfrazada de texto
    if _is_textual(series) and _looks_like_datetime(series):
        notes.append(
            "Los valores son texto pero parsean como fecha de forma "
            "consistente; considerar convertir a datetime."
        )
        return "datetime", notes

    # 6. Numérico disfrazado de texto
    if _is_textual(series) and _looks_like_numeric(series):
        notes.append(
            "Los valores son texto pero parecen representar números; "
            "considerar convertir a tipo numérico."
        )
        return "numeric", notes

    # 7. Texto libre vs identificador por unicidad vs categórica
    if _is_textual(series):
        non_null = series.dropna().astype(str)
        avg_len = non_null.str.len().mean() if len(non_null) else 0
        uniqueness_ratio = n_unique / n if n else 0

        # Unicidad alta + strings largos = texto libre, no identificador.
        if uniqueness_ratio > 0.5 and avg_len > 20:
            notes.append(
                "Alta cardinalidad y strings largos: probablemente texto "
                "libre, no una categoría."
            )
            return "text", notes

        if _looks_like_identifier_by_uniqueness(n_unique, n):
            notes.append(
                "Unicidad casi total sin nombre sugestivo: probable "
                "identificador (ej. UUID, código de referencia)."
            )
            return "identifier", notes

        return "categorical", notes

    notes.append("No encajó en ninguna heurística conocida; revisar manualmente.")
    return "unknown", notes


# ---------------------------------------------------------------------------
# Perfilado de columnas y dataset completo
# ---------------------------------------------------------------------------


def profile_column(series: pd.Series, name: str) -> ColumnProfile:
    n = len(series)
    n_missing = int(series.isna().sum())
    n_unique = int(series.nunique(dropna=True))
    role, notes = infer_role(series, name)

    sample = series.dropna().unique()[:5].tolist()

    return ColumnProfile(
        name=name,
        dtype=str(series.dtype),
        role=role,
        n_missing=n_missing,
        pct_missing=round(n_missing / n * 100, 2) if n else 0.0,
        n_unique=n_unique,
        pct_unique=round(n_unique / n * 100, 2) if n else 0.0,
        sample_values=sample,
        notes=notes,
        is_discrete=_is_discrete_numeric(series),
    )


def profile_dataset(df: pd.DataFrame) -> DatasetProfile:
    """Construye un perfil completo del dataset."""
    columns = [profile_column(df[col], col) for col in df.columns]

    return DatasetProfile(
        n_rows=len(df),
        n_columns=len(df.columns),
        n_duplicates=int(df.duplicated().sum()),
        columns=columns,
    )


def profile_summary(profiles: list[ColumnProfile]) -> pd.DataFrame:
    """Vista tabular rápida para inspección humana."""
    return pd.DataFrame(
        [
            {
                "column": p.name,
                "dtype": p.dtype,
                "role": p.role,
                "missing_%": p.pct_missing,
                "unique_%": p.pct_unique,
                "n_unique": p.n_unique,
            }
            for p in profiles
        ]
    )
