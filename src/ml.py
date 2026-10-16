"""
ml.py

Detección de target, clasificación de problema y recomendación de modelos.

Este módulo:
- detecta candidatos a target (columna a predecir)
- clasifica el problema (binaria / multiclase / regresión)
- lista features candidatas y excluidas (ids, texto libre, columnas constantes)
- recomienda modelos de forma determinista (tabla de reglas, sin sklearn)

NO:
- entrena ni evalúa modelos (eso sería F7 futura, con scikit-learn)
- toca el DataFrame original
- importa sklearn
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.config import (
    ML_MAX_TARGET_CANDIDATES,
    ML_MIN_TARGET_CARDINALITY,
    ML_MULTICLASE_MAX_CLASSES,
    ML_REGRESSION_MIN_UNIQUE,
)
from src.profiling import ColumnProfile, DatasetProfile
from src.relationships import (
    CategoricalCategoricalRelationship,
    CategoricalNumericRelationship,
    NumericRelationship,
)

# ---------------------------------------------------------------------------
# Estructuras
# ---------------------------------------------------------------------------


@dataclass
class TargetCandidate:
    """Columna candidata a ser el target (lo que el usuario quiere predecir)."""

    column_name: str
    task_type: str  # "binary" | "multiclass" | "regression"
    n_unique: int
    pct_missing: float
    role: str
    # Número de otras columnas con las que se asocia (correlación o V).
    n_explanatory: int
    score: float  # puntuación heurística 0-1 (mayor = más probable target)
    reasons: list[str] = field(default_factory=list)


@dataclass
class ModelRecommendation:
    """Recomendación de modelo por tipo de problema y tamaño de dataset."""

    model: str
    task_type: str
    # Por qué este modelo encaja con este dataset.
    rationale: str
    # Cuándo priorizarlo sobre otros.
    when_to_use: str


@dataclass
class MLRecommendation:
    """Resultado completo de la recomendación ML para un dataset."""

    task_type: str | None  # None si no se detectó target
    target: str | None
    target_candidates: list[TargetCandidate] = field(default_factory=list)
    # Features sugeridas (excluye target, ids, texto libre, constantes).
    candidate_features: list[str] = field(default_factory=list)
    excluded_features: list[str] = field(default_factory=list)
    exclusion_reasons: dict[str, str] = field(default_factory=dict)
    recommended_models: list[ModelRecommendation] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Tabla de modelos (determinista, sin sklearn)
# ---------------------------------------------------------------------------

_MODEL_TABLE: list[dict] = [
    {
        "model": "Logistic Regression",
        "tasks": {"binary", "multiclass"},
        "min_rows": 100,
        "max_rows": None,
        "rationale": "Lineal, rápido, interpretable; buena línea base para datos tabulares pequeños/medianos.",
        "when_to_use": "Primera línea; si las features son mayormente numéricas o one-hot.",
    },
    {
        "model": "Random Forest",
        "tasks": {"binary", "multiclass", "regression"},
        "min_rows": 200,
        "max_rows": None,
        "rationale": "Captura no-linealidades e interacciones sin mucha ingeniería de features.",
        "when_to_use": "Dataset mediano-grande; mezcla de numéricas y categóricas.",
    },
    {
        "model": "XGBoost / LightGBM",
        "tasks": {"binary", "multiclass", "regression"},
        "min_rows": 1000,
        "max_rows": None,
        "rationale": "Ganadores en tabulares; maneja missing y categóricas bien.",
        "when_to_use": "Dataset grande (>1000 filas); prioriza sobre Random Forest.",
    },
    {
        "model": "Gradient Boosting (sklearn)",
        "tasks": {"binary", "multiclass", "regression"},
        "min_rows": 500,
        "max_rows": None,
        "rationale": "Boosting general-purpose sin dependencias extra.",
        "when_to_use": "Alternativa a XGBoost si no quieres dependencias externas.",
    },
    {
        "model": "k-Nearest Neighbors",
        "tasks": {"binary", "multiclass", "regression"},
        "min_rows": 100,
        "max_rows": 5000,
        "rationale": "Simple, no paramétrico; útil como referencia en datasets pequeños.",
        "when_to_use": "Solo datasets pequeños; se degrada con muchas features.",
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _classify_task(series: pd.Series) -> str | None:
    """Clasifica la columna target en binaria / multiclass / regression."""
    non_null = series.dropna()
    n_unique = int(non_null.nunique())

    if n_unique < ML_MIN_TARGET_CARDINALITY:
        return None

    if pd.api.types.is_bool_dtype(non_null) or n_unique == 2:
        return "binary"

    if not pd.api.types.is_numeric_dtype(non_null):
        if n_unique <= ML_MULTICLASE_MAX_CLASSES:
            return "multiclass"
        return None

    # Numérica: discreta con pocos valores → regresión si mucha variabilidad,
    # clasificación si pocos enteros únicos.
    if n_unique >= ML_REGRESSION_MIN_UNIQUE:
        return "regression"

    return None


def _count_explanatory(
    column_name: str,
    numeric_relationships: list[NumericRelationship],
    categorical_numeric: list[CategoricalNumericRelationship],
    categorical_categorical: list[CategoricalCategoricalRelationship],
) -> int:
    """Cuenta con cuántas otras columnas se asocia este candidato."""
    count = 0
    for rel in numeric_relationships:
        if rel.column_y == column_name or rel.column_x == column_name:
            count += 1
    for rel in categorical_numeric:
        if rel.column_categorical == column_name or rel.column_numeric == column_name:
            count += 1
    for rel in categorical_categorical:
        if rel.column_x == column_name or rel.column_y == column_name:
            count += 1
    return count


def _score_target(
    profile: ColumnProfile,
    n_explanatory: int,
    n_total_columns: int,
) -> tuple[float, list[str]]:
    """Puntuación heurística 0-1 de qué tan probable es que sea el target."""
    score = 0.0
    reasons: list[str] = []

    # Rol: boolean/categorical con cardinalidad razonable favorece target.
    if profile.role == "boolean":
        score += 0.35
        reasons.append("boolean (binary classification)")
    elif profile.role == "categorical":
        score += 0.15
        reasons.append("categorical")
    elif profile.role == "numeric":
        score += 0.1

    # Cardinalidad: pocos únicos relativo al total → más probable target.
    if profile.n_unique <= 20:
        score += 0.2
        reasons.append(f"low cardinality ({profile.n_unique} unique)")
    elif profile.n_unique <= 100:
        score += 0.1
    else:
        # Mucha variabilidad → regresión (price, revenue, etc.)
        score += 0.15
        reasons.append(f"high variability ({profile.n_unique} unique)")

    # Missingness: target con missing es raro (pero posible).
    if profile.pct_missing > 50:
        score -= 0.3
        reasons.append("high missingness")
    elif profile.pct_missing > 0:
        score -= 0.05

    # Explanatory power: asociada con muchas otras columnas → más probable target.
    if n_explanatory > 0:
        score += min(0.4, n_explanatory / max(n_total_columns - 1, 1) * 0.4)
        reasons.append(f"associated with {n_explanatory} columns")

    # Nombre sugestivo (solo keywords fuertes, no subcadenas ambiguas).
    name_lower = profile.name.lower()
    for keyword in ("target", "label", "outcome", "churn", "price", "sale", "income", "revenue"):
        if keyword in name_lower:
            score += 0.3
            reasons.append(f"column name contains '{keyword}'")
            break

    return max(0.0, min(1.0, score)), reasons


# ---------------------------------------------------------------------------
# Funciones principales
# ---------------------------------------------------------------------------


def detect_target_candidates(
    profile: DatasetProfile,
    df: pd.DataFrame,
    numeric_relationships: list[NumericRelationship],
    categorical_numeric: list[CategoricalNumericRelationship],
    categorical_categorical: list[CategoricalCategoricalRelationship],
) -> list[TargetCandidate]:
    """Detecta candidatos a target ordenados por score descendente."""
    candidates: list[TargetCandidate] = []
    n_columns = profile.n_columns

    for col in profile.columns:
        # Identificadores y texto libre nunca son target.
        if col.role in ("identifier", "text", "datetime", "unknown"):
            continue

        task_type = _classify_task(df[col.name])
        if task_type is None:
            continue

        n_explanatory = _count_explanatory(
            col.name, numeric_relationships, categorical_numeric, categorical_categorical
        )
        score, reasons = _score_target(col, n_explanatory, n_columns)

        candidates.append(
            TargetCandidate(
                column_name=col.name,
                task_type=task_type,
                n_unique=col.n_unique,
                pct_missing=col.pct_missing,
                role=col.role,
                n_explanatory=n_explanatory,
                score=round(score, 3),
                reasons=reasons,
            )
        )

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:ML_MAX_TARGET_CANDIDATES]


def recommend_models(
    task_type: str | None,
    n_rows: int,
) -> list[ModelRecommendation]:
    """Recomienda modelos por tipo de problema y tamaño de dataset."""
    if task_type is None:
        return []

    matches = []
    for entry in _MODEL_TABLE:
        if task_type not in entry["tasks"]:
            continue
        if entry["min_rows"] is not None and n_rows < entry["min_rows"]:
            continue
        if entry["max_rows"] is not None and n_rows > entry["max_rows"]:
            continue
        matches.append(
            ModelRecommendation(
                model=entry["model"],
                task_type=task_type,
                rationale=entry["rationale"],
                when_to_use=entry["when_to_use"],
            )
        )

    return matches


def build_ml_recommendation(
    profile: DatasetProfile,
    df: pd.DataFrame,
    numeric_relationships: list[NumericRelationship],
    categorical_numeric: list[CategoricalNumericRelationship],
    categorical_categorical: list[CategoricalCategoricalRelationship],
) -> MLRecommendation:
    """Construye la recomendación ML completa para un dataset."""
    candidates = detect_target_candidates(
        profile, df, numeric_relationships, categorical_numeric, categorical_categorical
    )

    if not candidates:
        return MLRecommendation(
            task_type=None,
            target=None,
            notes=["No clear target candidate detected."],
        )

    best = candidates[0]
    target_name = best.column_name

    # Features: todas las columnas excepto target, identificadores, texto libre y constantes.
    candidate_features: list[str] = []
    excluded_features: list[str] = []
    exclusion_reasons: dict[str, str] = {}

    for col in profile.columns:
        if col.name == target_name:
            excluded_features.append(col.name)
            exclusion_reasons[col.name] = "target"
            continue

        if col.role == "identifier":
            excluded_features.append(col.name)
            exclusion_reasons[col.name] = "identifier"
            continue

        if col.role == "text":
            excluded_features.append(col.name)
            exclusion_reasons[col.name] = "free text (needs NLP, not tabular ML)"
            continue

        if col.n_unique <= 1:
            excluded_features.append(col.name)
            exclusion_reasons[col.name] = "constant (no variance)"
            continue

        if col.pct_missing > 90:
            excluded_features.append(col.name)
            exclusion_reasons[col.name] = f"high missingness ({col.pct_missing}%)"
            continue

        candidate_features.append(col.name)

    models = recommend_models(best.task_type, profile.n_rows)

    notes: list[str] = []
    if best.score < 0.3:
        notes.append(
            f"Target '{target_name}' has a low confidence score ({best.score}); "
            "verify it matches your business question."
        )

    return MLRecommendation(
        task_type=best.task_type,
        target=target_name,
        target_candidates=candidates,
        candidate_features=candidate_features,
        excluded_features=excluded_features,
        exclusion_reasons=exclusion_reasons,
        recommended_models=models,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Generación de insights
# ---------------------------------------------------------------------------


def generate_ml_insights(recommendation: MLRecommendation) -> list:
    """Genera insights a partir de la recomendación ML."""
    from src.insights import Insight

    insights: list[Insight] = []

    if recommendation.task_type is None or recommendation.target is None:
        return insights

    task_labels = {
        "binary": "binary classification",
        "multiclass": "multiclass classification",
        "regression": "regression",
    }
    task_label = task_labels.get(
        recommendation.task_type, recommendation.task_type
    )

    # Insight principal: target + problema + features.
    features_summary = ", ".join(recommendation.candidate_features[:5])
    if len(recommendation.candidate_features) > 5:
        features_summary += f", … (+{len(recommendation.candidate_features) - 5} more)"

    insights.append(
        Insight(
            category="ml_recommendations",
            severity="info",
            title=f"ML target: {recommendation.target}",
            message=(
                f"Detected '{recommendation.target}' as the likely target for "
                f"{task_label} with {len(recommendation.candidate_features)} candidate "
                f"features ({features_summary}). "
                f"Excluded: {len(recommendation.excluded_features)} "
                f"(ids, text, constants)."
            ),
            columns=[recommendation.target],
        )
    )

    # Insight de modelos recomendados.
    if recommendation.recommended_models:
        model_names = ", ".join(m.model for m in recommendation.recommended_models[:3])
        insights.append(
            Insight(
                category="ml_recommendations",
                severity="info",
                title=f"Recommended models for {task_label}",
                message=(
                    f"Based on dataset size ({len(recommendation.candidate_features)} "
                    f"features), consider: {model_names}."
                ),
                columns=[recommendation.target],
            )
        )

    # Notas de bajo confidence.
    for note in recommendation.notes:
        insights.append(
            Insight(
                category="ml_recommendations",
                severity="info",
                title="ML target confidence",
                message=note,
                columns=[recommendation.target],
            )
        )

    return insights
