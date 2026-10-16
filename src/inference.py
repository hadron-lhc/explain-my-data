"""
Inferencia estadística (F4): de "se separan" a "con qué confianza
y tamaño de efecto".

Para cada par categórica↔numérica (cat-num) se calcula:
- teste de significancia: ANOVA de un factor si se cumplen las
  suposiciones (Levene para homocedasticidad + Shapiro sobre
  residuales para normalidad); si no, Kruskal-Wallis (robusto).
- tamaño de efecto: η² global (SS_between/SS_total) y Cohen's d de
  un contraste simplificado "mejor grupo vs resto" con su IC95%.
- corrección Benjamini-Hochberg sobre la familia de tests del
  dataset (todos los pares cat-num probados).

No sustituye a `group_separation` (sigue siendo el score de
selección del planner): la inferencia lo complementa.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import f_oneway, kruskal, levene, shapiro, t

from src.config import (
    INFERENCE_ALPHA,
    INFERENCE_BH_ENABLED,
    INFERENCE_D_NEGLIGIBLE,
    INFERENCE_ETA2_NEGLIGIBLE,
    INFERENCE_MAX_TESTS,
    INFERENCE_MIN_GROUPS,
    INFERENCE_MIN_OBSERVATIONS,
    INFERENCE_SMALL_SAMPLE,
)
from src.insights import Insight
from src.relationships import CategoricalNumericRelationship

# Shapiro tiene un límite interno de 5000; por encima se muestrea.
_SHAPIRO_MAX_N = 5000
_SHAPIRO_RANDOM_STATE = 42


@dataclass
class CategoricalNumericInference:
    """Resultado inferencial de un par cat-num."""

    column_categorical: str
    column_numeric: str

    n_observations: int
    n_groups: int

    test_used: str  # "anova" | "kruskal"
    statistic: float
    p_value: float
    p_value_bh: float | None = None

    eta_squared: float = 0.0
    cohen_d: float | None = None
    mean_difference: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None

    top_group: str | None = None
    top_mean: float | None = None
    rest_mean: float | None = None

    levene_p: float | None = None
    shapiro_p: float | None = None

    @property
    def significant(self) -> bool:
        threshold = (
            self.p_value_bh
            if self.p_value_bh is not None
            else self.p_value
        )
        return threshold < INFERENCE_ALPHA

    @property
    def small_sample(self) -> bool:
        return self.n_observations < INFERENCE_SMALL_SAMPLE

    @property
    def negligible_effect(self) -> bool:
        return (
            self.eta_squared < INFERENCE_ETA2_NEGLIGIBLE
            and (self.cohen_d is None or abs(self.cohen_d) < INFERENCE_D_NEGLIGIBLE)
        )


def benjamini_hochberg(pvalues: list[float]) -> list[float]:
    """Ajuste BH estándar; devuelve los p-values ajustados alineados."""
    n = len(pvalues)

    if n == 0:
        return []

    order = sorted(range(n), key=lambda i: pvalues[i])
    adjusted = [0.0] * n
    prev = 1.0

    for rank_index in range(n - 1, -1, -1):
        original_index = order[rank_index]
        rank = rank_index + 1
        value = pvalues[original_index] * n / rank
        prev = min(prev, value)
        adjusted[original_index] = min(prev, 1.0)

    return adjusted


def _eta_squared(groups: list[np.ndarray]) -> float:
    """η² = SS_between / SS_total."""
    all_values = np.concatenate(groups)
    grand_mean = all_values.mean()
    ss_total = float(((all_values - grand_mean) ** 2).sum())

    if ss_total == 0:
        return 0.0

    ss_between = 0.0

    for group in groups:
        ss_between += len(group) * float((group.mean() - grand_mean) ** 2)

    return ss_between / ss_total


def _top_vs_rest(
    groups: dict[str, np.ndarray],
) -> tuple[str, float, float, float, float | None, float | None, float | None]:
    """Contraste 'mejor grupo (mayor media) vs resto'.

    Retorna (top_group, top_mean, rest_mean, mean_diff, cohen_d,
    ci_low, ci_high)."""
    means = {name: float(values.mean()) for name, values in groups.items()}
    top_group = max(means, key=lambda name: means[name])
    top_values = groups[top_group]

    rest_values = np.concatenate(
        [values for name, values in groups.items() if name != top_group]
    )

    top_mean = float(top_values.mean())
    rest_mean = float(rest_values.mean())
    mean_diff = top_mean - rest_mean

    if len(top_values) < 2 or len(rest_values) < 2:
        return top_group, top_mean, rest_mean, mean_diff, None, None, None

    var_top = float(top_values.var(ddof=1))
    var_rest = float(rest_values.var(ddof=1))

    pooled = np.sqrt(
        ((len(top_values) - 1) * var_top + (len(rest_values) - 1) * var_rest)
        / (len(top_values) + len(rest_values) - 2)
    )

    cohen_d = float(mean_diff / pooled) if pooled > 0 else None

    # IC95% de la diferencia de medias (Welch).
    se = np.sqrt(var_top / len(top_values) + var_rest / len(rest_values))

    if se == 0:
        return top_group, top_mean, rest_mean, mean_diff, cohen_d, None, None

    numerator = (var_top / len(top_values) + var_rest / len(rest_values)) ** 2
    denominator = (var_top / len(top_values)) ** 2 / (len(top_values) - 1) + (
        var_rest / len(rest_values)
    ) ** 2 / (len(rest_values) - 1)
    df = numerator / denominator if denominator > 0 else 1.0
    margin = float(t.ppf(0.975, df) * se)

    return (
        top_group,
        top_mean,
        rest_mean,
        mean_diff,
        cohen_d,
        float(mean_diff - margin),
        float(mean_diff + margin),
    )


def _assumptions(data: pd.DataFrame) -> tuple[float | None, float | None]:
    """Levene (homocedasticidad) y Shapiro (normalidad de residuales)."""
    groups = [
        group.to_numpy(dtype=float)
        for _, group in data.groupby("categorical")["numeric"]
    ]

    if any(len(values) < 2 for values in groups):
        return None, None

    try:
        levene_p = float(levene(*groups, center="median").pvalue)
    except ValueError:
        levene_p = None

    if levene_p is not None and not np.isfinite(levene_p):
        levene_p = None

    residuals = data["numeric"] - data.groupby("categorical")["numeric"].transform("mean")
    sample = residuals

    if len(residuals) > _SHAPIRO_MAX_N:
        sample = residuals.sample(_SHAPIRO_MAX_N, random_state=_SHAPIRO_RANDOM_STATE)

    try:
        shapiro_p = float(shapiro(sample).pvalue)
    except ValueError:
        shapiro_p = None

    if shapiro_p is not None and not np.isfinite(shapiro_p):
        shapiro_p = None

    return levene_p, shapiro_p


def infer_categorical_numeric(
    df: pd.DataFrame,
    relationship: CategoricalNumericRelationship,
) -> CategoricalNumericInference | None:
    """Corre el test inferencial (ANOVA o Kruskal) para un par cat-num."""
    column_categorical = relationship.column_categorical
    column_numeric = relationship.column_numeric

    if (
        column_categorical not in df.columns
        or column_numeric not in df.columns
    ):
        return None

    data = pd.DataFrame(
        {
            "numeric": pd.to_numeric(df[column_numeric], errors="coerce"),
            "categorical": df[column_categorical],
        }
    ).dropna()

    if len(data) < INFERENCE_MIN_OBSERVATIONS:
        return None

    grouped = {
        str(name): group.to_numpy(dtype=float)
        for name, group in data.groupby("categorical")["numeric"]
        if len(group) >= 2
    }

    if len(grouped) < INFERENCE_MIN_GROUPS:
        return None

    arrays = list(grouped.values())
    eta_squared = _eta_squared(arrays)
    levene_p, shapiro_p = _assumptions(data)

    assumptions_ok = (
        levene_p is not None
        and shapiro_p is not None
        and levene_p >= INFERENCE_ALPHA
        and shapiro_p >= INFERENCE_ALPHA
    )

    if assumptions_ok:
        result = f_oneway(*arrays)
        test_used = "anova"
    else:
        result = kruskal(*arrays)
        test_used = "kruskal"

    # Datos degenerados (p.ej. columnas constantes) producen NaN/inf
    # en scipy; el test no es interpretable, se descarta.
    if not np.isfinite(result.statistic) or not np.isfinite(result.pvalue):
        return None

    (
        top_group,
        top_mean,
        rest_mean,
        mean_difference,
        cohen_d,
        ci_low,
        ci_high,
    ) = _top_vs_rest(grouped)

    return CategoricalNumericInference(
        column_categorical=column_categorical,
        column_numeric=column_numeric,
        n_observations=len(data),
        n_groups=len(grouped),
        test_used=test_used,
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        eta_squared=eta_squared,
        cohen_d=cohen_d,
        mean_difference=mean_difference,
        ci_low=ci_low,
        ci_high=ci_high,
        top_group=top_group,
        top_mean=top_mean,
        rest_mean=rest_mean,
        levene_p=levene_p,
        shapiro_p=shapiro_p,
    )


def run_inference(
    df: pd.DataFrame,
    relationships: list[CategoricalNumericRelationship],
) -> list[CategoricalNumericInference]:
    """
    Corre la inferencia sobre los pares cat-num (los de mayor
    separación primero, tope INFERENCE_MAX_TESTS) y aplica la
    corrección Benjamini-Hochberg sobre esa familia.
    """
    ranked = sorted(
        relationships,
        key=lambda rel: rel.group_separation,
        reverse=True,
    )[:INFERENCE_MAX_TESTS]

    results: list[CategoricalNumericInference] = []

    for relationship in ranked:
        inference = infer_categorical_numeric(df, relationship)

        if inference is not None:
            results.append(inference)

    if INFERENCE_BH_ENABLED and results:
        adjusted = benjamini_hochberg([result.p_value for result in results])

        for result, adjusted_p in zip(results, adjusted):
            result.p_value_bh = adjusted_p

    return results


def _effect_label(eta_squared: float, cohen_d: float | None) -> str:
    if eta_squared >= 0.14 or (cohen_d is not None and abs(cohen_d) >= 0.8):
        return "large"
    if eta_squared >= 0.06 or (cohen_d is not None and abs(cohen_d) >= 0.5):
        return "medium"
    return "small"


def generate_inference_insights(
    inferences: list[CategoricalNumericInference],
) -> list[Insight]:
    """
    Emite insights solo para los tests significativos, calificando
    el tamaño de efecto y separando significancia de efecto práctico.
    """
    insights: list[Insight] = []

    for inference in inferences:
        if not inference.significant:
            continue

        p_display = (
            inference.p_value_bh
            if inference.p_value_bh is not None
            else inference.p_value
        )

        caveats: list[str] = [f"test: {inference.test_used}"]

        if inference.small_sample:
            caveats.append("small sample")

        if inference.test_used == "anova":
            if inference.levene_p is not None:
                caveats.append(f"levene p={inference.levene_p:.3g}")
            if inference.shapiro_p is not None:
                caveats.append(f"shapiro p={inference.shapiro_p:.3g}")

        columns = [inference.column_categorical, inference.column_numeric]
        chart_id = f"catnum:{inference.column_categorical}:{inference.column_numeric}"

        if inference.negligible_effect:
            title = (
                f"{inference.column_numeric} differs across "
                f"{inference.column_categorical} groups but the effect "
                f"is negligible"
            )
            message = (
                f"The difference in {inference.column_numeric} across "
                f"{inference.column_categorical} groups is statistically "
                f"significant ({inference.test_used}, p={inference.p_value:.3g}"
                f"{f', BH q={p_display:.3g}' if inference.p_value_bh is not None else ''}) "
                f"but the effect size is negligible (η²={inference.eta_squared:.3f}): "
                f"the groups separate statistically, not practically."
            )
            score = 5.0
        else:
            label = _effect_label(inference.eta_squared, inference.cohen_d)
            ci_text = (
                f", 95% CI [{inference.ci_low:.2f}, {inference.ci_high:.2f}]"
                if inference.ci_low is not None and inference.ci_high is not None
                else ""
            )
            d_text = (
                f", Cohen's d={inference.cohen_d:.2f}"
                if inference.cohen_d is not None
                else ""
            )
            title = (
                f"{inference.column_numeric} differs across "
                f"{inference.column_categorical} groups "
                f"({label} effect)"
            )
            message = (
                f"The difference in {inference.column_numeric} across "
                f"{inference.column_categorical} groups is statistically "
                f"significant ({inference.test_used}, p={inference.p_value:.3g}"
                f"{f', BH q={p_display:.3g}' if inference.p_value_bh is not None else ''}) "
                f"with a {label} effect (η²={inference.eta_squared:.3f}{d_text}): "
                f"'{inference.top_group}' averages {inference.top_mean:.2f} vs "
                f"{inference.rest_mean:.2f} for the rest "
                f"(diff {inference.mean_difference:+.2f}{ci_text})."
            )
            score = inference.eta_squared * 100.0 + (
                abs(inference.cohen_d) if inference.cohen_d is not None else 0.0
            )

        evidence: dict[str, float | int | str] = {
            "test": inference.test_used,
            "p_value": inference.p_value,
            "eta_squared": round(inference.eta_squared, 4),
            "top_group": inference.top_group,
            "n_observations": inference.n_observations,
            "n_groups": inference.n_groups,
        }

        if inference.p_value_bh is not None:
            evidence["p_value_bh"] = inference.p_value_bh

        if inference.cohen_d is not None:
            evidence["cohen_d"] = round(inference.cohen_d, 4)

        insights.append(
            Insight(
                title=title,
                message=message,
                severity="info",
                score=score,
                category="inference",
                columns=columns,
                chart_id=chart_id,
                evidence=evidence,
                caveats=caveats,
            )
        )

    return insights
