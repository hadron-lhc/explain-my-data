"""
distributions.py

Describe la forma de las distribuciones univariadas.

- Histograma (bins Freedman-Diaconis con fallback Sturges)
- Asimetría y curtosis
- Concentración del tramo superior
- Modalidad por picos del histograma

No modifica el dataset ni genera conclusiones en lenguaje natural.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import COUNT_MAX_UNIQUE
from src.utils_stats import finite_numeric

MAX_BINS = 50
MIN_OBSERVATIONS = 10


@dataclass
class NumericDistribution:
    column_name: str
    n_observations: int

    # Histograma
    bin_edges: list[float] = field(default_factory=list)
    bin_counts: list[int] = field(default_factory=list)

    # Forma
    skewness: float = 0.0
    kurtosis: float = 0.0
    modality: str = "unimodal"  # unimodal | multimodal
    n_modes: int = 1

    # Porcentaje del total concentrado en el 5% superior.
    # Solo se calcula para valores no negativos con total positivo.
    top_5pct_share: float | None = None

    # Cardinalidad de los valores observados. Se captura en el cálculo
    # para que los generadores de insights puedan clasificar la columna
    # (conteo / discreto de baja cardinalidad / continua) sin volver
    # al DataFrame crudo.
    n_unique: int = 0

    # Cinco números resumen. Permiten contextualizar asimetría y
    # extremos en el insight sin volver al DataFrame crudo.
    minimum: float | None = None
    q1: float | None = None
    median: float | None = None
    q3: float | None = None
    maximum: float | None = None

    # La columna parece una variable de conteo: entera, no negativa y
    # con cardinalidad acotada. Solo se emite si el total de
    # observaciones supera el mínimo del histograma.
    looks_like_count: bool = False

    @property
    def is_low_cardinality_discrete(self) -> bool:
        """Numérica discreta con pocos valores posibles (p.ej. pclass)."""
        return self.n_unique <= COUNT_MAX_UNIQUE and self.n_unique > 0


def _n_bins_freedman_diaconis(values: pd.Series) -> int:
    n = len(values)
    iqr = float(values.quantile(0.75) - values.quantile(0.25))

    if iqr > 0:
        bin_width = 2 * iqr / np.cbrt(n)

        if bin_width > 0:
            span = float(values.max() - values.min())
            n_bins = int(np.ceil(span / bin_width)) if span > 0 else 1
            return min(max(n_bins, 1), MAX_BINS)

    # Fallback Sturges cuando el IQR es cero.
    return min(max(int(np.ceil(np.log2(n) + 1)), 1), MAX_BINS)


def _detect_modality(counts: np.ndarray) -> tuple[str, int]:
    """
    Cuenta modos como picos locales (incluyendo los bordes) que además
    estén separados por un valle claro (< 50% del pico más bajo).

    Evita clasificar como multimodal el ruido de picos menores de un
    histograma unimodal.
    """
    if len(counts) < 3:
        return "unimodal", 1

    max_count = float(counts.max())

    if max_count <= 0:
        return "unimodal", 1

    peaks: list[int] = []

    for index in range(len(counts)):
        left = float(counts[index - 1]) if index > 0 else -1.0
        right = float(counts[index + 1]) if index < len(counts) - 1 else -1.0

        if counts[index] > left and counts[index] >= right:
            peaks.append(index)

    significant_peaks = [peak for peak in peaks if counts[peak] >= 0.3 * max_count]

    if len(significant_peaks) < 2:
        return "unimodal", 1

    n_modes = 1
    last_peak = significant_peaks[0]

    for peak in significant_peaks[1:]:
        valley = float(counts[last_peak : peak + 1].min())
        smaller_peak = min(float(counts[last_peak]), float(counts[peak]))

        if valley <= 0.5 * smaller_peak:
            n_modes += 1
            last_peak = peak

    if n_modes >= 2:
        return "multimodal", n_modes

    return "unimodal", 1


def calculate_numeric_distribution(
    series: pd.Series,
    column_name: str,
) -> NumericDistribution | None:
    """Calcula histograma y medidas de forma para una columna numérica."""
    values = finite_numeric(series).dropna()

    if len(values) < MIN_OBSERVATIONS:
        return None

    n = len(values)

    n_bins = _n_bins_freedman_diaconis(values)
    counts, edges = np.histogram(values, bins=n_bins)

    modality, n_modes = _detect_modality(counts.astype(float))

    top_5pct_share: float | None = None
    total = float(values.sum())

    if (values >= 0).all() and total > 0:
        top_n = max(int(np.ceil(0.05 * n)), 1)
        top_5pct_share = round(float(values.nlargest(top_n).sum() / total * 100), 2)

    n_unique = int(values.nunique())

    # Conteo: entera, no negativa y de cardinalidad acotada. Un valor
    # negativo o fraccionario descarta la interpretación de conteo; una
    # cardinalidad alta descarta la acotación (una continua con valores
    # enteros por azar no es un conteo).
    looks_like_count = bool(
        n_unique > 0
        and n_unique <= COUNT_MAX_UNIQUE
        and bool((values >= 0).all())
        and bool((values == values.round()).all())
    )

    return NumericDistribution(
        column_name=column_name,
        n_observations=n,
        bin_edges=[float(edge) for edge in edges],
        bin_counts=[int(count) for count in counts],
        skewness=float(values.skew()),
        kurtosis=float(values.kurt()),
        modality=modality,
        n_modes=n_modes,
        top_5pct_share=top_5pct_share,
        n_unique=n_unique,
        looks_like_count=looks_like_count,
        minimum=float(values.min()),
        q1=float(values.quantile(0.25)),
        median=float(values.quantile(0.5)),
        q3=float(values.quantile(0.75)),
        maximum=float(values.max()),
    )


def calculate_numeric_distributions(
    df: pd.DataFrame,
    numeric_columns: list[str],
) -> list[NumericDistribution]:
    """Calcula distribuciones para las columnas numéricas seleccionadas."""
    distributions: list[NumericDistribution] = []

    for column in dict.fromkeys(numeric_columns):
        if column not in df.columns:
            continue

        distribution = calculate_numeric_distribution(df[column], column)

        if distribution is not None:
            distributions.append(distribution)

    return distributions
