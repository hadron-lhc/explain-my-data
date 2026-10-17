import numpy as np
import pandas as pd
import pytest

from src.distributions import (
    calculate_numeric_distribution,
    calculate_numeric_distributions,
)
from src.insights import generate_distribution_insights


def test_calculates_histogram_and_shape():
    rng = np.random.default_rng(42)
    values = rng.normal(loc=50, scale=5, size=500)

    dist = calculate_numeric_distribution(pd.Series(values), "score")

    assert dist is not None
    assert dist.column_name == "score"
    assert dist.n_observations == 500
    assert len(dist.bin_edges) == len(dist.bin_counts) + 1
    assert sum(dist.bin_counts) == 500
    assert dist.modality == "unimodal"
    assert abs(dist.skewness) < 1.0


def test_detects_right_skew_and_concentration():
    values = [float(i) for i in range(1, 201)]
    values[-1] = 50000.0

    dist = calculate_numeric_distribution(pd.Series(values), "revenue")

    assert dist is not None
    assert dist.skewness > 1.0
    assert dist.top_5pct_share is not None
    assert dist.top_5pct_share > 40.0


def test_detects_multimodal_distribution():
    rng = np.random.default_rng(3)
    cluster_a = rng.normal(loc=0, scale=1, size=300)
    cluster_b = rng.normal(loc=20, scale=1, size=300)

    dist = calculate_numeric_distribution(
        pd.Series(np.concatenate([cluster_a, cluster_b])),
        "mix",
    )

    assert dist is not None
    assert dist.modality == "multimodal"
    assert dist.n_modes >= 2


def test_handles_constant_column():
    dist = calculate_numeric_distribution(pd.Series([5.0] * 50), "fixed")

    assert dist is not None
    assert dist.n_observations == 50
    assert sum(dist.bin_counts) == 50


def test_returns_none_for_few_observations():
    assert calculate_numeric_distribution(pd.Series([1.0, 2.0]), "x") is None


def test_ignores_missing_and_infinite_values():
    values = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0]
    series = pd.Series(values + [np.nan, np.inf])

    dist = calculate_numeric_distribution(series, "x")

    assert dist is not None
    assert dist.n_observations == 11


def test_does_not_modify_original_series():
    series = pd.Series([float(i) for i in range(20)])
    original = series.copy()

    calculate_numeric_distribution(series, "x")

    pd.testing.assert_series_equal(series, original)


def test_calculate_numeric_distributions_skips_missing_columns():
    df = pd.DataFrame({"a": [float(i) for i in range(20)]})

    distributions = calculate_numeric_distributions(df, ["a", "missing"])

    assert len(distributions) == 1
    assert distributions[0].column_name == "a"


def test_negative_values_have_no_top_share():
    values = [-float(i) for i in range(1, 51)]

    dist = calculate_numeric_distribution(pd.Series(values), "loss")

    assert dist is not None
    assert dist.top_5pct_share is None


# ---------------------------------------------------------------------------
# Distribution insights
# ---------------------------------------------------------------------------


def test_skewness_insight():
    values = [float(i) for i in range(1, 101)]
    values[-1] = 5000.0

    dist = calculate_numeric_distribution(pd.Series(values), "revenue")
    insights = generate_distribution_insights([dist])

    assert any("skew" in insight.title.lower() for insight in insights)


def test_concentration_insight():
    values = [1.0] * 195 + [1000.0] * 5

    dist = calculate_numeric_distribution(pd.Series(values), "revenue")
    insights = generate_distribution_insights([dist])

    assert any("dominate" in insight.title for insight in insights)


def test_no_insights_for_balanced_distribution():
    rng = np.random.default_rng(11)
    dist = calculate_numeric_distribution(
        pd.Series(rng.normal(0, 1, 400)),
        "score",
    )

    insights = generate_distribution_insights([dist])

    assert all("skew" not in insight.title.lower() for insight in insights)
    assert not any("dominate" in insight.title for insight in insights)


@pytest.mark.parametrize("modality", ["unimodal"])
def test_modality_insight_only_for_multimodal(modality):
    rng = np.random.default_rng(5)
    dist = calculate_numeric_distribution(
        pd.Series(rng.normal(0, 1, 300)),
        "score",
    )

    insights = generate_distribution_insights([dist])

    assert not any("subpopulations" in insight.title for insight in insights)


# ---------------------------------------------------------------------------
# P2 — Trivialidades: conteos y discretos de baja cardinalidad
# ---------------------------------------------------------------------------


def test_flags_count_columns():
    # sibsp-style: entera, no negativa, cardinalidad acotada.
    values = [0.0, 1.0, 1.0, 2.0, 1.0, 3.0] * 40
    dist = calculate_numeric_distribution(pd.Series(values), "sibsp")

    assert dist is not None
    assert dist.looks_like_count
    assert dist.is_low_cardinality_discrete


def test_does_not_flag_fractional_or_negative_as_count():
    fractional = calculate_numeric_distribution(
        pd.Series([1.5, 2.5, 3.5] * 20), "fare"
    )
    assert fractional is not None
    assert not fractional.looks_like_count

    negative = calculate_numeric_distribution(
        pd.Series([-1.0, 0.0, 1.0] * 20), "delta"
    )
    assert negative is not None
    assert not negative.looks_like_count


def test_does_not_flag_high_cardinality_integer_as_count():
    # Una continua que resulta entera por azar (fnlwgt, 28k valores)
    # no es un conteo.
    values = [float(i) for i in range(2000, 2400)]
    dist = calculate_numeric_distribution(pd.Series(values), "fnlwgt")

    assert dist is not None
    assert not dist.looks_like_count


def test_count_skew_insight_suppressed():
    # Conteo sesgado a la derecha (skew ~1.5 < COUNT_SKIP_SKEW): trivial.
    values = [0.0] * 80 + [1.0] * 12 + [2.0] * 5 + [3.0] * 2 + [4.0] * 1
    dist = calculate_numeric_distribution(pd.Series(values), "sibsp")
    insights = generate_distribution_insights([dist])

    assert not any("skew" in insight.title.lower() for insight in insights)


def test_count_extreme_skew_still_reported():
    # Mismo conteo pero con una cola larguísima: ya no es la forma
    # esperada del conteo, así que se emite.
    values = [0.0] * 95 + [1.0] * 3 + [2000.0] * 2
    dist = calculate_numeric_distribution(pd.Series(values), "visitors")
    insights = generate_distribution_insights([dist])

    assert any("skew" in insight.title.lower() for insight in insights)


def test_count_concentration_suppressed_below_threshold():
    # Conteo con ~90% de observaciones en cero: el 5% superior se lo
    # lleva casi todo el total, pero por debajo del umbral de señal
    # (COUNT_SKIP_TOP_PCT) sigue siendo trivial para un conteo.
    values = [0.0] * 180 + [1.0] * 10 + [2.0] * 6 + [3.0] * 4
    dist = calculate_numeric_distribution(pd.Series(values), "sibsp")
    insights = generate_distribution_insights([dist])

    assert dist.top_5pct_share is not None
    assert dist.top_5pct_share < 80.0
    assert not any("dominate" in insight.title for insight in insights)


def test_count_extreme_concentration_still_reported():
    # Conteo con ~90% del total en el 5% superior: señal real.
    values = [0.0] * 195 + [1000.0] * 5
    dist = calculate_numeric_distribution(pd.Series(values), "revenue")
    insights = generate_distribution_insights([dist])

    assert dist.top_5pct_share is not None
    assert dist.top_5pct_share >= 80.0
    assert any("dominate" in insight.title for insight in insights)


def test_low_cardinality_discrete_modality_suppressed():
    # pclass: 3 valores posibles, el histograma marca 3 modas. Es
    # ruido, no subpoblaciones mezcladas.
    values = [1.0] * 216 + [2.0] * 184 + [3.0] * 491
    dist = calculate_numeric_distribution(pd.Series(values), "pclass")

    assert dist is not None
    assert dist.n_unique == 3
    assert dist.is_low_cardinality_discrete

    insights = generate_distribution_insights([dist])

    assert not any("subpopulations" in insight.title for insight in insights)


def test_continuous_modality_still_reported():
    # age: 88 valores, no discreta de baja cardinalidad → se emite.
    rng = np.random.default_rng(9)
    cluster_a = rng.normal(0, 1, 250)
    cluster_b = rng.normal(20, 1, 250)
    dist = calculate_numeric_distribution(
        pd.Series(np.concatenate([cluster_a, cluster_b])),
        "age",
    )

    assert dist is not None
    assert not dist.is_low_cardinality_discrete

    insights = generate_distribution_insights([dist])

    assert any("subpopulations" in insight.title for insight in insights)
