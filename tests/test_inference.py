"""Tests del módulo de inferencia estadística (F4)."""

import numpy as np
import pandas as pd

from src.inference import (
    benjamini_hochberg,
    generate_inference_insights,
    infer_categorical_numeric,
    run_inference,
)
from src.relationships import CategoricalNumericRelationship


def _strong_effect_frame(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    region = rng.choice(["A", "B", "C"], n)
    effect = {"A": 0.0, "B": 10.0, "C": 20.0}
    values = np.array([effect[r] for r in region]) + rng.normal(0, 1, n)

    return pd.DataFrame({"region": region, "sales": values})


def _null_frame(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    region = rng.choice(["A", "B", "C"], n)
    values = rng.normal(100, 1, n)

    return pd.DataFrame({"region": region, "sales": values})


def _relationship(categorical: str, numeric: str) -> CategoricalNumericRelationship:
    return CategoricalNumericRelationship(
        column_categorical=categorical,
        column_numeric=numeric,
        group_means={},
        group_medians={},
        group_counts={},
        n_observations=0,
        group_separation=1.0,
        n_groups=3,
        group_outliers={},
    )


def _low_coverage_relationship(
    categorical: str,
    numeric: str,
    coverage_pct: float = 22.8,
) -> CategoricalNumericRelationship:
    relationship = _relationship(categorical, numeric)
    relationship.coverage_pct = coverage_pct
    return relationship


class TestBenjaminiHochberg:
    def test_empty(self):
        assert benjamini_hochberg([]) == []

    def test_known_values(self):
        pvalues = [0.01, 0.04, 0.03, 0.5]
        adjusted = benjamini_hochberg(pvalues)

        assert len(adjusted) == 4

        for original, adj in zip(pvalues, adjusted):
            assert adj >= original

        assert all(adj <= 1.0 for adj in adjusted)

    def test_monotonicity(self):
        pvalues = [0.001, 0.01, 0.02, 0.03, 0.5]
        adjusted = benjamini_hochberg(pvalues)

        for i in range(1, len(adjusted)):
            assert adjusted[i] >= adjusted[i - 1]


class TestInferCategoricalNumeric:
    def test_detects_strong_effect(self):
        df = _strong_effect_frame()
        inference = infer_categorical_numeric(df, _relationship("region", "sales"))

        assert inference is not None
        assert inference.test_used in ("anova", "kruskal")
        assert inference.significant
        assert inference.eta_squared > 0.5
        assert inference.top_group == "C"
        assert inference.ci_low is not None and inference.ci_high is not None
        assert inference.ci_low < inference.ci_high

    def test_no_effect_not_significant(self):
        df = _null_frame()
        inference = infer_categorical_numeric(df, _relationship("region", "sales"))

        assert inference is not None
        assert not inference.significant

    def test_too_few_observations_returns_none(self):
        df = _strong_effect_frame(n=10)
        inference = infer_categorical_numeric(df, _relationship("region", "sales"))

        assert inference is None

    def test_missing_column_returns_none(self):
        df = _strong_effect_frame()
        inference = infer_categorical_numeric(df, _relationship("missing", "sales"))

        assert inference is None

    def test_constant_column_returns_none(self):
        df = pd.DataFrame(
            {
                "region": ["A", "B"] * 50,
                "sales": [1.0] * 100,
            }
        )
        inference = infer_categorical_numeric(df, _relationship("region", "sales"))

        assert inference is None

    def test_skips_groups_with_fewer_than_two(self):
        df = _strong_effect_frame()
        df.loc[df.index[:5], "region"] = "D"
        df = df[df["region"] != "D"].reset_index(drop=True)
        df.loc[df.index[0], "region"] = "E"

        inference = infer_categorical_numeric(df, _relationship("region", "sales"))

        assert inference is not None
        assert inference.n_groups == 3


class TestRunInference:
    def test_applies_bh_correction(self):
        df = _strong_effect_frame()
        relationships = [
            _relationship("region", "sales"),
            _relationship("region", "sales"),
        ]
        relationships[0].column_numeric = "sales"

        inference_list = run_inference(df, relationships[:1])

        assert len(inference_list) == 1
        assert inference_list[0].p_value_bh is not None
        assert inference_list[0].p_value_bh >= inference_list[0].p_value

    def test_caps_at_max_tests(self):
        df = _strong_effect_frame()
        relationships = [_relationship("region", "sales") for _ in range(20)]

        inference_list = run_inference(df, relationships)

        assert len(inference_list) <= 10

    def test_low_coverage_is_degraded_not_suppressed(self):
        # Un coverage bajo no suprime la inferencia: la califica. La
        # supresión vive en el filtro de relaciones (grupos < n), no
        # aquí, para que el score y el caveat lleguen al usuario.
        df = _strong_effect_frame()
        relationships = [_relationship("region", "sales")]
        relationships[0].coverage_pct = 10.0

        results = run_inference(df, relationships)

        assert len(results) == 1
        assert results[0].low_coverage


class TestGenerateInferenceInsights:
    def test_significant_generates_insight(self):
        df = _strong_effect_frame()
        inference = infer_categorical_numeric(df, _relationship("region", "sales"))
        insights = generate_inference_insights([inference])

        assert len(insights) == 1
        insight = insights[0]
        assert insight.category == "inference"
        assert "significant" in insight.message
        assert "region" in insight.columns
        assert "sales" in insight.columns
        assert any("test: " in caveat for caveat in insight.caveats)

    def test_not_significant_generates_nothing(self):
        df = _null_frame()
        inference = infer_categorical_numeric(df, _relationship("region", "sales"))
        insights = generate_inference_insights([inference])

        assert insights == []

    def test_negligible_effect_calibrated(self):
        rng = np.random.default_rng(7)
        n = 200
        region = rng.choice(["A", "B"], n)
        # efecto tiny pero con n grande → significativo con η² < 0.01
        values = np.where(region == "B", 0.01, 0.0) + rng.normal(0, 1, n)
        df = pd.DataFrame({"region": region, "sales": values})

        inference = infer_categorical_numeric(df, _relationship("region", "sales"))
        assert inference is not None

        if inference.significant and inference.negligible_effect:
            insights = generate_inference_insights([inference])
            assert len(insights) == 1
            assert "negligible" in insights[0].title

    def test_small_sample_caveat(self):
        df = _strong_effect_frame(n=30)
        inference = infer_categorical_numeric(df, _relationship("region", "sales"))

        assert inference is not None
        assert inference.small_sample

        if inference.significant:
            insights = generate_inference_insights([inference])
            assert any("small sample" in c for c in insights[0].caveats)

    def test_low_coverage_degrades_and_caveats(self):
        df = _strong_effect_frame()
        full = _strong_effect_frame()

        inference = infer_categorical_numeric(df, _low_coverage_relationship("region", "sales"))
        assert inference is not None
        assert inference.low_coverage

        full_inference = infer_categorical_numeric(full, _relationship("region", "sales"))

        if inference.significant and full_inference.significant:
            insights = generate_inference_insights([inference])
            full_insights = generate_inference_insights([full_inference])

            assert any("of rows" in c for c in insights[0].caveats)
            assert insights[0].score < full_insights[0].score
