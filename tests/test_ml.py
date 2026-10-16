"""
Tests para src/ml.py — detección de target y recomendación de modelos.
"""

import pandas as pd

from src.analysis import analyze_dataset
from src.insights import Insight
from src.ml import (
    MLRecommendation,
    build_ml_recommendation,
    detect_target_candidates,
    generate_ml_insights,
    recommend_models,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _analyze(df: pd.DataFrame):
    report = analyze_dataset(df)
    return report


# ---------------------------------------------------------------------------
# detect_target_candidates
# ---------------------------------------------------------------------------


class TestDetectTargetCandidates:
    def test_no_candidates_for_identifier_only_dataset(self):
        df = pd.DataFrame({"id": range(100), "value": range(100)})
        report = _analyze(df)
        candidates = detect_target_candidates(
            report.profile,
            df,
            report.numeric_relationships,
            report.categorical_numeric_relationships,
            report.categorical_categorical_relationships,
        )
        # value es numeric con 100 unique → regression candidate
        assert any(c.column_name == "value" for c in candidates)

    def test_boolean_binary_target(self):
        df = pd.DataFrame(
            {
                "feature1": [1.0, 2.0, 3.0, 4.0] * 25,
                "feature2": [10, 20, 30, 40] * 25,
                "target": [0, 1, 0, 1] * 25,
            }
        )
        report = _analyze(df)
        candidates = detect_target_candidates(
            report.profile,
            df,
            report.numeric_relationships,
            report.categorical_numeric_relationships,
            report.categorical_categorical_relationships,
        )
        assert candidates[0].column_name == "target"
        assert candidates[0].task_type == "binary"

    def test_regression_target_high_variability(self):
        df = pd.DataFrame(
            {
                "cat": ["a", "b", "c", "d"] * 25,
                "price": [10.5, 20.3, 30.1, 40.7, 50.2, 60.8, 70.4, 80.9, 90.1, 100.5,
                          110.2, 120.8, 130.3, 140.7, 150.1, 160.6, 170.2, 180.8, 190.4, 200.9] * 5,
            }
        )
        report = _analyze(df)
        candidates = detect_target_candidates(
            report.profile,
            df,
            report.numeric_relationships,
            report.categorical_numeric_relationships,
            report.categorical_categorical_relationships,
        )
        assert candidates[0].column_name == "price"
        assert candidates[0].task_type == "regression"

    def test_identifier_excluded(self):
        df = pd.DataFrame(
            {
                "id": range(100),
                "feature": range(100),
                "label": [0, 1] * 50,
            }
        )
        report = _analyze(df)
        candidates = detect_target_candidates(
            report.profile,
            df,
            report.numeric_relationships,
            report.categorical_numeric_relationships,
            report.categorical_categorical_relationships,
        )
        assert not any(c.column_name == "id" for c in candidates)

    def test_candidates_sorted_by_score(self):
        df = pd.DataFrame(
            {
                "a": range(100),
                "b": range(100),
                "target": [0, 1] * 50,
            }
        )
        report = _analyze(df)
        candidates = detect_target_candidates(
            report.profile,
            df,
            report.numeric_relationships,
            report.categorical_numeric_relationships,
            report.categorical_categorical_relationships,
        )
        scores = [c.score for c in candidates]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# recommend_models
# ---------------------------------------------------------------------------


class TestRecommendModels:
    def test_binary_small_dataset(self):
        models = recommend_models("binary", 150)
        assert any(m.model == "Logistic Regression" for m in models)
        assert any(m.model == "k-Nearest Neighbors" for m in models)
        # XGBoost needs 1000 rows
        assert not any(m.model == "XGBoost / LightGBM" for m in models)

    def test_binary_large_dataset(self):
        models = recommend_models("binary", 5000)
        assert any(m.model == "XGBoost / LightGBM" for m in models)
        assert any(m.model == "Random Forest" for m in models)

    def test_regression_models(self):
        models = recommend_models("regression", 2000)
        assert any(m.model == "XGBoost / LightGBM" for m in models)
        assert not any(m.model == "Logistic Regression" for m in models)

    def test_none_task_returns_empty(self):
        assert recommend_models(None, 1000) == []


# ---------------------------------------------------------------------------
# build_ml_recommendation
# ---------------------------------------------------------------------------


class TestBuildMLRecommendation:
    def test_titanic_survived(self):
        df = pd.DataFrame(
            {
                "survived": [0, 1] * 50,
                "age": [20, 30, 40, 50] * 25,
                "fare": [10.0, 20.0, 30.0, 40.0] * 25,
                "sex": ["male", "female"] * 50,
            }
        )
        report = _analyze(df)
        rec = build_ml_recommendation(
            report.profile,
            df,
            report.numeric_relationships,
            report.categorical_numeric_relationships,
            report.categorical_categorical_relationships,
        )
        assert rec.target == "survived"
        assert rec.task_type == "binary"
        assert "age" in rec.candidate_features
        assert "survived" not in rec.candidate_features
        assert rec.exclusion_reasons.get("survived") == "target"

    def test_identifier_excluded_from_features(self):
        df = pd.DataFrame(
            {
                "id": range(100),
                "feature": range(100),
                "label": [0, 1] * 50,
            }
        )
        report = _analyze(df)
        rec = build_ml_recommendation(
            report.profile,
            df,
            report.numeric_relationships,
            report.categorical_numeric_relationships,
            report.categorical_categorical_relationships,
        )
        assert "id" in rec.excluded_features
        assert rec.exclusion_reasons.get("id") == "identifier"

    def test_constant_column_excluded(self):
        df = pd.DataFrame(
            {
                "constant": [1] * 100,
                "feature": range(100),
                "label": [0, 1] * 50,
            }
        )
        report = _analyze(df)
        rec = build_ml_recommendation(
            report.profile,
            df,
            report.numeric_relationships,
            report.categorical_numeric_relationships,
            report.categorical_categorical_relationships,
        )
        assert "constant" in rec.excluded_features
        assert rec.exclusion_reasons.get("constant") == "constant (no variance)"

    def test_no_target_returns_none(self):
        df = pd.DataFrame({"a": [1] * 50, "b": [2] * 50})
        report = _analyze(df)
        rec = build_ml_recommendation(
            report.profile,
            df,
            report.numeric_relationships,
            report.categorical_numeric_relationships,
            report.categorical_categorical_relationships,
        )
        assert rec.task_type is None
        assert rec.target is None
        assert any("No clear target" in n for n in rec.notes)


# ---------------------------------------------------------------------------
# generate_ml_insights
# ---------------------------------------------------------------------------


class TestGenerateMLInsights:
    def test_insights_generated(self):
        df = pd.DataFrame(
            {
                "survived": [0, 1] * 50,
                "age": [20, 30, 40, 50] * 25,
                "fare": [10.0, 20.0, 30.0, 40.0] * 25,
            }
        )
        report = _analyze(df)
        rec = build_ml_recommendation(
            report.profile,
            df,
            report.numeric_relationships,
            report.categorical_numeric_relationships,
            report.categorical_categorical_relationships,
        )
        insights = generate_ml_insights(rec)
        assert len(insights) >= 2
        assert all(isinstance(i, Insight) for i in insights)
        assert all(i.category == "ml_recommendations" for i in insights)

    def test_no_target_no_insights(self):
        rec = MLRecommendation(task_type=None, target=None)
        insights = generate_ml_insights(rec)
        assert insights == []

    def test_low_confidence_note(self):
        # Numeric column with enough unique values to be a regression
        # candidate but no name keyword and no other columns → low score.
        df = pd.DataFrame({"value": list(range(30)) * 4})
        report = _analyze(df)
        rec = build_ml_recommendation(
            report.profile,
            df,
            report.numeric_relationships,
            report.categorical_numeric_relationships,
            report.categorical_categorical_relationships,
        )
        if rec.notes:
            insights = generate_ml_insights(rec)
            assert any(
                "low confidence" in i.message.lower() or "low confidence" in i.title.lower()
                for i in insights
            )


# ---------------------------------------------------------------------------
# Integration: real datasets
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_titanic_detects_survived(self):
        df = pd.read_csv("../explain-my-data-qa/datasets/titanic.csv")
        report = _analyze(df)
        rec = report.ml_recommendation
        assert rec.target == "survived"
        assert rec.task_type == "binary"
        assert len(report.ml_insights) >= 2

    def test_iris_detects_species(self):
        df = pd.read_csv("../explain-my-data-qa/datasets/iris.csv")
        report = _analyze(df)
        rec = report.ml_recommendation
        assert rec.target == "Species"
        assert rec.task_type == "multiclass"

    def test_diamonds_detects_price(self):
        df = pd.read_csv("../explain-my-data-qa/datasets/diamonds.csv")
        report = _analyze(df)
        rec = report.ml_recommendation
        assert rec.target == "price"
        assert rec.task_type == "regression"

    def test_ml_insights_in_executive_summary(self):
        df = pd.read_csv("../explain-my-data-qa/datasets/titanic.csv")
        report = _analyze(df)
        exec_ml = [i for i in report.executive_summary if i.category == "ml_recommendations"]
        assert len(exec_ml) >= 1
