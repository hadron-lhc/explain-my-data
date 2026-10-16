"""Tests de CleaningRecommendation + recommend_cleaning_actions (F5)."""

import numpy as np
import pandas as pd

from src.cleaning import recommend_cleaning_actions
from src.profiling import profile_dataset
from src.quality import quality_dataset


def _report(df: pd.DataFrame):
    profile = profile_dataset(df)
    quality = quality_dataset(df, profile)
    return profile, quality


class TestRecommendCleaningActions:
    def test_no_issues_no_recommendations(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": ["x", "y", "x"]})
        profile, quality = _report(df)

        assert recommend_cleaning_actions(quality, profile) == []

    def test_duplicates_recommendation(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 2.0], "b": ["x", "y", "y"]})
        profile, quality = _report(df)
        recs = recommend_cleaning_actions(quality, profile)

        dupes = [r for r in recs if r.action == "remove_duplicates"]
        assert len(dupes) == 1
        assert dupes[0].column_name is None
        assert dupes[0].affected_rows == 1
        assert "duplicate" in dupes[0].why

    def test_invalid_values_recommendation(self):
        df = pd.DataFrame({"a": [1.0, np.inf, 3.0]})
        profile, quality = _report(df)
        recs = recommend_cleaning_actions(quality, profile)

        invalid = [r for r in recs if r.action == "replace_invalid_values"]
        assert len(invalid) == 1
        assert invalid[0].column_name == "a"
        assert invalid[0].affected_rows == 1

    def test_missing_low_pct_recommends_impute(self):
        df = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]})
        profile, quality = _report(df)
        recs = recommend_cleaning_actions(quality, profile)

        impute = [r for r in recs if r.action == "impute_missing_median"]
        assert len(impute) == 1
        assert impute[0].column_name == "a"
        assert impute[0].affected_rows == 1

    def test_missing_high_pct_no_impute(self):
        df = pd.DataFrame({"a": [np.nan] * 6 + [1.0, 2.0, 3.0, 4.0]})
        profile, quality = _report(df)
        recs = recommend_cleaning_actions(quality, profile)

        impute = [r for r in recs if r.action == "impute_missing_median"]
        assert impute == []

    def test_max_recommendations_cap(self):
        rows = []
        for i in range(30):
            rows.append({"a": float(i), "b": float(i), "c": float(i)})
        df = pd.DataFrame(rows)
        df = pd.concat([df, df.iloc[:10]], ignore_index=True)

        df.loc[df.index[:5], "a"] = np.nan
        df.loc[df.index[5:8], "b"] = np.inf

        profile, quality = _report(df)
        recs = recommend_cleaning_actions(quality, profile, max_recommendations=3)

        assert len(recs) <= 3

    def test_sorted_by_affected_rows_desc(self):
        df = pd.DataFrame({"a": [1.0] * 5 + [np.inf] * 3 + [np.nan] * 2})
        df = pd.concat([df, df.iloc[:4]], ignore_index=True)

        profile, quality = _report(df)
        recs = recommend_cleaning_actions(quality, profile)

        affected = [r.affected_rows for r in recs]
        assert affected == sorted(affected, reverse=True)

    def test_never_modifies_dataframe(self):
        df = pd.DataFrame({"a": [1.0, np.inf, np.nan, 4.0]})
        original = df.copy(deep=True)
        profile, quality = _report(df)

        recommend_cleaning_actions(quality, profile)

        pd.testing.assert_frame_equal(df, original)
