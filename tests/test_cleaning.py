import numpy as np
import pandas as pd
import pytest

from src.cleaning import CleaningConfig, clean_dataset
from src.profiling import profile_dataset
from src.quality import quality_dataset


def build_profile_and_quality(df: pd.DataFrame):
    profile = profile_dataset(df)
    quality = quality_dataset(df, profile)
    return profile, quality


def test_removes_duplicate_rows():
    df = pd.DataFrame(
        {
            "price": [100, 200, 200, 300],
            "category": ["a", "b", "b", "c"],
        }
    )
    profile, quality = build_profile_and_quality(df)
    config = CleaningConfig(remove_duplicates=True)

    cleaned, report = clean_dataset(df, profile, quality, config)

    assert len(cleaned) == 3
    assert report.original_rows == 4
    assert report.final_rows == 3
    assert report.duplicates_removed == 1


def test_can_keep_duplicate_rows():
    df = pd.DataFrame(
        {
            "price": [100, 200, 200],
            "category": ["a", "b", "b"],
        }
    )
    profile, quality = build_profile_and_quality(df)
    config = CleaningConfig(remove_duplicates=False)

    cleaned, report = clean_dataset(df, profile, quality, config)

    assert len(cleaned) == 3
    assert report.duplicates_removed == 0


@pytest.mark.parametrize(
    ("strategy", "expected_value"),
    [
        ("median", 200.0),
        ("mean", 200.0),
    ],
)
def test_fills_numeric_missing_values(strategy, expected_value):
    df = pd.DataFrame({"price": [100.0, 200.0, np.nan, 300.0]})
    profile, quality = build_profile_and_quality(df)
    config = CleaningConfig(
        remove_duplicates=False,
        missing_numeric_strategy=strategy,
    )

    cleaned, report = clean_dataset(df, profile, quality, config)

    assert cleaned["price"].isna().sum() == 0
    assert cleaned["price"].iloc[2] == expected_value
    assert report.missing_values_filled == 1


def test_can_keep_numeric_missing_values():
    df = pd.DataFrame({"price": [100.0, np.nan, 300.0]})
    profile, quality = build_profile_and_quality(df)
    config = CleaningConfig(
        remove_duplicates=False,
        missing_numeric_strategy="keep",
    )

    cleaned, report = clean_dataset(df, profile, quality, config)

    assert cleaned["price"].isna().sum() == 1
    assert report.missing_values_filled == 0
    assert report.missing_rows_removed == 0


def test_can_remove_rows_with_numeric_missing_values():
    df = pd.DataFrame({"price": [100.0, np.nan, 300.0]})
    profile, quality = build_profile_and_quality(df)
    config = CleaningConfig(
        remove_duplicates=False,
        missing_numeric_strategy="remove",
    )

    cleaned, report = clean_dataset(df, profile, quality, config)

    assert len(cleaned) == 2
    assert cleaned["price"].isna().sum() == 0
    assert report.missing_rows_removed == 1


def test_fills_categorical_missing_values_with_mode():
    df = pd.DataFrame({"category": ["house", "apartment", "house", None]})
    profile, quality = build_profile_and_quality(df)
    config = CleaningConfig(
        remove_duplicates=False,
        missing_categorical_strategy="mode",
    )

    cleaned, report = clean_dataset(df, profile, quality, config)

    assert cleaned["category"].isna().sum() == 0
    assert cleaned["category"].iloc[3] == "house"
    assert report.missing_values_filled == 1


def test_can_keep_categorical_missing_values():
    df = pd.DataFrame({"category": ["house", "apartment", None]})
    profile, quality = build_profile_and_quality(df)
    config = CleaningConfig(
        remove_duplicates=False,
        missing_categorical_strategy="keep",
    )

    cleaned, report = clean_dataset(df, profile, quality, config)

    assert cleaned["category"].isna().sum() == 1
    assert report.missing_values_filled == 0


def test_can_remove_rows_with_categorical_missing_values():
    df = pd.DataFrame({"category": ["house", "apartment", None]})
    profile, quality = build_profile_and_quality(df)
    config = CleaningConfig(
        remove_duplicates=False,
        missing_categorical_strategy="remove",
    )

    cleaned, report = clean_dataset(df, profile, quality, config)

    assert len(cleaned) == 2
    assert cleaned["category"].isna().sum() == 0
    assert report.missing_rows_removed == 1


def test_replaces_infinite_numeric_values_with_missing():
    df = pd.DataFrame({"price": [100.0, np.inf, 300.0]})
    profile, quality = build_profile_and_quality(df)
    config = CleaningConfig(
        remove_duplicates=False,
        missing_numeric_strategy="keep",
    )

    cleaned, report = clean_dataset(df, profile, quality, config)

    assert cleaned["price"].isna().sum() == 1
    assert not np.isinf(cleaned["price"]).any()
    assert report.invalid_values_replaced == 1


def test_does_not_replace_invalid_values_when_disabled():
    df = pd.DataFrame({"price": [100.0, np.inf, 300.0]})
    profile, quality = build_profile_and_quality(df)
    config = CleaningConfig(
        remove_duplicates=False,
        replace_invalid_values=False,
        missing_numeric_strategy="keep",
    )

    cleaned, report = clean_dataset(df, profile, quality, config)

    assert np.isinf(cleaned["price"]).sum() == 1
    assert report.invalid_values_replaced == 0


def test_does_not_convert_type_mismatch_by_default():
    df = pd.DataFrame({"price": ["100", "200", "300"]})
    profile, quality = build_profile_and_quality(df)
    config = CleaningConfig(remove_duplicates=False)

    cleaned, report = clean_dataset(df, profile, quality, config)

    assert cleaned["price"].dtype == df["price"].dtype
    assert report.type_conversions == 0


def test_can_explicitly_convert_numeric_type_mismatch():
    df = pd.DataFrame({"price": ["100", "200", "300"]})
    profile, quality = build_profile_and_quality(df)
    config = CleaningConfig(
        remove_duplicates=False,
        convert_type_mismatches=True,
    )

    cleaned, report = clean_dataset(df, profile, quality, config)

    assert pd.api.types.is_numeric_dtype(cleaned["price"])
    assert report.type_conversions == 1

    conversion_actions = [
        action for action in report.actions if action.action == "convert_type"
    ]
    assert len(conversion_actions) == 1
    assert conversion_actions[0].column_name == "price"
    assert conversion_actions[0].original_dtype is not None
    assert conversion_actions[0].resulting_dtype is not None


def test_does_not_apply_lossy_type_conversion():
    df = pd.DataFrame({"price": ["100", "200", "not_available"]})
    profile, quality = build_profile_and_quality(df)
    config = CleaningConfig(
        remove_duplicates=False,
        convert_type_mismatches=True,
    )

    cleaned, report = clean_dataset(df, profile, quality, config)

    assert cleaned["price"].dtype == df["price"].dtype
    assert report.type_conversions == 0


def test_does_not_modify_original_dataframe():
    df = pd.DataFrame(
        {
            "price": [100.0, np.nan, 300.0],
            "category": ["a", "b", "b"],
        }
    )
    original = df.copy(deep=True)
    profile, quality = build_profile_and_quality(df)
    config = CleaningConfig(missing_numeric_strategy="median")

    cleaned, _ = clean_dataset(df, profile, quality, config)

    pd.testing.assert_frame_equal(df, original)
    assert cleaned is not df


def test_does_not_remove_outliers_by_default():
    df = pd.DataFrame({"price": [100.0, 110.0, 120.0, 10000.0]})
    profile, quality = build_profile_and_quality(df)
    config = CleaningConfig(
        remove_duplicates=False,
        missing_numeric_strategy="keep",
    )

    cleaned, report = clean_dataset(df, profile, quality, config)

    assert len(cleaned) == len(df)
    assert cleaned["price"].max() == 10000.0
    assert report.final_rows == report.original_rows
    assert report.outlier_rows_removed == 0


def test_removes_outliers_from_selected_column():
    df = pd.DataFrame({"price": [100.0, 110.0, 120.0, 10000.0]})
    profile, quality = build_profile_and_quality(df)
    config = CleaningConfig(
        remove_duplicates=False,
        missing_numeric_strategy="keep",
        outlier_strategy="remove",
        outlier_columns={"price"},
    )

    cleaned, report = clean_dataset(df, profile, quality, config)

    assert len(cleaned) == 3
    assert 10000.0 not in cleaned["price"].values
    assert report.outlier_rows_removed == 1

    outlier_actions = [
        action for action in report.actions if action.action == "remove_outliers"
    ]
    assert len(outlier_actions) == 1
    assert outlier_actions[0].affected_rows == 1


def test_does_not_remove_outliers_from_unselected_columns():
    df = pd.DataFrame(
        {
            "price": [100.0, 110.0, 120.0, 10000.0],
            "age": [20.0, 21.0, 22.0, 23.0],
        }
    )
    profile, quality = build_profile_and_quality(df)
    config = CleaningConfig(
        remove_duplicates=False,
        missing_numeric_strategy="keep",
        outlier_strategy="remove",
        outlier_columns={"age"},
    )

    cleaned, report = clean_dataset(df, profile, quality, config)

    assert len(cleaned) == 4
    assert 10000.0 in cleaned["price"].values
    assert report.outlier_rows_removed == 0


def test_invalid_missing_strategy_raises_error():
    with pytest.raises(ValueError):
        CleaningConfig(missing_numeric_strategy="invalid")


def test_report_records_multiple_actions():
    df = pd.DataFrame({"price": [100.0, 100.0, np.nan, np.inf]})
    profile, quality = build_profile_and_quality(df)
    config = CleaningConfig(missing_numeric_strategy="median")

    _, report = clean_dataset(df, profile, quality, config)

    action_types = {action.action for action in report.actions}

    assert "remove_duplicates" in action_types
    assert "replace_invalid_values" in action_types
    assert "fill_missing_values" in action_types
