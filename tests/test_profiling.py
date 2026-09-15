import pandas as pd
from src.profiling import infer_role
from src.profiling import infer_role, profile_column
from src.profiling import infer_role, profile_column, profile_dataset


def test_infer_role():
    test_cases = [
        (
            pd.Series(range(100, 200)),
            "customer_id",
            "identifier",
        ),
        (
            pd.Series([20, 21, 22, 23, 24] * 20),
            "age",
            "numeric",
        ),
        (
            pd.Series(["yes", "no", "yes", "yes"]),
            "active",
            "boolean",
        ),
        (
            pd.Series(
                [
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-03",
                    "2026-01-04",
                ]
            ),
            "created_at",
            "datetime",
        ),
        (
            pd.Series(["MDQ", "BA", "MDQ", "BA"]),
            "city",
            "categorical",
        ),
        (
            pd.Series([1.2, 2.5, 3.1, 4.7]),
            "score",
            "numeric",
        ),
    ]

    for series, name, expected_role in test_cases:
        role, _ = infer_role(series, name)

        assert role == expected_role, (
            f"{name!r}: expected {expected_role!r}, got {role!r}"
        )


def test_profile_column():
    series = pd.Series([10, 20, None, 40, 20])

    profile = profile_column(series, "age")

    assert profile.name == "age"
    assert profile.dtype == "float64"
    assert profile.role == "numeric"

    assert profile.n_missing == 1
    assert profile.pct_missing == 20.0

    assert profile.n_unique == 3
    assert profile.pct_unique == 60.0

    assert profile.sample_values == [10.0, 20.0, 40.0]


def test_profile_dataset():
    df = pd.DataFrame(
        {
            "customer_id": range(100, 120),
            "age": [20, 21, 22, 23, 24] * 4,
            "active": ["yes", "no", "yes", "yes", "no"] * 4,
            "city": ["MDQ", "BA"] * 10,
        }
    )

    profile = profile_dataset(df)

    assert profile.n_rows == 20
    assert profile.n_columns == 4
    assert profile.n_duplicates == 0

    assert len(profile.columns) == 4

    roles = {column.name: column.role for column in profile.columns}

    assert roles == {
        "customer_id": "identifier",
        "age": "numeric",
        "active": "boolean",
        "city": "categorical",
    }
