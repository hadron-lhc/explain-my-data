"""Tests del análisis temporal (F3)."""

import pandas as pd

from src.temporal import (
    analyze_temporal_series,
    generate_temporal_insights,
    infer_granularity,
    select_temporal_pairs,
)


def _monthly_frame(n: int = 48, slope: float = 2.0) -> pd.DataFrame:
    dates = pd.date_range("2021-01-01", periods=n, freq="ME")
    values = 100 + slope * pd.RangeIndex(n)

    return pd.DataFrame(
        {
            "month": dates.strftime("%Y-%m-%d"),
            "sales": values.astype(float),
        }
    )


def test_infer_granularity_daily_weekly_monthly_yearly():
    day = pd.Series(pd.date_range("2024-01-01", periods=30, freq="D"))
    week = pd.Series(pd.date_range("2024-01-01", periods=20, freq="W"))
    month = pd.Series(pd.date_range("2024-01-01", periods=24, freq="ME"))
    year = pd.Series(pd.date_range("2000-01-01", periods=10, freq="YE"))

    assert infer_granularity(day) == "day"
    assert infer_granularity(week) == "week"
    assert infer_granularity(month) == "month"
    assert infer_granularity(year) == "year"


def test_analyze_detects_linear_trend():
    df = _monthly_frame(slope=3.0)
    series = analyze_temporal_series(df, "month", "sales")

    assert series is not None
    assert series.granularity == "month"
    assert series.n_periods == 48
    assert series.trend is not None
    assert series.trend.direction == "up"
    assert series.trend.slope > 2.5
    assert series.missing_periods == []


def test_analyze_flat_series_has_no_trend():
    dates = pd.date_range("2021-01-01", periods=24, freq="ME")
    df = pd.DataFrame(
        {"month": dates.strftime("%Y-%m-%d"), "value": [100.0] * 24}
    )
    series = analyze_temporal_series(df, "month", "value")

    assert series is not None
    assert series.trend is None
    assert series.outliers == []


def test_analyze_detects_monthly_seasonality():
    n = 48
    dates = pd.date_range("2021-01-01", periods=n, freq="ME")
    seasonal = [100 + (50 if i % 12 == 11 else 0) for i in range(n)]
    df = pd.DataFrame(
        {"month": dates.strftime("%Y-%m-%d"), "sales": seasonal}
    )

    series = analyze_temporal_series(df, "month", "sales")

    assert series is not None
    assert series.seasonality is not None
    assert series.seasonality.lag == 12
    assert series.seasonality.cycles == 4


def test_analyze_counts_missing_periods():
    dates = pd.date_range("2021-01-01", periods=24, freq="ME")
    df = pd.DataFrame(
        {"month": dates.strftime("%Y-%m-%d"), "value": range(24)}
    ).drop(index=[5, 6, 10])

    series = analyze_temporal_series(df, "month", "value")

    assert series is not None
    assert len(series.missing_periods) == 3


def test_analyze_detects_regime_shift():
    n = 60
    dates = pd.date_range("2020-01-01", periods=n, freq="ME")
    values = [10.0] * 30 + [30.0] * 30
    df = pd.DataFrame({"month": dates.strftime("%Y-%m-%d"), "value": values})

    series = analyze_temporal_series(df, "month", "value")

    assert series is not None
    assert series.regime is not None
    assert series.regime.before_mean == 10.0
    assert series.regime.after_mean == 30.0
    assert series.regime.split_date == "2022-07-01"


def test_analyze_detects_extreme_periods():
    n = 24
    dates = pd.date_range("2021-01-01", periods=n, freq="ME")
    values = [100.0 + (i % 3) for i in range(n)]
    values[20] = 400.0
    df = pd.DataFrame({"month": dates.strftime("%Y-%m-%d"), "value": values})

    series = analyze_temporal_series(df, "month", "value")

    assert series is not None
    assert len(series.outliers) >= 1
    assert series.outliers[0].value == 400.0


def test_select_pairs_respects_score_floor_and_caps():
    n = 48
    dates = pd.date_range("2021-01-01", periods=n, freq="ME")
    rng = pd.Series(range(n), dtype=float)
    strong = (100 + 5 * rng).tolist()
    flat = [50.0] * n
    # Pseudoaleatorio determinístico: sin tendencia ni ciclo en lag 12.
    noise = [((i * 7919) % 101) - 50.0 for i in range(n)]

    df = pd.DataFrame(
        {
            "month": dates.strftime("%Y-%m-%d"),
            "strong": strong,
            "flat": flat,
            "noise": noise,
        }
    )

    pairs = select_temporal_pairs(df, ["month"], ["strong", "flat", "noise"])
    values = [value for _date, value in pairs]

    assert "strong" in values
    assert "noise" not in values


def test_select_pairs_caps_metrics_per_date():
    n = 48
    dates = pd.date_range("2021-01-01", periods=n, freq="ME")
    rng = pd.Series(range(n), dtype=float)
    df = pd.DataFrame({"month": dates.strftime("%Y-%m-%d")})

    for i in range(5):
        df[f"m{i}"] = (100 + (i + 1) * 5 * rng).tolist()

    pairs = select_temporal_pairs(df, ["month"], [f"m{i}" for i in range(5)])

    assert len(pairs) <= 3
    assert all(date == "month" for date, _value in pairs)


def test_select_pairs_caps_date_columns():
    n = 48
    dates = pd.date_range("2021-01-01", periods=n, freq="ME")
    rng = pd.Series(range(n), dtype=float)

    df = pd.DataFrame(
        {
            "date_a": dates.strftime("%Y-%m-%d"),
            "date_b": (dates + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            "date_c": (dates + pd.Timedelta(days=2)).strftime("%Y-%m-%d"),
            "value": (100 + 5 * rng).tolist(),
        }
    )

    pairs = select_temporal_pairs(
        df, ["date_a", "date_b", "date_c"], ["value"]
    )
    date_cols = {date for date, _value in pairs}

    assert len(date_cols) <= 2


def test_analyze_returns_none_below_min_periods():
    df = _monthly_frame(n=5)
    assert analyze_temporal_series(df, "month", "sales") is None


def test_generate_insights_messages_and_chart_id():
    df = _monthly_frame(n=48, slope=3.0)
    series = analyze_temporal_series(df, "month", "sales")
    assert series is not None

    insights = generate_temporal_insights(series)
    titles = [insight.title for insight in insights]

    assert any("trends up" in title for title in titles)
    assert all(insight.category == "temporal" for insight in insights)
    assert all(
        insight.chart_id == "temporal:month:sales" for insight in insights
    )
    assert all(
        insight.columns == ["month", "sales"] for insight in insights
    )
    assert all(
        "month means" in caveat
        for insight in insights
        for caveat in insight.caveats
    )
