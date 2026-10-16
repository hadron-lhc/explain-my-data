"""
Análisis temporal (F3): tendencia, estacionalidad, huecos,
cambio de régimen y extremos sobre medias de período.

Una columna datetime (detectada por el profiling) se empareja con
hasta `TEMPORAL_MAX_METRICS_PER_DATE` métricas numéricas. Las filas
se agregan al lag natural inferido de las fechas (día, semana,
mes, trimestre, año) y cada serie se estudia con reglas
deterministas sobre esas medias de período. Sin forecasting.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from scipy.stats import linregress

from src.config import (
    TEMPORAL_MAX_DATE_COLUMNS,
    TEMPORAL_MAX_METRICS_PER_DATE,
    TEMPORAL_MIN_PERIODS,
    TEMPORAL_OUTLIER_IQR_MULT,
    TEMPORAL_PAIR_MIN_SCORE,
    TEMPORAL_REGIME_MIN_SCORE,
    TEMPORAL_SEASONAL_MIN_AUTOCORR,
    TEMPORAL_TREND_MIN_PCT,
)
from src.insights import Insight

# Lag natural (en períodos) por granularidad; None = sin ciclo.
_NATURAL_LAG: dict[str, int | None] = {
    "day": 7,
    "week": 52,
    "month": 12,
    "quarter": 4,
    "year": None,
}

_PERIOD_FREQ: dict[str, str] = {
    "day": "D",
    "week": "W",
    "month": "M",
    "quarter": "Q",
    "year": "Y",
}


@dataclass
class TemporalTrend:
    slope: float
    pct_change: float | None
    direction: str  # "up" | "down"
    first_mean: float
    last_mean: float


@dataclass
class TemporalSeasonality:
    lag: int
    autocorr: float
    cycles: int


@dataclass
class TemporalRegime:
    split_date: str
    before_mean: float
    after_mean: float
    score: float


@dataclass
class TemporalOutlier:
    period: str
    value: float


@dataclass
class TemporalSeries:
    date_column: str
    value_column: str
    granularity: str
    n_rows: int
    n_periods: int
    start: str
    end: str
    periods: list[str]
    means: list[float]
    rolling_mean: list[float] | None
    rolling_window: int | None
    missing_periods: list[str]
    trend: TemporalTrend | None = None
    seasonality: TemporalSeasonality | None = None
    regime: TemporalRegime | None = None
    outliers: list[TemporalOutlier] = field(default_factory=list)


def infer_granularity(dates: pd.Series) -> str:
    """Granularidad natural: mediana del delta entre fechas únicas."""
    unique = pd.Series(dates.dropna().unique())
    unique = pd.to_datetime(unique).sort_values()

    if len(unique) < 2:
        return "day"

    deltas = unique.diff().dropna()

    if deltas.empty:
        return "day"

    median_days = float(deltas.dt.total_seconds().median()) / 86400.0

    if median_days <= 1.5:
        return "day"
    if median_days <= 10.0:
        return "week"
    if median_days <= 45.0:
        return "month"
    if median_days <= 135.0:
        return "quarter"

    return "year"


def _period_means(
    dates: pd.Series,
    values: pd.Series,
    granularity: str,
) -> pd.Series:
    parsed = pd.to_datetime(dates, errors="coerce", format="mixed")
    numeric = pd.to_numeric(values, errors="coerce")

    data = pd.DataFrame({"date": parsed, "value": numeric}).dropna()

    if data.empty:
        return pd.Series(dtype=float)

    periods = data["date"].dt.to_period(_PERIOD_FREQ[granularity])

    return data.groupby(periods)["value"].mean().sort_index()


def _autocorr_at(series: pd.Series, lag: int) -> float | None:
    values = series.to_numpy(dtype=float)

    if len(values) <= lag:
        return None

    left = values[:-lag]
    right = values[lag:]

    if left.std() == 0 or right.std() == 0:
        return None

    return float(
        pd.Series(left).corr(pd.Series(right), method="pearson")
    )


def _detrended(series: pd.Series) -> pd.Series:
    x = list(range(len(series)))
    result = linregress(x, series.to_numpy(dtype=float))
    fitted = result.intercept + result.slope * pd.Series(x, index=series.index)
    return series - fitted


def _trend_of(series: pd.Series) -> TemporalTrend | None:
    x = list(range(len(series)))
    result = linregress(x, series.to_numpy(dtype=float))

    first = float(series.iloc[0])
    last = float(series.iloc[-1])

    if first != 0:
        pct = (last - first) / abs(first) * 100.0
    elif last == 0:
        pct = 0.0
    else:
        pct = None

    if pct is not None and abs(pct) < TEMPORAL_TREND_MIN_PCT:
        return None

    if pct is None:
        spread = float(series.std())

        if spread == 0 or abs(result.slope) < 0.25 * spread:
            return None

    # Si la pendiente del ajuste lineal y el cambio neto first→last
    # van en signos opuestos, la tendencia no es robusta: un outlier
    # de palanca está distorsionando el ajuste. No emitimos tendencia.
    if pct is not None and result.slope != 0 and (pct > 0) != (result.slope > 0):
        return None

    direction = "up" if result.slope > 0 else "down"

    return TemporalTrend(
        slope=float(result.slope),
        pct_change=pct,
        direction=direction,
        first_mean=first,
        last_mean=last,
    )


def _seasonality_of(series: pd.Series, granularity: str) -> TemporalSeasonality | None:
    lag = _NATURAL_LAG[granularity]

    if lag is None or len(series) < 2 * lag:
        return None

    autocorr = _autocorr_at(_detrended(series), lag)

    if autocorr is None or abs(autocorr) < TEMPORAL_SEASONAL_MIN_AUTOCORR:
        return None

    return TemporalSeasonality(
        lag=lag,
        autocorr=autocorr,
        cycles=len(series) // lag,
    )


def _regime_of(series: pd.Series) -> TemporalRegime | None:
    """
    Detección de cambio de régimen sobre la serie DETRENDEADA.

    Sobre la serie cruda, cualquier tendencia lineal pronunciada
    dispara un "shifted around" falso (la media de la segunda mitad
    siempre supera a la de la primera). Al quitar la tendencia
    lineal primero, solo sobreviven los cambios de nivel reales.
    """
    n = len(series)

    if n < 6:
        return None

    work = _detrended(series)
    spread = float(work.std())

    if spread == 0:
        return None

    best: tuple[float, int] | None = None

    for split in range(int(0.3 * n), int(0.7 * n)):
        before = work.iloc[:split]
        after = work.iloc[split:]

        if len(before) < 3 or len(after) < 3:
            continue

        score = abs(float(after.mean() - before.mean())) / spread

        if best is None or score > best[0]:
            best = (score, split)

    if best is None or best[0] < TEMPORAL_REGIME_MIN_SCORE:
        return None

    score, split = best
    # Las medias reportadas son de la serie cruda (unidad original).
    before_mean = float(series.iloc[:split].mean())
    after_mean = float(series.iloc[split:].mean())

    return TemporalRegime(
        split_date=str(series.index[split].start_time.date()),
        before_mean=before_mean,
        after_mean=after_mean,
        score=score,
    )


def _outliers_of(series: pd.Series) -> list[TemporalOutlier]:
    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr = q3 - q1

    if iqr == 0:
        return []

    lower = q1 - TEMPORAL_OUTLIER_IQR_MULT * iqr
    upper = q3 + TEMPORAL_OUTLIER_IQR_MULT * iqr

    return [
        TemporalOutlier(
            period=str(period.start_time.date()),
            value=float(value),
        )
        for period, value in series.items()
        if value < lower or value > upper
    ]


def analyze_temporal_series(
    df: pd.DataFrame,
    date_column: str,
    value_column: str,
) -> TemporalSeries | None:
    """Analiza una serie (fecha, métrica); None si no alcanza el mínimo."""
    if date_column not in df.columns or value_column not in df.columns:
        return None

    parsed_dates = pd.to_datetime(
        df[date_column], errors="coerce", format="mixed"
    )
    numeric = pd.to_numeric(df[value_column], errors="coerce")

    valid = int((parsed_dates.notna() & numeric.notna()).sum())

    if valid < TEMPORAL_MIN_PERIODS:
        return None

    granularity = infer_granularity(parsed_dates)

    means = _period_means(df[date_column], df[value_column], granularity)

    if len(means) < TEMPORAL_MIN_PERIODS:
        return None

    grid = pd.period_range(
        means.index.min(), means.index.max(), freq=_PERIOD_FREQ[granularity]
    )
    missing = [str(p.start_time.date()) for p in grid.difference(means.index)]

    rolling_window = 3 if len(means) >= 6 else None
    rolling = (
        means.rolling(rolling_window).mean().round(4).tolist()
        if rolling_window
        else None
    )

    return TemporalSeries(
        date_column=date_column,
        value_column=value_column,
        granularity=granularity,
        n_rows=valid,
        n_periods=len(means),
        start=str(means.index.min().start_time.date()),
        end=str(means.index.max().start_time.date()),
        periods=[str(p.start_time.date()) for p in means.index],
        means=[float(v) for v in means],
        rolling_mean=rolling,
        rolling_window=rolling_window,
        missing_periods=missing,
        trend=_trend_of(means),
        seasonality=_seasonality_of(means, granularity),
        regime=_regime_of(means),
        outliers=_outliers_of(means),
    )


def select_temporal_pairs(
    df: pd.DataFrame,
    datetime_columns: list[str],
    numeric_columns: list[str],
) -> list[tuple[str, str]]:
    """
    Empareja cada columna datetime (top TEMPORAL_MAX_DATE_COLUMNS)
    con sus mejores métricas: score = max(|spearman| con el tiempo,
    |autocorr| en el lag natural). Determinístico: empates por nombre.
    """
    scored_dates: list[tuple[float, str]] = []

    for date_column in datetime_columns:
        parsed = pd.to_datetime(
            df[date_column], errors="coerce", format="mixed"
        )

        metric_scores: list[tuple[float, str]] = []

        for value_column in numeric_columns:
            numeric = pd.to_numeric(df[value_column], errors="coerce")
            data = pd.DataFrame({"date": parsed, "value": numeric}).dropna()

            if len(data) < TEMPORAL_MIN_PERIODS:
                continue

            granularity = infer_granularity(data["date"])
            means = _period_means(data["date"], data["value"], granularity)

            if len(means) < TEMPORAL_MIN_PERIODS:
                continue

            trend_strength = abs(
                float(
                    pd.Series(range(len(means)))
                    .corr(pd.Series(means.to_numpy()), method="spearman")
                )
            )

            lag = _NATURAL_LAG[granularity]
            seasonal_strength = 0.0

            if lag is not None and len(means) >= 2 * lag:
                autocorr = _autocorr_at(_detrended(means), lag)

                if autocorr is not None:
                    seasonal_strength = abs(autocorr)

            score = max(trend_strength, seasonal_strength)

            if score >= TEMPORAL_PAIR_MIN_SCORE:
                metric_scores.append((score, value_column))

        if metric_scores:
            metric_scores.sort(key=lambda item: (-item[0], item[1]))
            best = metric_scores[0][0]
            scored_dates.append((best, date_column))

    scored_dates.sort(key=lambda item: (-item[0], item[1]))

    pairs: list[tuple[str, str]] = []

    for _score, date_column in scored_dates[:TEMPORAL_MAX_DATE_COLUMNS]:
        parsed = pd.to_datetime(
            df[date_column], errors="coerce", format="mixed"
        )
        metrics: list[tuple[float, str]] = []

        for value_column in numeric_columns:
            numeric = pd.to_numeric(df[value_column], errors="coerce")
            data = pd.DataFrame({"date": parsed, "value": numeric}).dropna()

            if len(data) < TEMPORAL_MIN_PERIODS:
                continue

            granularity = infer_granularity(data["date"])
            means = _period_means(data["date"], data["value"], granularity)

            if len(means) < TEMPORAL_MIN_PERIODS:
                continue

            trend_strength = abs(
                float(
                    pd.Series(range(len(means)))
                    .corr(pd.Series(means.to_numpy()), method="spearman")
                )
            )

            lag = _NATURAL_LAG[granularity]
            seasonal_strength = 0.0

            if lag is not None and len(means) >= 2 * lag:
                autocorr = _autocorr_at(_detrended(means), lag)

                if autocorr is not None:
                    seasonal_strength = abs(autocorr)

            score = max(trend_strength, seasonal_strength)

            if score >= TEMPORAL_PAIR_MIN_SCORE:
                metrics.append((score, value_column))

        metrics.sort(key=lambda item: (-item[0], item[1]))

        for _score, value_column in metrics[:TEMPORAL_MAX_METRICS_PER_DATE]:
            pairs.append((date_column, value_column))

    return pairs


def generate_temporal_insights(series: TemporalSeries) -> list[Insight]:
    """Insights en inglés (estilo del resto del pipeline)."""
    chart_id = f"temporal:{series.date_column}:{series.value_column}"
    columns = [series.date_column, series.value_column]
    caveat = f"values aggregated as {series.granularity} means"
    insights: list[Insight] = []

    if series.trend is not None:
        trend = series.trend
        verb = "rose" if trend.direction == "up" else "fell"
        pct = (
            f"{trend.pct_change:+.1f}%"
            if trend.pct_change is not None
            else "n/a"
        )
        trend_evidence: dict[str, float | int | str] = {
            "slope": round(trend.slope, 4),
            "direction": trend.direction,
            "first_mean": round(trend.first_mean, 4),
            "last_mean": round(trend.last_mean, 4),
            "n_periods": series.n_periods,
        }

        if trend.pct_change is not None:
            trend_evidence["pct_change"] = round(trend.pct_change, 2)

        insights.append(
            Insight(
                title=(
                    f"{series.value_column} trends {trend.direction} "
                    f"over {series.date_column}"
                ),
                message=(
                    f"From {series.start} to {series.end}, the average "
                    f"{series.value_column} {verb} "
                    f"{pct} ({trend.first_mean:.2f} → {trend.last_mean:.2f} "
                    f"per {series.granularity}) — slope {trend.slope:+.2f} per "
                    f"{series.granularity} across {series.n_periods} periods."
                ),
                severity="info",
                score=abs(trend.pct_change or abs(trend.slope)),
                category="temporal",
                columns=columns,
                chart_id=chart_id,
                evidence=trend_evidence,
                caveats=[caveat],
            )
        )

    if series.seasonality is not None:
        season = series.seasonality
        insights.append(
            Insight(
                title=(
                    f"{series.value_column} shows a {season.lag}-"
                    f"{series.granularity} cycle"
                ),
                message=(
                    f"Detrended {series.value_column} repeats with "
                    f"autocorrelation {season.autocorr:.2f} at lag "
                    f"{season.lag} {series.granularity}s "
                    f"({season.cycles} cycles observed)."
                ),
                severity="info",
                score=abs(season.autocorr) * 10.0,
                category="temporal",
                columns=columns,
                chart_id=chart_id,
                evidence={
                    "lag": season.lag,
                    "autocorr": round(season.autocorr, 4),
                    "cycles": season.cycles,
                },
                caveats=[caveat],
            )
        )

    if series.regime is not None:
        regime = series.regime
        delta_pct = (
            (regime.after_mean - regime.before_mean)
            / abs(regime.before_mean)
            * 100.0
            if regime.before_mean != 0
            else None
        )
        delta_text = f"{delta_pct:+.1f}%" if delta_pct is not None else "n/a"
        insights.append(
            Insight(
                title=(
                    f"{series.value_column} shifted around "
                    f"{regime.split_date}"
                ),
                message=(
                    f"Average {series.value_column} moved from "
                    f"{regime.before_mean:.2f} to {regime.after_mean:.2f} "
                    f"({delta_text}, score {regime.score:.2f}σ around "
                    f"{regime.split_date} per {series.granularity})."
                ),
                severity="info",
                score=regime.score * 10.0,
                category="temporal",
                columns=columns,
                chart_id=chart_id,
                evidence={
                    "split_date": regime.split_date,
                    "before_mean": round(regime.before_mean, 4),
                    "after_mean": round(regime.after_mean, 4),
                    "score": round(regime.score, 4),
                },
                caveats=[caveat],
            )
        )

    if series.missing_periods:
        n_missing = len(series.missing_periods)
        expected = series.n_periods + n_missing
        insights.append(
            Insight(
                title=(
                    f"{series.date_column} has {n_missing} missing "
                    f"{series.granularity}s"
                ),
                message=(
                    f"The {series.start}–{series.end} range implies "
                    f"{expected} {series.granularity}s at {series.granularity} "
                    f"granularity; {n_missing} are absent from the data "
                    f"(first gap: {series.missing_periods[0]})."
                ),
                severity="info",
                score=n_missing / expected * 100.0,
                category="temporal",
                columns=columns,
                chart_id=chart_id,
                evidence={
                    "missing_count": n_missing,
                    "expected_periods": expected,
                    "first_gap": series.missing_periods[0],
                },
                caveats=[caveat],
            )
        )

    if series.outliers:
        worst = max(series.outliers, key=lambda item: abs(item.value))
        listed = ", ".join(
            f"{item.period} ({item.value:.2f})"
            for item in sorted(series.outliers, key=lambda i: i.period)[:5]
        )
        n_out = len(series.outliers)
        unit = "mean" if n_out == 1 else "means"
        insights.append(
            Insight(
                title=(
                    f"{series.value_column} contains "
                    f"{n_out} extreme {series.granularity} {unit}"
                ),
                message=(
                    f"{n_out} {series.granularity} means lie "
                    f"outside 3×IQR of period means; most extreme: "
                    f"{worst.period} ({worst.value:.2f}). Outliers: {listed}."
                ),
                severity="info",
                score=len(series.outliers) * 10.0,
                category="temporal",
                columns=columns,
                chart_id=chart_id,
                evidence={
                    "outlier_count": len(series.outliers),
                    "worst_period": worst.period,
                    "worst_value": round(worst.value, 4),
                },
                caveats=[caveat],
            )
        )

    return insights
