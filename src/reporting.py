"""
reporting.py

Renderiza un informe HTML autocontenido a partir de un AnalysisReport.

El texto del informe proviene de las plantillas de insights
(sin modelos de lenguaje). Los gráficos se construyen con plotly
y se emben vía CDN.

El documento está pensado también para imprimirse ("print-to-PDF"):
cada `.page` es una diapositiva y el hash `#print` dispara el
diálogo de impresión del navegador.
"""

from __future__ import annotations

import html
from dataclasses import replace
from datetime import UTC, datetime

import plotly.graph_objects as go

from src.analysis import AnalysisReport
from src.insights import (
    Insight,
    chart_id_categorical_categorical,
    chart_id_categorical_numeric,
    chart_id_histogram,
    chart_id_scatter,
)

# Campos de AnalysisReport que contienen insights renderizables.
_INSIGHT_LIST_FIELDS = (
    "quality_insights",
    "relationship_insights",
    "distribution_insights",
    "pattern_insights",
    "structure_insights",
    "aggregation_insights",
    "comparative_insights",
    "temporal_insights",
    "inference_insights",
    "executive_summary",
)


def _filter_hidden_insights(
    report: AnalysisReport,
    hide_ids: set[str] | None,
) -> AnalysisReport:
    """Copia del reporte sin los insights cuyo id está en `hide_ids`."""
    if not hide_ids:
        return report

    updates = {
        field_name: [
            insight
            for insight in getattr(report, field_name)
            if insight.id not in hide_ids
        ]
        for field_name in _INSIGHT_LIST_FIELDS
    }

    return replace(report, **updates)

_STYLESHEET = """
* { box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  margin: 0; padding: 24px; background: #f6f7f9; color: #1f2933;
}
main { max-width: 1100px; margin: 0 auto; }
h1 { font-size: 1.6rem; margin: 0 0 4px; }
h2 { font-size: 1.2rem; margin: 0 0 12px; border-bottom: 2px solid #d9e2ec; padding-bottom: 6px; }
h3 { font-size: 1rem; margin: 20px 0 8px; }
p { line-height: 1.5; }
.meta { color: #616e7c; font-size: 0.9rem; margin-bottom: 24px; }
.section-sub { color: #616e7c; font-size: 0.9rem; margin: -4px 0 14px; }

.page {
  background: #fff; border: 1px solid #e4e7eb; border-radius: 10px;
  padding: 28px; margin: 0 auto 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}

.cover { text-align: center; padding: 72px 28px; }
.cover h1 { font-size: 2.4rem; margin-bottom: 8px; }
.cover-subtitle { color: #486581; font-size: 1.05rem; margin-bottom: 36px; }
.cover-stats { display: flex; justify-content: center; gap: 16px; flex-wrap: wrap; margin-bottom: 36px; }
.cover-stat {
  background: #f8fafc; border: 1px solid #e4e7eb; border-radius: 10px;
  padding: 16px 22px; min-width: 120px;
}
.cover-stat strong { display: block; font-size: 1.6rem; }
.cover-stat span { color: #616e7c; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.04em; }

.kpis { display: flex; gap: 12px; flex-wrap: wrap; margin: 0 0 16px; }
.kpi {
  background: #f8fafc; border: 1px solid #e4e7eb; border-radius: 8px;
  padding: 10px 16px; min-width: 110px;
}
.kpi strong { display: block; font-size: 1.3rem; }
.kpi span { color: #616e7c; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em; }

.card {
  background: #fff; border: 1px solid #e4e7eb; border-radius: 8px;
  padding: 16px; margin: 12px 0; box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}
.insight-title { font-weight: 600; margin-bottom: 4px; }
.insight-message { color: #334e68; margin-bottom: 6px; }
.evidence { margin: 8px 0 4px; padding-left: 18px; color: #486581; font-size: 0.85rem; }
.evidence li { margin-bottom: 2px; }
.badge {
  display: inline-block; font-size: 0.7rem; font-weight: 700;
  padding: 2px 8px; border-radius: 999px; margin-right: 6px;
  text-transform: uppercase;
}
.badge-warning { background: #ffe3e3; color: #c92a2a; }
.badge-info { background: #d0ebff; color: #1864ab; }
.badge-category { background: #e6fcf5; color: #087f5b; }
.caveat { color: #8a6d3b; font-size: 0.85rem; font-style: italic; }
.page-kicker {
  color: #829ab1; font-size: 0.75rem; font-weight: 700;
  text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 10px;
}
table { border-collapse: collapse; width: 100%; font-size: 0.85rem; }
th, td { border: 1px solid #e4e7eb; padding: 6px 10px; text-align: left; }
th { background: #f1f3f5; }
.chart { background: #fff; border: 1px solid #e4e7eb; border-radius: 8px; padding: 12px; margin: 12px 0; }
.empty { color: #616e7c; font-style: italic; }

@page { size: A4; margin: 14mm; }

@media print {
  body { background: #fff; padding: 0; font-size: 12px; }
  main { max-width: none; }
  .page {
    margin: 0; padding: 0; border: none; border-radius: 0;
    box-shadow: none; page-break-after: always;
  }
  .page:last-of-type { page-break-after: auto; }
  h2 { margin-top: 0; }
  .cover { display: flex; flex-direction: column; justify-content: center; min-height: 230mm; }
  .card, .chart, table, .kpis, .cover-stats { break-inside: avoid; }
  .chart { padding: 4px; }
}
"""

_AUTO_PRINT_SCRIPT = """
window.addEventListener("load", function () {
  if (window.location.hash !== "#print") return;
  setTimeout(function () { window.print(); }, 700);
});
"""


def _esc(value: object) -> str:
    return html.escape(str(value))


def _table(headers: list[str], rows: list[list[object]]) -> str:
    if not rows:
        return '<p class="empty">No data.</p>'

    header_html = "".join(f"<th>{_esc(header)}</th>" for header in headers)

    body_rows = []

    for row in rows:
        cells = "".join(f"<td>{_esc(cell)}</td>" for cell in row)
        body_rows.append(f"<tr>{cells}</tr>")

    return (
        "<table>"
        f"<tr>{header_html}</tr>"
        + "".join(body_rows)
        + "</table>"
    )


def _insight_html(insight: Insight) -> str:
    caveats = "".join(
        f'<div class="caveat">caveat: {_esc(caveat)}</div>'
        for caveat in insight.caveats
    )

    evidence_items = "".join(
        f"<li>{_esc(key)}: {_esc(value)}</li>"
        for key, value in insight.evidence.items()
    )

    evidence = (
        f'<ul class="evidence">{evidence_items}</ul>' if evidence_items else ""
    )

    return (
        '<div class="card">'
        "<div>"
        f'<span class="badge badge-{_esc(insight.severity)}">'
        f"{_esc(insight.severity)}</span>"
        f'<span class="badge badge-category">{_esc(insight.category)}</span>'
        "</div>"
        f'<div class="insight-title">{_esc(insight.title)}</div>'
        f'<div class="insight-message">{_esc(insight.message)}</div>'
        f"{evidence}"
        f"{caveats}"
        "</div>"
    )


def _insights_list(insights: list[Insight], empty_text: str) -> str:
    if not insights:
        return f'<p class="empty">{_esc(empty_text)}</p>'

    return "".join(_insight_html(insight) for insight in insights)


def _chart_wrap(figure: go.Figure) -> str:
    return (
        f'<div class="chart">'
        f"{figure.to_html(full_html=False, include_plotlyjs=False)}"
        f"</div>"
    )


def _scatter_chunk(scatter) -> str:
    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=scatter.x_values,
            y=scatter.y_values,
            mode="markers",
            name="observations",
            marker={"size": 5, "opacity": 0.6},
        )
    )

    if scatter.trend_x is not None and scatter.trend_y is not None:
        figure.add_trace(
            go.Scatter(
                x=scatter.trend_x,
                y=scatter.trend_y,
                mode="lines",
                name="trend",
                line={"color": "firebrick", "width": 2},
            )
        )

    figure.update_layout(
        title=f"{scatter.column_x} vs {scatter.column_y}",
        xaxis_title=scatter.column_x,
        yaxis_title=scatter.column_y,
        height=420,
        margin={"t": 40, "b": 40, "l": 40, "r": 20},
    )

    return _chart_wrap(figure)


def _line_chunk(line) -> str:
    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=line.periods,
            y=line.means,
            mode="lines+markers",
            name=f"mean {line.column_value}",
            line={"color": "steelblue", "width": 2},
            marker={"size": 4},
        )
    )

    if line.rolling_mean is not None:
        figure.add_trace(
            go.Scatter(
                x=line.periods,
                y=line.rolling_mean,
                mode="lines",
                name=f"rolling({line.rolling_window})",
                line={"color": "darkorange", "width": 2, "dash": "dash"},
            )
        )

    figure.update_layout(
        title=f"{line.column_value} over {line.column_date}",
        xaxis_title=line.column_date,
        yaxis_title=line.column_value,
        height=420,
        margin={"t": 40, "b": 80, "l": 40, "r": 20},
    )

    return _chart_wrap(figure)


def _bar_chunk(bar) -> str:
    figure = go.Figure(
        go.Bar(
            x=bar.categories,
            y=bar.values,
            name=bar.column_numeric,
        )
    )

    figure.update_layout(
        title=f"Average {bar.column_numeric} by {bar.column_categorical}",
        xaxis_title=bar.column_categorical,
        yaxis_title=bar.column_numeric,
        height=420,
        margin={"t": 40, "b": 80, "l": 40, "r": 20},
    )

    return _chart_wrap(figure)


def _boxplot_chunk(box) -> str:
    grouped: dict[str, list[float]] = {}

    for category, value in zip(box.categories, box.values):
        grouped.setdefault(str(category), []).append(float(value))

    # Limitar el payload del gráfico.
    grouped = dict(list(grouped.items())[:15])

    figure = go.Figure(
        data=[
            go.Box(
                y=values,
                name=category,
                boxpoints="outliers",
            )
            for category, values in grouped.items()
        ]
    )

    figure.update_layout(
        title=f"{box.column_numeric} by {box.column_categorical}",
        yaxis_title=box.column_numeric,
        height=420,
        showlegend=False,
        margin={"t": 40, "b": 40, "l": 40, "r": 20},
    )

    return _chart_wrap(figure)


def _heatmap_chunk(heat) -> str:
    figure = go.Figure(
        go.Heatmap(
            x=heat.x_labels,
            y=heat.y_labels,
            z=heat.values,
            colorscale="Blues",
            hovertemplate=(
                f"{heat.column_x}: %{{x}}<br>{heat.column_y}: %{{y}}<br>"
                f"{heat.value_label}: %{{z}}<extra></extra>"
            ),
        )
    )

    figure.update_layout(
        title=f"{heat.column_x} vs {heat.column_y} ({heat.value_label})",
        xaxis_title=heat.column_x,
        yaxis_title=heat.column_y,
        height=420,
        margin={"t": 40, "b": 60, "l": 60, "r": 20},
    )

    return _chart_wrap(figure)


def _histogram_chunk(distribution) -> str:
    midpoints = [
        (distribution.bin_edges[i] + distribution.bin_edges[i + 1]) / 2
        for i in range(len(distribution.bin_counts))
    ]

    figure = go.Figure(
        go.Bar(
            x=midpoints,
            y=distribution.bin_counts,
            name=distribution.column_name,
        )
    )

    figure.update_layout(
        title=f"Distribution of {distribution.column_name}",
        xaxis_title=distribution.column_name,
        yaxis_title="count",
        height=360,
        margin={"t": 40, "b": 40, "l": 40, "r": 20},
    )

    return _chart_wrap(figure)


def _sorted_pair(column_x: str, column_y: str) -> str:
    """Par ordenado lexicográficamente, igual que la UI (visualizationIdentity)."""
    return ":".join(sorted((column_x, column_y)))


def _figure_chunks(report: AnalysisReport) -> list[tuple[str, str, str]]:
    """
    Lista de figuras como (chart_id de backend, identidad de UI, chunk).

    La identidad de UI es espejo exacto de `visualizationIdentity` en
    `web/frontend/charts.js`: es la que la UI usa para ocultar charts
    (`?hide_charts=...`), por lo que el orden de grupos también es el
    mismo que el de la sección "Additional charts" en pantalla.
    """
    chunks: list[tuple[str, str, str]] = []

    for distribution in report.numeric_distributions:
        if distribution.bin_counts:
            column = distribution.column_name
            chunks.append(
                (
                    chart_id_histogram(column),
                    f"histogram:{column}",
                    _histogram_chunk(distribution),
                )
            )

    for heat in report.heatmaps:
        chunks.append(
            (
                chart_id_categorical_categorical(heat.column_x, heat.column_y),
                f"heatmap:{_sorted_pair(heat.column_x, heat.column_y)}",
                _heatmap_chunk(heat),
            )
        )

    for line in report.line_plots:
        chunks.append(
            (
                f"temporal:{line.column_date}:{line.column_value}",
                f"line:{line.column_date}:{line.column_value}",
                _line_chunk(line),
            )
        )

    for scatter in report.scatter_plots:
        chunks.append(
            (
                chart_id_scatter(scatter.column_x, scatter.column_y),
                f"scatter:{_sorted_pair(scatter.column_x, scatter.column_y)}",
                _scatter_chunk(scatter),
            )
        )

    for bar in report.bar_plots:
        chunks.append(
            (
                chart_id_categorical_numeric(
                    bar.column_categorical,
                    bar.column_numeric,
                ),
                f"bar:{bar.column_categorical}:{bar.column_numeric}",
                _bar_chunk(bar),
            )
        )

    for box in report.boxplots:
        chunks.append(
            (
                chart_id_categorical_numeric(
                    box.column_categorical,
                    box.column_numeric,
                ),
                f"boxplot:{box.column_categorical}:{box.column_numeric}",
                _boxplot_chunk(box),
            )
        )

    return chunks


def _figures_by_chart_id(report: AnalysisReport) -> dict[str, list[str]]:
    """Indexa cada figura por el chart_id que la referencia."""

    figures: dict[str, list[str]] = {}

    for key, _ui_identity, chunk in _figure_chunks(report):
        figures.setdefault(key, []).append(chunk)

    return figures


def _resolve_figure_keys(
    chart_id: str | None,
    figures: dict[str, list[str]],
) -> list[str]:
    if chart_id is None:
        return []

    kind, *parts = chart_id.split(":")

    if kind in {"scatter", "catcat"} and len(parts) >= 2:
        reverse = f"{kind}:{parts[1]}:{parts[0]}"

        if chart_id in figures:
            return [chart_id]

        if reverse in figures:
            return [reverse]

        return []

    if chart_id in figures:
        return [chart_id]

    return []


def _kpi(label: str, value: object) -> str:
    return (
        '<div class="kpi">'
        f"<strong>{_esc(value)}</strong>"
        f"<span>{_esc(label)}</span>"
        "</div>"
    )


def _patterns_section(report: AnalysisReport) -> str:
    group_patterns = sorted(
        report.group_difference_patterns,
        key=lambda pattern: abs(pattern.standardized_difference),
        reverse=True,
    )

    group_rows = [
        [
            pattern.group_column,
            pattern.group_value,
            f"{pattern.group_mean:.2f}",
            f"{pattern.overall_mean:.2f}",
            f"{pattern.difference:.2f}",
            f"{pattern.standardized_difference:.2f}",
            pattern.group_size,
        ]
        for pattern in group_patterns
    ]

    outlier_patterns = sorted(
        report.numeric_outlier_patterns,
        key=lambda pattern: pattern.outlier_count,
        reverse=True,
    )

    outlier_rows = [
        [
            pattern.column,
            pattern.outlier_count,
            (
                f"{pattern.outlier_count / pattern.n_observations * 100:.1f}%"
                if pattern.n_observations
                else "-"
            ),
            f"{pattern.lower_bound:.2f}",
            f"{pattern.upper_bound:.2f}",
        ]
        for pattern in outlier_patterns
        if pattern.outlier_count > 0
    ]

    if not group_rows and not outlier_rows:
        return '<p class="empty">No structural patterns detected.</p>'

    parts: list[str] = []

    if group_rows:
        parts.append("<h3>Group differences</h3>")
        parts.append(
            _table(
                [
                    "column",
                    "group",
                    "group mean",
                    "overall mean",
                    "difference",
                    "std dev",
                    "group size",
                ],
                group_rows,
            )
        )

    if outlier_rows:
        parts.append("<h3>Numeric outliers</h3>")
        parts.append(
            _table(
                [
                    "column",
                    "outliers",
                    "% of observations",
                    "lower bound",
                    "upper bound",
                ],
                outlier_rows,
            )
        )

    return "".join(parts)


def _insight_pages(
    all_insights: list[Insight],
    figures: dict[str, list[str]],
    consumed: set[str],
) -> str:
    pages: list[str] = []

    for position, insight in enumerate(all_insights, start=1):
        keys = _resolve_figure_keys(insight.chart_id, figures)
        consumed.update(keys)

        charts = "".join(
            chunk for key in keys for chunk in figures[key][:2]
        )

        pages.append(
            '<section class="page insight-page">'
            f'<div class="page-kicker">Insight {position} · '
            f"{_esc(insight.category)}</div>"
            f"{_insight_html(insight)}"
            f"{charts}"
            "</section>"
        )

    if not pages:
        return (
            '<section class="page">'
            "<h2>Insights</h2>"
            '<p class="empty">No insights detected.</p>'
            "</section>"
        )

    return "".join(pages)


def _extra_charts_section(
    report: AnalysisReport,
    consumed: set[str],
    hide_charts: set[str] | None = None,
) -> str:
    hidden = hide_charts or set()
    hidden_skipped = 0

    chunks: list[str] = []

    for key, ui_identity, chunk in _figure_chunks(report):
        if key in consumed:
            continue

        if ui_identity in hidden:
            hidden_skipped += 1
            continue

        chunks.append(chunk)

    if not chunks:
        if hidden_skipped:
            return (
                '<section class="page">'
                "<h2>Additional charts</h2>"
                '<p class="empty">All extra charts are hidden from the '
                "UI.</p>"
                "</section>"
            )

        return ""

    return (
        '<section class="page">'
        "<h2>Additional charts</h2>"
        '<p class="section-sub">Charts not attached to a specific insight.</p>'
        f"{''.join(chunks)}"
        "</section>"
    )


def _quality_findings_section(report: AnalysisReport) -> str:
    if not report.quality_insights:
        return ""

    return "<h3>Quality findings</h3>" + _insights_list(
        report.quality_insights,
        "No quality issues detected.",
    )


def _cleaning_recommendations_section(report: AnalysisReport) -> str:
    if not report.cleaning_recommendations:
        return ""

    rows = [
        [
            rec.column_name or "(dataset)",
            rec.action,
            rec.affected_rows,
            rec.why,
            rec.impact,
        ]
        for rec in report.cleaning_recommendations
    ]

    table = _table(
        ["Column", "Suggested action", "Rows", "Why", "Impact"],
        rows,
    )

    return (
        "<h3>Cleaning recommendations</h3>"
        "<p><em>Suggested only — nothing is applied automatically. "
        "Apply explicitly via the cleaning pipeline.</em></p>"
        + table
    )


def render_html_report(
    report: AnalysisReport,
    hide_ids: set[str] | None = None,
    hide_charts: set[str] | None = None,
) -> str:
    """
    Genera el informe HTML completo a partir del reporte de análisis.

    `hide_ids` excluye insights que el usuario ocultó en la UI y
    `hide_charts` excluye charts de "Additional charts" (identidades
    `visualizationIdentity` de la UI), para que el PDF impreso
    coincida con lo que se ve en pantalla.
    """
    report = _filter_hidden_insights(report, hide_ids)
    profile = report.profile

    profile_rows = [
        [
            column.name,
            column.dtype,
            column.role,
            f"{column.pct_missing:.1f}%",
            column.n_unique,
        ]
        for column in profile.columns
    ]

    quality_rows = [
        [
            column.column_name,
            column.missing_count,
            f"{column.outlier_percentage:.1f}%",
            column.n_extreme_values,
            column.is_type_mismatch,
        ]
        for column in report.quality.columns
    ]

    numeric_rows = [
        [
            stats.column_name,
            stats.count,
            f"{stats.mean:.2f}",
            f"{stats.median:.2f}",
            f"{stats.std:.2f}",
            f"{stats.min:.2f}",
            f"{stats.max:.2f}",
            f"{stats.skewness:.2f}",
        ]
        for stats in report.numeric_statistics
    ]

    categorical_rows = [
        [
            stats.column_name,
            stats.count,
            stats.n_unique,
            stats.dominant_value,
            f"{stats.dominant_percentage:.1f}%",
        ]
        for stats in report.categorical_statistics
    ]

    distribution_rows = [
        [
            dist.column_name,
            dist.n_observations,
            f"{dist.skewness:.2f}",
            f"{dist.kurtosis:.2f}",
            dist.modality,
            (
                f"{dist.top_5pct_share:.1f}%"
                if dist.top_5pct_share is not None
                else "-"
            ),
        ]
        for dist in report.numeric_distributions
    ]

    numeric_rel_rows = [
        [
            rel.column_x,
            rel.column_y,
            f"{rel.pearson:.3f}",
            f"{rel.spearman:.3f}",
            rel.n_observations,
        ]
        for rel in report.numeric_relationships
    ]

    cat_num_rel_rows = [
        [
            rel.column_categorical,
            rel.column_numeric,
            f"{rel.group_separation:.2f}",
            rel.n_groups,
            rel.n_observations,
        ]
        for rel in report.categorical_numeric_relationships
    ]

    cat_cat_rel_rows = [
        [
            rel.column_x,
            rel.column_y,
            f"{rel.cramers_v:.3f}",
            f"{rel.chi2:.2f}",
            f"{rel.p_value:.2g}",
        ]
        for rel in report.categorical_categorical_relationships
    ]

    temporal_rows = []

    for series in report.temporal_series:
        trend_text = "—"

        if series.trend is not None:
            pct = (
                f"{series.trend.pct_change:+.1f}%"
                if series.trend.pct_change is not None
                else "n/a"
            )
            trend_text = f"{series.trend.direction} {pct}"

        temporal_rows.append(
            [
                series.date_column,
                series.value_column,
                series.granularity,
                series.n_periods,
                series.start,
                series.end,
                trend_text,
                len(series.missing_periods),
            ]
        )

    if temporal_rows:
        temporal_section = (
            '<section class="page">'
            "<h2>Temporal series</h2>"
            + _table(
                [
                    "date column",
                    "metric",
                    "granularity",
                    "periods",
                    "start",
                    "end",
                    "trend",
                    "missing",
                ],
                temporal_rows,
            )
            + "</section>"
        )
    else:
        temporal_section = ""

    all_insights = (
        report.quality_insights
        + report.relationship_insights
        + report.distribution_insights
        + report.pattern_insights
        + report.structure_insights
        + report.aggregation_insights
        + report.comparative_insights
        + report.temporal_insights
        + report.inference_insights
    )

    # Orden temático: los hallazgos de la misma columna quedan
    # juntos (el usuario navega por hallazgo, no por familia) y
    # dentro de cada columna los más fuertes primero.
    all_insights = sorted(
        all_insights,
        key=lambda insight: (
            insight.columns[0].lower() if insight.columns else "",
            -insight.score,
        ),
    )

    figures = _figures_by_chart_id(report)
    consumed: set[str] = set()

    missing_cells = sum(column.n_missing for column in profile.columns)

    generated_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M")

    executive_compact = _insights_list(
        report.executive_summary,
        "No findings to summarize.",
    )

    insight_pages = _insight_pages(all_insights, figures, consumed)

    extra_charts = _extra_charts_section(report, consumed, hide_charts)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Explain My Data — Report</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>{_STYLESHEET}</style>
<script>{_AUTO_PRINT_SCRIPT}</script>
</head>
<body>
<main>

<section class="page cover">
<h1>Explain My Data</h1>
<p class="cover-subtitle">Heuristic dataset analysis — no ML, no LLM.</p>
<div class="cover-stats">
<div class="cover-stat"><strong>{profile.n_rows}</strong><span>rows</span></div>
<div class="cover-stat"><strong>{profile.n_columns}</strong><span>columns</span></div>
<div class="cover-stat"><strong>{profile.n_duplicates}</strong><span>duplicates</span></div>
<div class="cover-stat"><strong>{missing_cells}</strong><span>missing cells</span></div>
<div class="cover-stat"><strong>{len(all_insights)}</strong><span>insights</span></div>
</div>
<p class="meta">Generated {generated_at}</p>
</section>

<section class="page">
<h2>Overview</h2>
<div class="kpis">
{_kpi("rows", profile.n_rows)}
{_kpi("columns", profile.n_columns)}
{_kpi("duplicate rows", profile.n_duplicates)}
{_kpi("missing cells", missing_cells)}
</div>
<h3>Column profile</h3>
{_table(["column", "dtype", "role", "missing", "unique"], profile_rows)}
</section>

<section class="page">
<h2>Quality</h2>
{_table(
    ["column", "missing", "outliers %", "extreme", "type mismatch"],
    quality_rows,
)}
{_quality_findings_section(report)}
{_cleaning_recommendations_section(report)}
</section>

<section class="page">
<h2>Patterns</h2>
<p class="section-sub">Technical observations detected in the data — statistical facts, not conclusions.</p>
{_patterns_section(report)}
</section>

<section class="page">
<h2>Executive summary</h2>
{executive_compact}
</section>

{insight_pages}

{extra_charts}

<section class="page">
<h2>Distributions</h2>
{_table(
    ["column", "n", "skewness", "kurtosis", "modality", "top 5% share"],
    distribution_rows,
)}
</section>

<section class="page">
<h2>Numeric statistics</h2>
{_table(
    ["column", "n", "mean", "median", "std", "min", "max", "skewness"],
    numeric_rows,
)}
<h2 style="margin-top:24px">Categorical statistics</h2>
{_table(
    ["column", "n", "unique", "dominant", "dominant %"],
    categorical_rows,
)}
</section>

<section class="page">
<h2>Relationships: numeric ↔ numeric</h2>
{_table(["x", "y", "pearson", "spearman", "n"], numeric_rel_rows)}
<h2 style="margin-top:24px">Relationships: categorical ↔ numeric</h2>
{_table(
    ["categorical", "numeric", "separation", "groups", "n"],
    cat_num_rel_rows,
)}
<h2 style="margin-top:24px">Relationships: categorical ↔ categorical</h2>
{_table(["x", "y", "cramers_v", "chi2", "p-value"], cat_cat_rel_rows)}
</section>

{temporal_section}

</main>
</body>
</html>
"""
