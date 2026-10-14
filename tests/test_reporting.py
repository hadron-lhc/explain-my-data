import pandas as pd

from src.analysis import analyze_dataset
from src.reporting import _figure_chunks, render_html_report


def _sample_report():
    df = pd.DataFrame(
        {
            "country": ["AR", "CL"] * 30,
            "revenue": [float(i % 5) * 20 + 50 for i in range(60)],
            "units": [1 + (i % 4) for i in range(60)],
        }
    )

    return analyze_dataset(df)


def test_render_html_report_contains_key_sections():
    report = _sample_report()

    html_content = render_html_report(report)

    assert html_content.startswith("<!DOCTYPE html>")
    assert "Explain My Data" in html_content
    assert "Executive summary" in html_content
    assert "Column profile" in html_content
    assert "Numeric statistics" in html_content
    assert "Relationships" in html_content


def test_render_html_report_escapes_user_data():
    df = pd.DataFrame(
        {
            "city": ["<script>alert(1)</script>"] * 30 + ["ok"] * 30,
            "revenue": [float(i) for i in range(60)],
        }
    )

    report = analyze_dataset(df)
    html_content = render_html_report(report)

    assert "<script>alert(1)</script>" not in html_content
    assert "&lt;script&gt;" in html_content


def test_render_html_report_includes_insights():
    df = pd.DataFrame(
        {
            "country": ["A"] * 40 + ["B"] * 40,
            "revenue": [10.0] * 40 + [200.0] * 40,
        }
    )

    report = analyze_dataset(df)
    html_content = render_html_report(report)

    assert report.executive_summary
    assert "country" in html_content


def test_render_html_report_handles_minimal_dataset():
    df = pd.DataFrame({"x": [float(i) for i in range(15)]})

    report = analyze_dataset(df)
    html_content = render_html_report(report)

    assert "<!DOCTYPE html>" in html_content
    assert "No insights detected." in html_content or "Executive summary" in html_content


def test_render_html_report_includes_heatmap():
    df = pd.DataFrame(
        {
            "size": ["S", "L"] * 50,
            "color": ["red", "blue"] * 50,
        }
    )

    report = analyze_dataset(df)
    html_content = render_html_report(report)

    assert report.heatmaps
    assert "heatmap" in html_content


def test_render_html_report_has_cover_and_print_setup():
    report = _sample_report()

    html_content = render_html_report(report)

    assert 'class="page cover"' in html_content
    assert "Heuristic dataset analysis" in html_content
    assert "@page" in html_content
    assert "@media print" in html_content
    assert "page-break-after: always" in html_content
    assert 'window.location.hash !== "#print"' in html_content
    assert "window.print()" in html_content


def test_render_html_report_includes_patterns_section():
    report = _sample_report()

    html_content = render_html_report(report)

    assert "<h2>Patterns</h2>" in html_content

    if report.group_difference_patterns:
        assert "Group differences" in html_content

    if any(pattern.outlier_count > 0 for pattern in report.numeric_outlier_patterns):
        assert "Numeric outliers" in html_content


def test_render_html_report_renders_insights_as_pages_with_charts():
    df = pd.DataFrame(
        {
            "country": ["A"] * 40 + ["B"] * 40,
            "revenue": [10.0] * 40 + [200.0] * 40,
        }
    )

    report = analyze_dataset(df)
    html_content = render_html_report(report)

    assert report.executive_summary
    assert 'class="page insight-page"' in html_content
    assert "<div class=\"chart\">" in html_content
    assert "Insight 1" in html_content


def test_render_html_report_filters_hidden_insights():
    report = _sample_report()

    assert report.executive_summary

    hidden = report.executive_summary[0]
    hidden_html = render_html_report(report, hide_ids={hidden.id})

    assert hidden.title not in hidden_html
    assert hidden.message not in hidden_html

    for other in report.executive_summary[1:]:
        assert other.title in hidden_html


def test_render_html_report_hide_without_ids_is_noop():
    report = _sample_report()

    baseline = render_html_report(report)
    with_empty = render_html_report(report, hide_ids=set())
    with_none = render_html_report(report, hide_ids=None)

    for insight in report.executive_summary:
        assert insight.title in baseline
        assert insight.title in with_empty
        assert insight.title in with_none


def test_render_html_report_hides_extra_charts_by_identity():
    report = _sample_report()

    baseline = render_html_report(report)
    assert "Distribution of revenue" in baseline
    assert "Distribution of units" in baseline

    hidden = render_html_report(report, hide_charts={"histogram:revenue"})

    assert "Distribution of revenue" not in hidden
    assert "Distribution of units" in hidden


def test_render_html_report_hide_charts_empty_state():
    report = _sample_report()

    everything = {
        identity for _key, identity, _chunk in _figure_chunks(report)
    }
    html = render_html_report(report, hide_charts=everything)

    assert "All extra charts are hidden" in html


def test_render_html_report_hide_charts_without_ids_is_noop():
    report = _sample_report()

    for content in (
        render_html_report(report),
        render_html_report(report, hide_charts=set()),
        render_html_report(report, hide_charts=None),
    ):
        assert "Distribution of revenue" in content
        assert "Distribution of units" in content
