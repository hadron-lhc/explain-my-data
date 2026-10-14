import io

import pandas as pd
from fastapi.testclient import TestClient

from web.backend.main import app

client = TestClient(app)


def _csv_upload(df: pd.DataFrame, filename: str = "data.csv"):
    return {"file": (filename, io.BytesIO(df.to_csv(index=False).encode()), "text/csv")}


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "country": ["AR", "CL"] * 10,
            "revenue": [float(i) for i in range(20)],
        }
    )


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_analyze_returns_report():
    response = client.post("/analyze", files=_csv_upload(_sample_df()))

    assert response.status_code == 200

    payload = response.json()

    assert payload["profile"]["n_rows"] == 20
    assert payload["profile"]["n_columns"] == 2
    assert payload["quality"]["n_duplicates"] == 0


def test_analyze_rejects_non_csv_extension():
    response = client.post(
        "/analyze",
        files={"file": ("data.xlsx", io.BytesIO(b"raw"), "application/octet-stream")},
    )

    assert response.status_code == 400
    assert "CSV" in response.json()["detail"]


def test_analyze_rejects_unreadable_csv():
    response = client.post(
        "/analyze",
        files={"file": ("data.csv", io.BytesIO(b"\xff\xfe\xfdinvalid"), "text/csv")},
    )

    assert response.status_code == 400


def test_clean_returns_report_analysis_and_data():
    response = client.post("/clean", files=_csv_upload(_sample_df()))

    assert response.status_code == 200

    payload = response.json()

    assert set(payload) == {"report", "analysis", "data"}
    assert len(payload["data"]) == 20
    assert payload["analysis"]["profile"]["n_rows"] == 20


def test_clean_rejects_unknown_outlier_column():
    response = client.post(
        "/clean",
        data={"outlier_columns": "no_such_column"},
        files=_csv_upload(_sample_df()),
    )

    assert response.status_code == 400
    assert "no_such_column" in response.json()["detail"]


def test_clean_rejects_invalid_numeric_ranges_json():
    response = client.post(
        "/clean",
        data={"numeric_ranges": "{not json"},
        files=_csv_upload(_sample_df()),
    )

    assert response.status_code == 400
    assert "numeric_ranges" in response.json()["detail"]


def test_clean_rejects_invalid_numeric_range_shape():
    response = client.post(
        "/clean",
        data={"numeric_ranges": '{"revenue": [1, 2, 3]}'},
        files=_csv_upload(_sample_df()),
    )

    assert response.status_code == 400


def test_clean_rejects_unknown_missing_strategy():
    response = client.post(
        "/clean",
        data={"missing_numeric_strategy": "drop"},
        files=_csv_upload(_sample_df()),
    )

    assert response.status_code == 400


def test_report_endpoint_returns_html():
    response = client.post("/report", files=_csv_upload(_sample_df()))

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "attachment" in response.headers["content-disposition"]

    content = response.text

    assert "<!DOCTYPE html>" in content
    assert "Executive summary" in content


def test_report_endpoint_rejects_non_csv():
    response = client.post(
        "/report",
        files={"file": ("data.xlsx", io.BytesIO(b"raw"), "application/octet-stream")},
    )

    assert response.status_code == 400


def test_report_endpoint_hides_insight_ids():
    df = pd.DataFrame(
        {
            "country": ["A"] * 40 + ["B"] * 40,
            "revenue": [10.0] * 40 + [200.0] * 40,
        }
    )

    analysis = client.post("/analyze", files=_csv_upload(df))
    assert analysis.status_code == 200

    executive = analysis.json()["executive_summary"]
    assert executive

    hidden = executive[0]

    response = client.post(
        f"/report?hide={hidden['id']}",
        files=_csv_upload(df),
    )

    assert response.status_code == 200
    assert hidden["title"] not in response.text

    for other in executive[1:]:
        assert other["title"] in response.text


def test_report_endpoint_hides_extra_charts():
    df = pd.DataFrame(
        {
            "country": ["AR", "CL"] * 30,
            "revenue": [float(i % 5) * 20 + 50 for i in range(60)],
            "units": [1 + (i % 4) for i in range(60)],
        }
    )

    baseline = client.post("/report", files=_csv_upload(df))
    assert baseline.status_code == 200
    assert "Distribution of revenue" in baseline.text
    assert "Distribution of units" in baseline.text

    response = client.post(
        "/report?hide_charts=histogram:revenue",
        files=_csv_upload(df),
    )

    assert response.status_code == 200
    assert "Distribution of revenue" not in response.text
    assert "Distribution of units" in response.text
