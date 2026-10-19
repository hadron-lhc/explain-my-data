"""Tests de la API web (D0: cloud-readiness).

Cubre health, SPA estática, límite de upload, /clean?format=csv y
errores de formato.
"""

import io

import pandas as pd
from fastapi.testclient import TestClient

import web.backend.main as backend
from web.backend.main import app

client = TestClient(app)

TINY_CSV = b"a,b,label\n1,2,x\n3,4,y\n5,6,x\n"


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_spa_served_at_root():
    response = client.get("/")

    assert response.status_code == 200
    assert "Explain My Data" in response.text


def test_analyze_small_csv():
    response = client.post(
        "/analyze",
        files={"file": ("tiny.csv", TINY_CSV, "text/csv")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["profile"]["n_rows"] == 3


def test_report_returns_html_download():
    response = client.post(
        "/report",
        files={"file": ("tiny.csv", TINY_CSV, "text/csv")},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "attachment" in response.headers["content-disposition"]


def test_upload_over_limit_rejected(monkeypatch):
    monkeypatch.setattr(backend, "MAX_UPLOAD_BYTES", 10)

    response = client.post(
        "/analyze",
        files={"file": ("tiny.csv", TINY_CSV, "text/csv")},
    )

    assert response.status_code == 413


def test_non_csv_rejected():
    response = client.post(
        "/analyze",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 400


def test_clean_csv_download_format():
    response = client.post(
        "/clean?format=csv",
        files={"file": ("tiny.csv", TINY_CSV, "text/csv")},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert "cleaned-data.csv" in response.headers["content-disposition"]

    parsed = pd.read_csv(io.StringIO(response.text))
    assert list(parsed.columns) == ["a", "b", "label"]


def test_clean_invalid_format_rejected():
    response = client.post(
        "/clean?format=xml",
        files={"file": ("tiny.csv", TINY_CSV, "text/csv")},
    )

    assert response.status_code == 400


def test_clean_json_default_returns_preview():
    response = client.post(
        "/clean",
        files={"file": ("tiny.csv", TINY_CSV, "text/csv")},
    )

    assert response.status_code == 200
    body = response.json()
    assert "report" in body
    assert "analysis" in body
    assert len(body["data"]) == 3
