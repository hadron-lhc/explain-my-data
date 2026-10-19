from __future__ import annotations

import io
import json
import os
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from src.analysis import analyze_dataset
from src.cleaning import CleaningConfig, clean_dataset
from src.profiling import profile_dataset
from src.quality import quality_dataset
from src.reporting import render_html_report

# Límite de upload (cloud-readiness D0): el contenedor recibe CSVs
# pequeños (<5MB de trabajo); 20MB es el techo con holgura. Evita
# que un upload gigante agote la memoria del aislante.
MAX_UPLOAD_BYTES = int(os.environ.get("EMD_MAX_UPLOAD_BYTES", str(20 * 1024 * 1024)))

# Carpeta de la SPA. En el contenedor UI y API comparten origen, lo
# que permite que api.js use URLs relativas.
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(
    title="Explain My Data API",
    description="API for automated exploratory data analysis.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def read_csv_upload(file: UploadFile) -> pd.DataFrame:
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file provided.",
        )

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported.",
        )

    # Lectura acotada: nunca más de MAX_UPLOAD_BYTES en memoria.
    chunks: list[bytes] = []
    total = 0

    while chunk := file.file.read(1024 * 1024):
        total += len(chunk)

        if total > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)}MB "
                    "upload limit."
                ),
            )

        chunks.append(chunk)

    try:
        return pd.read_csv(io.BytesIO(b"".join(chunks)))
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read CSV file: {exc}",
        ) from exc


def parse_numeric_ranges(
    numeric_ranges: str,
) -> dict[str, tuple[float | None, float | None]]:
    if not numeric_ranges.strip():
        return {}

    try:
        parsed = json.loads(numeric_ranges)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid numeric_ranges JSON: {exc.msg}") from exc

    if not isinstance(parsed, dict):
        raise ValueError("numeric_ranges must be a JSON object.")

    ranges: dict[str, tuple[float | None, float | None]] = {}

    for column_name, bounds in parsed.items():
        if not isinstance(column_name, str):
            raise ValueError("Numeric range column names must be strings.")

        if not isinstance(bounds, list) or len(bounds) != 2:
            raise ValueError(
                f"Numeric range for '{column_name}' must be "
                "an array with exactly two values: [min, max]."
            )

        lower, upper = bounds

        if lower is not None and not isinstance(
            lower,
            (int, float),
        ):
            raise ValueError(
                f"Minimum value for '{column_name}' must be a number or null."
            )

        if upper is not None and not isinstance(
            upper,
            (int, float),
        ):
            raise ValueError(
                f"Maximum value for '{column_name}' must be a number or null."
            )

        ranges[column_name] = (
            float(lower) if lower is not None else None,
            float(upper) if upper is not None else None,
        )

    return ranges


@app.post("/analyze")
def analyze_csv(file: UploadFile = File(...)):
    df = read_csv_upload(file)

    try:
        report = analyze_dataset(df)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {exc}",
        ) from exc

    return report


@app.post("/report")
def report_csv(
    file: UploadFile = File(...),
    hide: str = Query(
        default="",
        description="Insight ids ocultos en la UI, separados por coma.",
    ),
    hide_charts: str = Query(
        default="",
        description=(
            "Identidades de charts ocultos en la UI "
            "(visualizationIdentity), separadas por coma."
        ),
    ),
):
    """Genera un informe HTML autocontenido listo para descargar."""
    df = read_csv_upload(file)
    hide_ids = {item.strip() for item in hide.split(",") if item.strip()}
    hidden_charts = {
        item.strip() for item in hide_charts.split(",") if item.strip()
    }

    try:
        report = analyze_dataset(df)
        html_content = render_html_report(
            report,
            hide_ids=hide_ids,
            hide_charts=hidden_charts,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Report generation failed: {exc}",
        ) from exc

    return HTMLResponse(
        content=html_content,
        headers={
            "Content-Disposition": 'attachment; filename="explain-my-data-report.html"',
        },
    )


@app.post("/clean")
def clean_csv(
    file: UploadFile = File(...),
    remove_duplicates: bool = Form(True),
    missing_numeric_strategy: str = Form("median"),
    missing_categorical_strategy: str = Form("mode"),
    replace_invalid_values: bool = Form(True),
    convert_type_mismatches: bool = Form(False),
    outlier_strategy: str = Form("keep"),
    outlier_columns: str = Form(""),
    numeric_ranges: str = Form(""),
    format: str = Query(
        default="json",
        description=(
            "json: informe + preview de filas (UI). "
            "csv: descarga del dataset limpio."
        ),
    ),
):
    df = read_csv_upload(file)

    if format not in ("json", "csv"):
        raise HTTPException(
            status_code=400,
            detail="format must be 'json' or 'csv'.",
        )

    try:
        profile = profile_dataset(df)
        quality = quality_dataset(df, profile)

        parsed_numeric_ranges = parse_numeric_ranges(
            numeric_ranges,
        )

        outlier_column_set = {
            column.strip() for column in outlier_columns.split(",") if column.strip()
        }

        unknown_outlier_columns = outlier_column_set - set(df.columns)

        if unknown_outlier_columns:
            raise ValueError(
                "Unknown columns selected for outlier removal: "
                f"{sorted(unknown_outlier_columns)}"
            )

        config = CleaningConfig(
            remove_duplicates=remove_duplicates,
            missing_numeric_strategy=missing_numeric_strategy,
            missing_categorical_strategy=missing_categorical_strategy,
            replace_invalid_values=replace_invalid_values,
            convert_type_mismatches=convert_type_mismatches,
            outlier_strategy=outlier_strategy,
            outlier_columns=outlier_column_set,
            numeric_ranges=parsed_numeric_ranges,
        )

        cleaned_df, cleaning_report = clean_dataset(
            df,
            profile,
            quality,
            config,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Cleaning failed: {exc}",
        ) from exc

    if format == "csv":
        # Descarga del CSV limpio sin pasar por el preview JSON
        # (que materializa todas las filas como dicts).
        buffer = io.StringIO()
        cleaned_df.to_csv(buffer, index=False)

        return StreamingResponse(
            iter([buffer.getvalue()]),
            media_type="text/csv",
            headers={
                "Content-Disposition": (
                    'attachment; filename="cleaned-data.csv"'
                ),
            },
        )

    try:
        cleaned_analysis = analyze_dataset(cleaned_df)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Cleaning failed: {exc}",
        ) from exc

    return {
        "report": cleaning_report,
        "analysis": cleaned_analysis,
        "data": (
            cleaned_df.astype(object)
            .where(cleaned_df.notna(), None)
            .to_dict(orient="records")
        ),
    }


# SPA montada al final (las rutas de API registradas antes tienen
# prioridad). En el contenedor esto sirve UI y API en un solo
# origen; para dev con http.server en otro puerto ver el script de
# config en index.html.
if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
