from __future__ import annotations

import io

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.analysis import analyze_dataset


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
    """Check whether the API is running."""

    return {"status": "ok"}


@app.post("/analyze")
def analyze_csv(file: UploadFile = File(...)):
    """Analyze an uploaded CSV file."""

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

    try:
        contents = file.file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read CSV file: {exc}",
        ) from exc

    try:
        report = analyze_dataset(df)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {exc}",
        ) from exc

    return report
