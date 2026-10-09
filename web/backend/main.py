from __future__ import annotations

import io
import json

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.analysis import analyze_dataset
from src.cleaning import CleaningConfig, clean_dataset
from src.profiling import profile_dataset
from src.quality import quality_dataset


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

    try:
        contents = file.file.read()
        return pd.read_csv(io.BytesIO(contents))
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
):
    df = read_csv_upload(file)

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

        cleaned_analysis = analyze_dataset(cleaned_df)

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

    return {
        "report": cleaning_report,
        "analysis": cleaned_analysis,
        "data": (
            cleaned_df.astype(object)
            .where(cleaned_df.notna(), None)
            .to_dict(orient="records")
        ),
    }
