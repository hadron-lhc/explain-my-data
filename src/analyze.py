"""
analyze.py

Interfaz de línea de comandos para Explain My Data.

Uso:

    uv run python -m src.analyze data/messy_sales.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.analysis import AnalysisReport, analyze_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analiza un dataset CSV y genera un diagnóstico exploratorio inicial."
        )
    )

    parser.add_argument(
        "csv_path",
        type=Path,
        help="Ruta al archivo CSV que se desea analizar.",
    )

    return parser.parse_args()


def print_header(title: str) -> None:
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def print_dataset_summary(report: AnalysisReport) -> None:
    profile = report.profile

    print_header("DATASET")

    print(f"Rows:             {profile.n_rows}")
    print(f"Columns:          {profile.n_columns}")
    print(f"Duplicate rows:   {profile.n_duplicates}")


def print_profile(report: AnalysisReport) -> None:
    profile = report.profile

    print_header("PROFILING")

    print(f"{'Column':<20}{'Role':<15}{'Missing':>10}{'Unique':>10}")

    print("-" * 55)

    for column in profile.columns:
        print(
            f"{column.name:<20}"
            f"{column.role:<15}"
            f"{column.pct_missing:>9.2f}%"
            f"{column.pct_unique:>9.2f}%"
        )

    notes = [column for column in profile.columns if column.notes]

    if notes:
        print()
        print("Notes:")

        for column in notes:
            print(f"  {column.name}:")

            for note in column.notes:
                print(f"    - {note}")


def print_quality(report: AnalysisReport) -> None:
    quality = report.quality

    print_header("QUALITY")

    print(
        f"Duplicate rows: {quality.n_duplicates} ({quality.duplicate_percentage:.2f}%)"
    )

    type_issues = [column for column in quality.columns if column.is_type_mismatch]

    if type_issues:
        print()
        print("Type issues:")

        for column in type_issues:
            print(f"  {column.column_name}:")
            print(f"    mismatches: {column.type_mismatch_count}")
            print(f"    conversion rate: {column.conversion_rate * 100:.2f}%")

    outlier_columns = [column for column in quality.columns if column.n_outliers > 0]

    if outlier_columns:
        print()
        print("Potential outliers:")

        for column in outlier_columns:
            print(
                f"  {column.column_name}: "
                f"{column.n_outliers} "
                f"({column.outlier_percentage:.2f}%)"
            )

    missing_columns = [column for column in quality.columns if column.missing_count > 0]

    if missing_columns:
        print()
        print("Missing values:")

        for column in missing_columns:
            print(
                f"  {column.column_name}: "
                f"{column.missing_count} "
                f"({column.missing_percentage:.2f}%)"
            )


def print_statistics(report: AnalysisReport) -> None:
    print_header("STATISTICS")

    if report.numeric_statistics:
        print()
        print("NUMERIC VARIABLES")
        print("-" * 60)

        for stats in report.numeric_statistics:
            print()
            print(stats.column_name)
            print(f"  count:     {stats.count}")
            print(f"  mean:      {stats.mean:.2f}")
            print(f"  median:    {stats.median:.2f}")
            print(f"  std:       {stats.std:.2f}")
            print(f"  min:       {stats.min:.2f}")
            print(f"  q1:        {stats.q1:.2f}")
            print(f"  q3:        {stats.q3:.2f}")
            print(f"  max:       {stats.max:.2f}")
            print(f"  skewness:  {stats.skewness:.2f}")

    if report.categorical_statistics:
        print()
        print("CATEGORICAL VARIABLES")
        print("-" * 60)

        for stats in report.categorical_statistics:
            print()
            print(stats.column_name)
            print(f"  count:              {stats.count}")
            print(f"  unique:             {stats.n_unique}")
            print(f"  dominant value:     {stats.dominant_value}")
            print(f"  dominant percentage: {stats.dominant_percentage:.2f}%")

            if stats.top_values:
                print("  top values:")

                for value, count in stats.top_values:
                    print(f"    {value}: {count}")


def print_relationships(report: AnalysisReport) -> None:
    print_header("RELATIONSHIPS")

    if not report.numeric_relationships:
        print("No numeric relationships found.")
        return

    print("NUMERIC ↔ NUMERIC")
    print("-" * 60)

    for relationship in report.numeric_relationships:
        print()
        print(f"{relationship.column_x} <-> {relationship.column_y}")
        print(f"  Pearson:   {relationship.pearson:.3f}")
        print(f"  Spearman:  {relationship.spearman:.3f}")
        print(f"  n:         {relationship.n_observations}")


def print_report(report: AnalysisReport) -> None:
    print()
    print("=" * 60)
    print("EXPLAIN MY DATA")
    print("=" * 60)

    print_dataset_summary(report)
    print_profile(report)
    print_quality(report)
    print_statistics(report)
    print_relationships(report)

    print()
    print("=" * 60)
    print("Analysis complete.")
    print("=" * 60)


def main() -> None:
    args = parse_args()

    if not args.csv_path.exists():
        raise SystemExit(f"Error: file not found: {args.csv_path}")

    if args.csv_path.suffix.lower() != ".csv":
        raise SystemExit("Error: the input file must be a CSV.")

    try:
        df = pd.read_csv(args.csv_path)
    except Exception as exc:
        raise SystemExit(f"Error reading CSV: {exc}") from exc

    if df.empty:
        raise SystemExit("Error: the dataset is empty.")

    report = analyze_dataset(df)

    print_report(report)


if __name__ == "__main__":
    main()
