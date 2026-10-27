# Explain My Data

**Live demo:** <https://explainmydata.help>

A lightweight CSV dataset analyzer that automatically detects patterns, generates insights, and creates visualizations. **No ML or LLMs** — all analysis is statistical, and conclusions are generated from deterministic templates.

Zero tokens, zero cost.

## What It Does

Given a CSV file, the pipeline:

1. **Profiles the dataset** — identifies column roles: numeric, categorical, boolean, date, and identifier.

2. **Detects data quality issues** — missing values, duplicates, invalid values, outliers (IQR), and failed type conversions.

3. **Analyzes relationships** — numeric ↔ numeric (Pearson/Spearman), categorical ↔ numeric (group separation), and categorical ↔ categorical (chi-squared + Cramér's V).

4. **Describes distributions** — histograms (Freedman–Diaconis), skewness, kurtosis, top-5% concentration, and modality.

5. **Finds patterns** — groups with substantially different means (≥1σ), column-level outliers, constant columns, and dominant aggregates.

6. **Generates insights** in English, backed by numerical evidence: R² for correlations; extreme-value ratios and sample sizes for group differences; five-number summaries for skewed distributions; prominent combinations for categorical associations; and "Related:" cross-links between insight cards that share a strong relationship. It also produces a **narrative executive summary**, with dataset context and one statement per insight family, referencing the corresponding insight cards. A per-family budget keeps the report focused rather than turning it into an overwhelming collection of cards.

7. **Generates comparative insights using explicit rules**, including group rankings by total with volume-versus-value comparisons, driver decomposition (does volume or value explain the result?), consistency across metrics ("the same group leads on two measures"), categorical distributions without a dominant value, and null findings (nearly identical binary proportions, weak correlations, and independent pairs), with exact numerical evidence in each message.

8. **Automatically generates visualizations** — scatter plots with trend lines, sorted bar charts, sampled box plots, histograms, and contingency heatmaps.

## Installation

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv sync
```

## Usage

### CLI

```bash
uv run python -m src.analyze data/messy_sales.csv
```

Prints the dataset profile, quality report, descriptive statistics, distributions, relationships, analysis candidates, insights, and executive summary.

### API and Web UI

```bash
uv run uvicorn web.backend.main:app --port 8000
```

The backend serves the single-page application from the same origin. Open `http://localhost:8000` — no additional setup required.

Available endpoints:

- `POST /analyze` — analyzes an uploaded dataset.
- `POST /clean` — returns a JSON preview of the cleaned dataset. Use `?format=csv` to download the cleaned dataset as a CSV file.
- `POST /report` — generates a self-contained HTML report ready for printing. Use `?hide=id1,id2` to exclude insights hidden in the UI, or `?hide_charts=histogram:col,bar:a:b` to exclude charts from "Additional charts".
- `GET /health` — health check.

The upload limit is 20 MB and can be changed with `EMD_MAX_UPLOAD_BYTES`. Typical working CSV files are expected to be under 5 MB.

The `/report` endpoint is designed for **print-to-PDF**. Each section occupies a separate page: cover, overview, data quality, patterns, executive summary, one insight per page with its charts, and a technical appendix. It uses `@media print` CSS and automatically opens the print dialog when loaded with the `#print` hash.

The UI is a static single-page application in `web/frontend/`, using Plotly via CDN. It defaults to same-origin requests and works when served by the backend.

To develop with the frontend on a separate port:

```bash
python -m http.server 5500 --directory web/frontend
```

Then open `http://localhost:5500`. This port automatically enables the development configuration, which points API requests to `http://127.0.0.1:8000` (see the script in `index.html`).

The upload area supports drag-and-drop or file selection. Analysis starts automatically once a file is dropped or selected.

The UI workflow is:

1. **Overview** — KPIs and column profiles.
2. **Data quality** — detected quality issues.
3. **Cleaning** — a drawer with detected issues, missing values, duplicates, outliers, and range problems, with undo support.
4. **Patterns** — detected patterns in the data.
5. **Insights** — a highlighted executive summary, followed by individual insights with explanations, associated patterns, and charts.
6. **Additional charts** — automatically generated visualizations.
7. **Downloads** — PDF (through the browser's print dialog), HTML report, and cleaned CSV.

Each insight card and each chart under "Additional charts" has an **✕** button to hide it. Hidden items are stored in `localStorage`, keyed by the file's SHA-256 fingerprint, and can be restored using "Restore all".

Downloaded PDF/HTML reports respect these choices through `/report?hide=...` and `/report?hide_charts=...`, using the UI's `visualizationIdentity` identifiers (e.g., `histogram:revenue` or `bar:country:revenue`). This keeps the printed report consistent with the on-screen view.

### Deployment (Cloudflare)

The backend and UI are deployed as a **Cloudflare Worker + Container**, with scale-to-zero behavior and a single origin.

See [`docs/DEPLOY.md`](docs/DEPLOY.md) for the complete deployment guide. To deploy:

```bash
npm install && npx wrangler deploy
```

### Tests and Linting

```bash
uv run pytest -q
uv run ruff check .
```

## Threshold Configuration

All analysis thresholds — including relationship-selection percentiles, IQR multipliers, chart-sampling limits, and executive-summary size — are centralized in `src/config.py` and imported by the relevant modules.

## Project Structure

```text
src/
  analyze.py                 CLI
  analysis.py                Pipeline orchestrator
  profiling.py               Column roles and profiling
  quality.py                 Column-level quality issues
  statistics.py              Descriptive statistics
  distributions.py           Histograms and univariate distributions
  relationships.py           Numeric-numeric, categorical-numeric, categorical-categorical
  aggregations.py             Group-level metrics
  patterns.py                 Group differences and outliers
  planner.py                  What to analyze and how to visualize it
  insights.py                 Insight templates and executive summary
  comparative_insights.py     Rankings, drivers, and null findings
  curation.py                 Insight deduplication and caps (one finding = one card)
  temporal.py                 Trends, seasonality, gaps, regimes, extremes
  visualizations.py           Chart builders
  reporting.py                HTML reports with embedded Plotly charts
  cleaning.py                 Data-cleaning transformations
  preparation.py              Analysis preparation
  config.py                   Centralized thresholds
  utils_stats.py              Shared IQR utilities
web/
  backend/                    FastAPI
  frontend/                   UI (vanilla JavaScript + Plotly)
tests/                         386 tests
docs/ROADMAP.md                Full specification
docs/PLAN_V2.md                Completed improvement plan
docs/PLAN_V3.md                Ongoing insight-quality improvement plan
docs/DEPLOY.md                 Cloudflare deployment (Worker + Container)
```

## Design Principles

- **No ML/LLMs** — only classical statistics and explicit rules.
- **Lightweight dependencies** — pandas, NumPy, SciPy, Plotly, and FastAPI.
- **Deterministic** — the same data produces the same insights, with fixed `random_state` values for sampling.
- **Non-destructive cleaning** — cleaning operations return copies rather than modifying the original data.

## Status

**V2 is complete**, covering the analysis pipeline, comparative insights, insight curation, temporal analysis, and ML-based recommendations. **V3 is in progress**, focusing on folding redundant findings (equivalent and derived relationships), suppressing trivial results, contextual validity (missingness and practical significance), per-family insight budgets, a narrative executive summary, and richer insight cards (R², extreme-value ratios, five-number summaries, and cross-links). The web application is also cloud-ready, with same-origin requests, configurable upload limits, and CSV downloads through `/clean?format=csv`.

The project currently has **386 tests passing and a clean Ruff lint check**.

Relationship selection includes explicit overrides — a minimum threshold for strong pairs (≥0.9) and top-N selection for otherwise underrepresented columns. This addresses a limitation where relying exclusively on the 75th percentile could exclude well-established relationships in datasets with redundant columns.

These changes were validated against a local QA playground (`explain-my-data-qa`) containing 16 classic and synthetic datasets with known ground truth. Validation results: **100/100 required findings detected, 59/59 claims validated, and 100% measured precision**.

See [`docs/PLAN_V3.md`](docs/PLAN_V3.md) for the ongoing insight-quality improvements, [`docs/PLAN_V2.md`](docs/PLAN_V2.md) for the completed improvements, and [`docs/ROADMAP.md`](docs/ROADMAP.md) for the long-term vision.
