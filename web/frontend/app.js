const fileInput = document.getElementById("file-input");
const analyzeButton = document.getElementById("analyze-button");
const status = document.getElementById("status");
const results = document.getElementById("results");

const overview = document.getElementById("overview");
const quality = document.getElementById("quality");
const charts = document.getElementById("charts");
const rawReport = document.getElementById("raw-report");

analyzeButton.addEventListener("click", analyzeDataset);

async function analyzeDataset() {
  const file = fileInput.files[0];

  if (!file) {
    status.textContent = "Please select a CSV file.";
    return;
  }

  status.textContent = "Analyzing dataset...";
  analyzeButton.disabled = true;

  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch("http://127.0.0.1:8000/analyze", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Analysis failed.");
    }

    const report = await response.json();

    renderReport(report);

    status.textContent = "Analysis completed.";
    results.classList.remove("hidden");
  } catch (error) {
    status.textContent = `Error: ${error.message}`;
  } finally {
    analyzeButton.disabled = false;
  }
}

function renderReport(report) {
  renderOverview(report);
  renderQuality(report);
  renderCharts(report);

  rawReport.textContent = JSON.stringify(report, null, 2);
}

function renderOverview(report) {
  overview.innerHTML = `
        <p>
            Rows: ${report.profile.n_rows}
        </p>

        <p>
            Columns: ${report.profile.n_columns}
        </p>

        <p>
            Duplicates: ${report.profile.n_duplicates}
        </p>
    `;
}

function renderQuality(report) {
  quality.innerHTML = `
        <p>
            Duplicate rows:
            ${report.quality.n_duplicates}
        </p>

        <p>
            Columns analyzed:
            ${report.quality.columns.length}
        </p>
    `;
}

function renderCharts(report) {
  charts.innerHTML = "";

  for (const candidate of report.analysis_candidates) {
    const container = document.createElement("div");

    container.className = "chart-container";

    charts.appendChild(container);

    renderCandidateChart(container, candidate, report);
  }
}

function renderCandidateChart(container, candidate, report) {
  if (candidate.relationship_type === "numeric_numeric") {
    renderNumericNumericChart(container, candidate, report);

    return;
  }

  if (candidate.relationship_type === "categorical_numeric") {
    renderCategoricalNumericChart(container, candidate, report);
  }
}

function renderNumericNumericChart(container, candidate, report) {
  // Temporary implementation.
  // The actual visualization data will
  // come from the backend later.

  container.innerHTML = `
        <h3>
            ${candidate.column_x}
            vs
            ${candidate.column_y}
        </h3>

        <p>
            Recommended visualization:
            ${candidate.recommended_visualizations.join(", ")}
        </p>

        <p>
            Relevance:
            ${candidate.relevance_score.toFixed(2)}
        </p>
    `;
}

function renderCategoricalNumericChart(container, candidate, report) {
  container.innerHTML = `
        <h3>
            ${candidate.column_x}
            vs
            ${candidate.column_y}
        </h3>

        <p>
            Recommended visualization:
            ${candidate.recommended_visualizations.join(", ")}
        </p>

        <p>
            Relevance:
            ${candidate.relevance_score.toFixed(2)}
        </p>
    `;
}
