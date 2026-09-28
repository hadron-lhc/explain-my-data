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

  for (const visualization of report.scatter_plots) {
    const container = document.createElement("div");

    container.className = "chart-container";

    charts.appendChild(container);

    renderScatterPlot(container, visualization);
  }
}

function renderScatterPlot(container, visualization) {
  const chart = document.createElement("div");

  container.appendChild(chart);

  Plotly.newPlot(
    chart,
    [
      {
        x: visualization.x_values,
        y: visualization.y_values,
        type: "scatter",
        mode: "markers",
        marker: {
          size: 6,
        },
      },
    ],
    {
      title: `${visualization.column_x} vs ${visualization.column_y}`,
      xaxis: {
        title: visualization.column_x,
      },
      yaxis: {
        title: visualization.column_y,
      },
    },
    {
      responsive: true,
    },
  );
}
