const fileInput = document.getElementById("file-input");
const analyzeButton = document.getElementById("analyze-button");
const status = document.getElementById("status");
const results = document.getElementById("results");

const overview = document.getElementById("overview");
const quality = document.getElementById("quality");
const relationships = document.getElementById("relationships");
const charts = document.getElementById("charts");
const rawReport = document.getElementById("raw-report");

const CHART_COLORS = [
  "#2563eb",
  "#7c3aed",
  "#0891b2",
  "#059669",
  "#d97706",
  "#dc2626",
];

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

    renderOverview(report);
    renderQualityInsights(report);
    renderRelationshipInsights(report);
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

function renderQualityInsights(report) {
  quality.innerHTML = "";

  const insights = report.quality_insights ?? [];

  if (insights.length === 0) {
    quality.innerHTML = `
      <div class="quality-card quality-ok">
        <strong>No quality issues detected</strong>
        <p>
          No issues requiring review were identified.
        </p>
      </div>
    `;
    return;
  }

  for (const insight of insights) {
    const card = document.createElement("div");
    card.className = "quality-card";

    card.innerHTML = `
      <div class="quality-card-title">
        <span class="quality-icon">⚠</span>
        <strong>${insight.title}</strong>
      </div>

      <p>${insight.message}</p>
    `;

    quality.appendChild(card);
  }
}

function renderRelationshipInsights(report) {
  relationships.innerHTML = "";

  const insights = report.relationship_insights ?? [];

  if (insights.length === 0) {
    relationships.innerHTML = `
      <div class="insight-card insight-empty">
        <strong>No strong relationships detected</strong>

        <p>
          No relationships met the current
          analysis criteria.
        </p>
      </div>
    `;

    return;
  }

  for (const insight of insights) {
    const section = document.createElement("div");
    section.className = "relationship-section";

    const card = document.createElement("div");
    card.className = "insight-card";

    card.innerHTML = `
      <div class="insight-card-title">
        <strong>${insight.title}</strong>
      </div>

      <p>${insight.message}</p>
    `;

    section.appendChild(card);

    // Primero hacemos visible el section en el DOM.
    relationships.appendChild(section);

    const visualizations = findVisualizationsForInsight(insight, report);

    for (const visualization of visualizations) {
      const chartContainer = document.createElement("div");

      chartContainer.className = "relationship-chart";

      section.appendChild(chartContainer);

      renderRelationshipVisualization(chartContainer, visualization);
    }
  }
}

function findVisualizationsForInsight(insight, report) {
  const numericMatch = insight.title.match(/between (.+) and (.+)$/);

  if (numericMatch) {
    const [, columnX, columnY] = numericMatch;

    const visualizations = [];

    const scatter = report.scatter_plots?.find(
      (plot) => plot.column_x === columnX && plot.column_y === columnY,
    );

    if (scatter) {
      visualizations.push({
        type: "scatter",
        data: scatter,
      });
    }

    const boxplot = report.boxplots?.find(
      (plot) =>
        (plot.column_categorical === columnX &&
          plot.column_numeric === columnY) ||
        (plot.column_categorical === columnY &&
          plot.column_numeric === columnX),
    );

    if (boxplot) {
      visualizations.push({
        type: "boxplot",
        data: boxplot,
      });
    }

    return visualizations;
  }
  const categoricalMatch = insight.title.match(
    /^(.+) differs across (.+) groups$/,
  );

  if (categoricalMatch) {
    const [, columnNumeric, columnCategorical] = categoricalMatch;

    const visualizations = [];

    const bar = report.bar_plots?.find(
      (plot) =>
        plot.column_numeric === columnNumeric &&
        plot.column_categorical === columnCategorical,
    );

    if (bar) {
      visualizations.push({
        type: "bar",
        data: bar,
      });
    }

    const boxplot = report.boxplots?.find(
      (plot) =>
        plot.column_numeric === columnNumeric &&
        plot.column_categorical === columnCategorical,
    );

    if (boxplot) {
      visualizations.push({
        type: "boxplot",
        data: boxplot,
      });
    }

    return visualizations;
  }

  return [];
}

function renderRelationshipVisualization(container, visualization) {
  if (visualization.type === "scatter") {
    renderScatterPlot(container, visualization.data);

    return;
  }

  if (visualization.type === "bar") {
    renderBarPlot(container, visualization.data);

    return;
  }

  if (visualization.type === "boxplot") {
    renderBoxPlot(container, visualization.data);
  }
}

function createChartElement(container) {
  const chart = document.createElement("div");

  chart.className = "plotly-chart";

  container.appendChild(chart);

  return chart;
}

function getPlotConfig() {
  return {
    responsive: true,
    displayModeBar: false,
  };
}

function getCommonLayout() {
  return {
    autosize: true,

    paper_bgcolor: "white",
    plot_bgcolor: "white",

    margin: {
      l: 60,
      r: 20,
      t: 50,
      b: 60,
    },

    font: {
      family: "Inter, system-ui, -apple-system, BlinkMacSystemFont, sans-serif",
    },
  };
}

function resizePlot(chart) {
  requestAnimationFrame(() => {
    Plotly.Plots.resize(chart);
  });
}

function renderScatterPlot(container, visualization) {
  const chart = createChartElement(container);

  const traces = [
    {
      x: visualization.x_values,
      y: visualization.y_values,

      type: "scatter",
      mode: "markers",

      marker: {
        size: 5,
        opacity: 0.35,
        color: CHART_COLORS[0],
      },

      hovertemplate:
        `${visualization.column_x}: %{x}<br>` +
        `${visualization.column_y}: %{y}` +
        "<extra></extra>",
    },
  ];

  if (visualization.trend_x && visualization.trend_y) {
    traces.push({
      x: visualization.trend_x,
      y: visualization.trend_y,

      type: "scatter",
      mode: "lines",

      line: {
        color: CHART_COLORS[3],
        width: 3,
      },

      hoverinfo: "skip",
    });
  }

  const layout = {
    ...getCommonLayout(),

    title: {
      text: `${visualization.column_x} vs ${visualization.column_y}`,
      font: {
        size: 16,
      },
    },

    xaxis: {
      title: visualization.column_x,
      gridcolor: "#e5e7eb",
      zeroline: false,
    },

    yaxis: {
      title: visualization.column_y,
      gridcolor: "#e5e7eb",
      zeroline: false,
    },

    showlegend: false,
  };

  Plotly.newPlot(chart, traces, layout, getPlotConfig());

  resizePlot(chart);
}

function renderBarPlot(container, visualization) {
  const chart = createChartElement(container);

  const layout = {
    ...getCommonLayout(),

    title: {
      text: `${visualization.column_numeric} by ${visualization.column_categorical}`,
      font: {
        size: 16,
      },
    },

    xaxis: {
      title: visualization.column_categorical,
      gridcolor: "#e5e7eb",
    },

    yaxis: {
      title: `Mean ${visualization.column_numeric}`,
      gridcolor: "#e5e7eb",
    },
  };

  Plotly.newPlot(
    chart,
    [
      {
        x: visualization.categories,
        y: visualization.values,
        type: "bar",

        marker: {
          color: CHART_COLORS[1],
        },
      },
    ],
    layout,
    getPlotConfig(),
  );

  resizePlot(chart);
}

function renderBoxPlot(container, visualization) {
  const chart = createChartElement(container);

  const groups = new Map();

  for (let i = 0; i < visualization.categories.length; i++) {
    const category = visualization.categories[i];
    const value = visualization.values[i];

    if (!groups.has(category)) {
      groups.set(category, []);
    }

    groups.get(category).push(value);
  }

  const traces = Array.from(groups.entries()).map(
    ([category, values], index) => ({
      y: values,
      type: "box",
      name: category,

      boxpoints: "outliers",

      marker: {
        color: CHART_COLORS[index % CHART_COLORS.length],
      },

      line: {
        color: CHART_COLORS[index % CHART_COLORS.length],
      },

      hovertemplate:
        `${visualization.column_categorical}: ${category}<br>` +
        `${visualization.column_numeric}: %{y}` +
        "<extra></extra>",
    }),
  );

  const layout = {
    ...getCommonLayout(),

    title: {
      text: `${visualization.column_numeric} by ${visualization.column_categorical}`,
      font: {
        size: 16,
      },
    },

    xaxis: {
      title: visualization.column_categorical,
      gridcolor: "#e5e7eb",
    },

    yaxis: {
      title: visualization.column_numeric,
      gridcolor: "#e5e7eb",
      zeroline: false,
    },

    showlegend: false,
  };

  Plotly.newPlot(chart, traces, layout, getPlotConfig());

  resizePlot(chart);
}
window.addEventListener("resize", () => {
  document.querySelectorAll(".plotly-chart").forEach((chart) => {
    Plotly.Plots.resize(chart);
  });
});
