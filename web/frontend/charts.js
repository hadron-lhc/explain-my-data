const CHART_COLORS = [
  "#2563eb",
  "#7c3aed",
  "#0891b2",
  "#059669",
  "#d97706",
  "#dc2626",
];

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
      font: { size: 16 },
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
      text:
        `${visualization.column_numeric} by ` +
        visualization.column_categorical,
      font: { size: 16 },
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
      text:
        `${visualization.column_numeric} by ` +
        visualization.column_categorical,
      font: { size: 16 },
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

function renderHeatmapPlot(container, visualization) {
  const chart = createChartElement(container);

  const traces = [
    {
      x: visualization.x_labels,
      y: visualization.y_labels,
      z: visualization.values,
      type: "heatmap",
      colorscale: "Blues",
      hovertemplate:
        `${visualization.column_x}: %{x}<br>` +
        `${visualization.column_y}: %{y}<br>` +
        `${visualization.value_label}: %{z}` +
        "<extra></extra>",
    },
  ];

  const layout = {
    ...getCommonLayout(),
    title: {
      text:
        `${visualization.column_x} vs ${visualization.column_y} ` +
        `(${visualization.value_label})`,
      font: { size: 16 },
    },
    xaxis: {
      title: visualization.column_x,
      gridcolor: "#e5e7eb",
    },
    yaxis: {
      title: visualization.column_y,
      autorange: "reversed",
      gridcolor: "#e5e7eb",
    },
  };

  Plotly.newPlot(chart, traces, layout, getPlotConfig());
  resizePlot(chart);
}

function renderHistogramPlot(container, distribution) {
  const chart = createChartElement(container);

  if (
    !distribution.bin_counts ||
    distribution.bin_counts.length === 0 ||
    !distribution.bin_edges
  ) {
    return;
  }

  const midpoints = distribution.bin_counts.map((_, index) => {
    const low = distribution.bin_edges[index];
    const high = distribution.bin_edges[index + 1];

    return (low + high) / 2;
  });

  const layout = {
    ...getCommonLayout(),
    title: {
      text: `Distribution of ${distribution.column_name}`,
      font: { size: 16 },
    },
    xaxis: {
      title: distribution.column_name,
      gridcolor: "#e5e7eb",
    },
    yaxis: {
      title: "count",
      gridcolor: "#e5e7eb",
    },
    showlegend: false,
  };

  Plotly.newPlot(
    chart,
    [
      {
        x: midpoints,
        y: distribution.bin_counts,
        type: "bar",
        marker: {
          color: CHART_COLORS[0],
        },
        hovertemplate:
          `${distribution.column_name}: %{x}<br>count: %{y}` +
          "<extra></extra>",
      },
    ],
    layout,
    getPlotConfig(),
  );

  resizePlot(chart);
}

function findVisualizationsForInsight(insight, report) {
  if (!insight.chart_id) {
    return [];
  }

  const [kind, ...parts] = insight.chart_id.split(":");
  const visualizations = [];

  if (kind === "scatter") {
    const [x, y] = parts;

    const scatter = report.scatter_plots?.find(
      (plot) =>
        (plot.column_x === x && plot.column_y === y) ||
        (plot.column_x === y && plot.column_y === x),
    );

    if (scatter) visualizations.push({ type: "scatter", data: scatter });

    return visualizations;
  }

  if (kind === "catnum") {
    const [columnCategorical, columnNumeric] = parts;

    const bar = report.bar_plots?.find(
      (plot) =>
        plot.column_categorical === columnCategorical &&
        plot.column_numeric === columnNumeric,
    );

    if (bar) visualizations.push({ type: "bar", data: bar });

    const boxplot = report.boxplots?.find(
      (plot) =>
        plot.column_categorical === columnCategorical &&
        plot.column_numeric === columnNumeric,
    );

    if (boxplot) visualizations.push({ type: "boxplot", data: boxplot });

    return visualizations;
  }

  if (kind === "catcat") {
    const [columnX, columnY] = parts;

    const heatmap = report.heatmaps?.find(
      (plot) =>
        (plot.column_x === columnX && plot.column_y === columnY) ||
        (plot.column_x === columnY && plot.column_y === columnX),
    );

    if (heatmap) visualizations.push({ type: "heatmap", data: heatmap });

    return visualizations;
  }

  if (kind === "hist") {
    const [columnName] = parts;

    const distribution = report.numeric_distributions?.find(
      (entry) => entry.column_name === columnName,
    );

    if (distribution) {
      visualizations.push({ type: "histogram", data: distribution });
    }

    return visualizations;
  }

  return visualizations;
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
    return;
  }

  if (visualization.type === "heatmap") {
    renderHeatmapPlot(container, visualization.data);
    return;
  }

  if (visualization.type === "histogram") {
    renderHistogramPlot(container, visualization.data);
  }
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function formatEvidence(value) {
  if (typeof value === "number" && Number.isFinite(value)) {
    if (Number.isInteger(value)) {
      return value.toLocaleString();
    }

    return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
  }

  return value;
}

function visualizationIdentity(visualization) {
  const { type, data } = visualization;

  if (type === "scatter") {
    return `scatter:${[data.column_x, data.column_y].sort().join(":")}`;
  }

  if (type === "bar") {
    return `bar:${data.column_categorical}:${data.column_numeric}`;
  }

  if (type === "boxplot") {
    return `boxplot:${data.column_categorical}:${data.column_numeric}`;
  }

  if (type === "heatmap") {
    return `heatmap:${[data.column_x, data.column_y].sort().join(":")}`;
  }

  if (type === "histogram") {
    return `histogram:${data.column_name}`;
  }

  return type;
}

function findPatternForInsight(insight, report) {
  const columns = insight.columns ?? [];

  if (columns.length === 0) {
    return null;
  }

  const groupPattern = (report.group_difference_patterns ?? []).find(
    (pattern) =>
      columns.includes(pattern.group_column) &&
      columns.includes(pattern.numeric_column),
  );

  if (groupPattern) {
    return { type: "group", pattern: groupPattern };
  }

  const outlierPattern = (report.numeric_outlier_patterns ?? []).find(
    (pattern) =>
      pattern.outlier_count > 0 && columns.includes(pattern.column),
  );

  // Solo ahí el outlier es parte del propio hallazgo: en rankings o
  // avisos de calidad sería ruido tangencial.
  if (
    outlierPattern &&
    (insight.category === "pattern" ||
      insight.category === "distribution")
  ) {
    return { type: "outlier", pattern: outlierPattern };
  }

  return null;
}

function patternHtml(insightPattern) {
  if (!insightPattern) {
    return "";
  }

  if (insightPattern.type === "group") {
    const pattern = insightPattern.pattern;

    return `
      <div class="insight-pattern">
        <span class="insight-pattern-label">Pattern</span>

        <p>
          In <strong>${escapeHtml(pattern.group_column)} =
          "${escapeHtml(pattern.group_value)}"</strong>,
          <strong>${escapeHtml(pattern.numeric_column)}</strong>
          averages <strong>${formatEvidence(pattern.group_mean)}</strong>
          vs an overall average of
          <strong>${formatEvidence(pattern.overall_mean)}</strong>
          (${formatEvidence(pattern.standardized_difference)}σ,
          n=${formatEvidence(pattern.group_size)}).
        </p>
      </div>
    `;
  }

  const pattern = insightPattern.pattern;

  const percentage =
    pattern.n_observations > 0
      ? (
          (pattern.outlier_count / pattern.n_observations) *
          100
        ).toFixed(1)
      : null;

  return `
    <div class="insight-pattern">
      <span class="insight-pattern-label">Pattern</span>

      <p>
        <strong>${escapeHtml(pattern.column)}</strong> has
        <strong>${formatEvidence(pattern.outlier_count)}</strong>
        values${percentage ? ` (${percentage}%)` : ""}
        outside the IQR range
        [<strong>${formatEvidence(pattern.lower_bound)}</strong>,
        <strong>${formatEvidence(pattern.upper_bound)}</strong>].
      </p>
    </div>
  `;
}

const CATEGORY_ORDER = [
  "comparative",
  "relationship",
  "pattern",
  "distribution",
  "aggregation",
  "structure",
  "quality",
];

function categoryLabel(category) {
  if (!category) {
    return "General";
  }

  return category.charAt(0).toUpperCase() + category.slice(1);
}

function appendInsightCard(parent, report, insight, options) {
  const { key = false, consumed, onHide = null } = options;

  const visualizations = findVisualizationsForInsight(
    insight,
    report,
  ).slice(0, 2);

  for (const visualization of visualizations) {
    consumed.add(visualizationIdentity(visualization));
  }

  const card = document.createElement("div");

  if (insight.id) {
    card.id = `insight-${insight.id}`;
  }

  const severity = insight.severity ?? "info";

  card.className =
    `insight-card severity-${severity}` + (key ? " insight-key" : "");

  const evidence = Object.entries(insight.evidence ?? {})
    .map(
      ([label, value]) =>
        `<li>${escapeHtml(label)}: ${escapeHtml(formatEvidence(value))}</li>`,
    )
    .join("");

  const caveats = (insight.caveats ?? [])
    .map((caveat) => escapeHtml(caveat))
    .join(" ");

  const pattern = findPatternForInsight(insight, report);

  card.innerHTML = `
    <div class="insight-card-header">
      ${
        onHide
          ? `<button
               type="button"
               class="insight-dismiss"
               title="Hide this insight"
               aria-label="Hide this insight"
             >&times;</button>`
          : ""
      }

      <div class="insight-badges">
        <span class="insight-category">
          ${escapeHtml(insight.category ?? "general")}
        </span>

        <span class="severity-badge severity-${escapeHtml(severity)}">
          ${escapeHtml(severity)}
        </span>

        ${key ? '<span class="key-badge">★ Key</span>' : ""}
      </div>

      <strong class="insight-card-title">
        ${escapeHtml(insight.title ?? "")}
      </strong>
    </div>

    <div class="insight-card-body">
      <div class="insight-card-text">
        <p>${escapeHtml(insight.message ?? "")}</p>

        ${evidence ? `<ul class="insight-evidence">${evidence}</ul>` : ""}

        ${caveats ? `<p class="insight-caveats">${caveats}</p>` : ""}

        ${patternHtml(pattern)}
      </div>

      ${
        visualizations.length
          ? '<div class="insight-card-charts"></div>'
          : ""
      }
    </div>
  `;

  parent.appendChild(card);

  if (onHide) {
    const dismissButton = card.querySelector(".insight-dismiss");

    dismissButton.addEventListener("click", (event) => {
      event.stopPropagation();
      onHide(insight.id);
    });
  }

  if (visualizations.length) {
    const charts = card.querySelector(".insight-card-charts");

    for (const visualization of visualizations) {
      const chartContainer = document.createElement("div");
      chartContainer.className = "relationship-chart";
      charts.appendChild(chartContainer);

      renderRelationshipVisualization(chartContainer, visualization);
    }
  }

  return card;
}

function appendKeyInsightLine(parent, insight, sourceById, options = {}) {
  const { onHide = null } = options;

  const source = insight.source_id
    ? sourceById.get(insight.source_id)
    : null;

  const card = document.createElement("div");

  card.className = "insight-card insight-key insight-key-line";

  const dismiss = onHide
    ? `<button
         type="button"
         class="insight-dismiss"
         title="Hide this summary line"
         aria-label="Hide this summary line"
       >&times;</button>`
    : "";

  const titleHtml = source
    ? `<a class="insight-key-link" href="#insight-${escapeHtml(source.id)}">${escapeHtml(source.title)}</a>`
    : escapeHtml(insight.title ?? "");

  const bodyHtml = source
    ? ""
    : `<div class="insight-card-body">
         <div class="insight-card-text">
           <p>${escapeHtml(insight.message ?? "")}</p>
         </div>
       </div>`;

  card.innerHTML = `
    <div class="insight-card-header">
      ${dismiss}

      <div class="insight-badges">
        <span class="key-badge">★ Key</span>
      </div>

      <strong class="insight-card-title">${titleHtml}</strong>
    </div>

    ${bodyHtml}
  `;

  parent.appendChild(card);

  if (onHide) {
    const dismissButton = card.querySelector(".insight-dismiss");

    dismissButton.addEventListener("click", (event) => {
      event.stopPropagation();
      onHide(insight.id);
    });
  }

  return card;
}

function appendInsightGroup(container, title, isKey = false) {
  const group = document.createElement("div");

  group.className = "insight-group" + (isKey ? " insight-group-key" : "");

  group.innerHTML = `
    <h3 class="insight-group-title">
      ${isKey ? '<span class="key-star">★</span>' : ""}
      ${escapeHtml(title)}
    </h3>
  `;

  container.appendChild(group);

  return group;
}

export function renderInsightsSection(report, container, options = {}) {
  const { hiddenIds = new Set(), onHide = null } = options;

  container.innerHTML = "";

  const executiveAll = report.executive_summary ?? [];

  const allInsights = [
    ...(report.quality_insights ?? []),
    ...(report.relationship_insights ?? []),
    ...(report.distribution_insights ?? []),
    ...(report.pattern_insights ?? []),
    ...(report.structure_insights ?? []),
    ...(report.aggregation_insights ?? []),
    ...(report.comparative_insights ?? []),
  ];

  const executive = executiveAll.filter(
    (insight) => !hiddenIds.has(insight.id),
  );

  const executiveIds = new Set(
    executive.map((insight) => insight.id).filter(Boolean),
  );

  // Dedupe por id: en el navegador los objetos de executive_summary y de
  // sus listas de origen son copias distintas (JSON no preserva
  // referencias), así que la comparación por identidad no funciona.
  const seenIds = new Set();

  const all = allInsights.filter((insight) => {
    if (hiddenIds.has(insight.id)) {
      return false;
    }

    if (seenIds.has(insight.id)) {
      return false;
    }

    seenIds.add(insight.id);
    return true;
  });

  const hadInsights = executiveAll.length > 0 || allInsights.length > 0;

  const consumed = new Set();

  if (all.length === 0 && executive.length === 0) {
    const everythingHidden = hadInsights && hiddenIds.size > 0;

    container.innerHTML = `
      <div class="insight-card insight-empty">
        <strong>${
          everythingHidden
            ? "All insights are hidden"
            : "No insights detected"
        }</strong>

        <p>
          ${
            everythingHidden
              ? 'Use "Restore all" above to bring them back.'
              : "The dataset did not trigger any of the current analysis templates."
          }
        </p>
      </div>
    `;

    return consumed;
  }

  if (executive.length > 0) {
    const group = appendInsightGroup(
      container,
      "Key insights",
      true,
    );

    // F4: el resumen ejecutivo es un índice, no una segunda copia
    // de las tarjetas. Si la fuente está visible, se enlaza a ella
    // (sin reimprimir el mensaje); si pertenece a una familia que
    // la SPA no renderiza (inference/temporal/ml), se muestra la
    // frase compacta una sola vez.
    const sourceById = new Map(
      all.map((insight) => [insight.id, insight]),
    );

    for (const insight of executive) {
      if (insight.source_id && hiddenIds.has(insight.source_id)) {
        continue;
      }

      appendKeyInsightLine(group, insight, sourceById, { onHide });
    }
  }

  const remaining = all.filter(
    (insight) => !executiveIds.has(insight.id),
  );

  const byCategory = new Map();

  for (const insight of remaining) {
    const category = insight.category ?? "general";

    if (!byCategory.has(category)) {
      byCategory.set(category, []);
    }

    byCategory.get(category).push(insight);
  }

  const orderedCategories = [...byCategory.keys()].sort((a, b) => {
    const indexA = CATEGORY_ORDER.indexOf(a);
    const indexB = CATEGORY_ORDER.indexOf(b);

    if (indexA === -1 && indexB === -1) return 0;
    if (indexA === -1) return 1;
    if (indexB === -1) return -1;

    return indexA - indexB;
  });

  for (const category of orderedCategories) {
    const insights = byCategory.get(category);

    const group = appendInsightGroup(
      container,
      `${categoryLabel(category)} · ${insights.length}`,
    );

    for (const insight of insights) {
      appendInsightCard(group, report, insight, { consumed, onHide });
    }
  }

  return consumed;
}

const EXTRA_CHART_LIMITS = {
  histogram: 6,
  heatmap: 3,
  scatter: 3,
  bar: 3,
  boxplot: 2,
};

export function reportChartIdentities(report) {
  const groups = [
    ["histogram", report.numeric_distributions ?? []],
    ["heatmap", report.heatmaps ?? []],
    ["scatter", report.scatter_plots ?? []],
    ["bar", report.bar_plots ?? []],
    ["boxplot", report.boxplots ?? []],
  ];

  const identities = [];

  for (const [type, entries] of groups) {
    for (const entry of entries) {
      identities.push(visualizationIdentity({ type, data: entry }));
    }
  }

  return identities;
}

function renderChartEntry(chartEl, type, entry) {
  if (type === "histogram") {
    renderHistogramPlot(chartEl, entry);
  } else if (type === "heatmap") {
    renderHeatmapPlot(chartEl, entry);
  } else if (type === "scatter") {
    renderScatterPlot(chartEl, entry);
  } else if (type === "bar") {
    renderBarPlot(chartEl, entry);
  } else if (type === "boxplot") {
    renderBoxPlot(chartEl, entry);
  } else {
    return false;
  }

  return true;
}

export function renderExtraCharts(report, container, consumed, options = {}) {
  const { hiddenCharts = new Set(), onHideChart = null } = options;

  container.innerHTML = "";

  const groups = [
    ["histogram", report.numeric_distributions ?? []],
    ["heatmap", report.heatmaps ?? []],
    ["scatter", report.scatter_plots ?? []],
    ["bar", report.bar_plots ?? []],
    ["boxplot", report.boxplots ?? []],
  ];

  let rendered = 0;
  let hiddenSkipped = 0;

  for (const [type, entries] of groups) {
    const limit = EXTRA_CHART_LIMITS[type] ?? 3;
    let typeRendered = 0;

    for (const entry of entries) {
      if (rendered >= 12) {
        break;
      }

      const identity = visualizationIdentity({ type, data: entry });

      if (consumed.has(identity)) {
        continue;
      }

      if (hiddenCharts.has(identity)) {
        hiddenSkipped += 1;
        continue;
      }

      const wrapper = document.createElement("div");
      wrapper.className = "extra-chart";
      container.appendChild(wrapper);

      const chartEl = document.createElement("div");
      chartEl.className = "relationship-chart";
      wrapper.appendChild(chartEl);

      if (!renderChartEntry(chartEl, type, entry)) {
        wrapper.remove();
        continue;
      }

      if (onHideChart) {
        const dismiss = document.createElement("button");
        dismiss.type = "button";
        dismiss.className = "chart-dismiss";
        dismiss.title = "Hide this chart";
        dismiss.setAttribute("aria-label", "Hide this chart");
        dismiss.innerHTML = "&times;";

        dismiss.addEventListener("click", (event) => {
          event.stopPropagation();
          onHideChart(identity);
        });

        wrapper.appendChild(dismiss);
      }

      rendered += 1;
      typeRendered += 1;

      if (typeRendered >= limit) {
        break;
      }
    }

    if (rendered >= 12) {
      break;
    }
  }

  if (rendered === 0) {
    container.innerHTML =
      hiddenSkipped > 0
        ? `
      <p class="empty-note">
        All remaining extra charts are hidden.
        Use "Restore all" above to bring them back.
      </p>
    `
        : `
      <p class="empty-note">
        Every generated chart is already attached to an insight above.
      </p>
    `;
  }
}
