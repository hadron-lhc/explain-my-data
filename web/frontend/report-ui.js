export function renderOverview(report, overview) {
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

export function renderQualityInsights(report, quality) {
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

        <strong>
          ${insight.title}
        </strong>
      </div>

      <p>
        ${insight.message}
      </p>
    `;

    quality.appendChild(card);
  }
}

export function renderStatistics(report, statistics) {
  statistics.innerHTML = "";

  const numericStatistics = report.numeric_statistics ?? [];
  const categoricalStatistics = report.categorical_statistics ?? [];

  if (numericStatistics.length === 0 && categoricalStatistics.length === 0) {
    statistics.innerHTML = "<p>No statistics available.</p>";
    return;
  }

  const rows = [
    ...numericStatistics.map((stat) => ({
      column: stat.column_name,
      type: "numeric",
      count: stat.count,
      unique: "—",
      mainValue: stat.mean.toFixed(2),
      mainLabel: "Mean",
      details: [
        ["Median", stat.median.toFixed(2)],
        ["Std", stat.std.toFixed(2)],
        ["Min", stat.min.toFixed(2)],
        ["Max", stat.max.toFixed(2)],
        ["Skewness", stat.skewness.toFixed(2)],
      ],
    })),

    ...categoricalStatistics.map((stat) => ({
      column: stat.column_name,
      type: "categorical",
      count: stat.count,
      unique: stat.n_unique,
      mainValue: `${stat.dominant_percentage.toFixed(2)}%`,
      mainLabel: "Dominant",
      details: [
        ["Dominant value", stat.dominant_value ?? "—"],
        ["Dominant percentage", `${stat.dominant_percentage.toFixed(2)}%`],
      ],
    })),
  ];

  const table = document.createElement("table");
  table.className = "statistics-table";

  table.innerHTML = `
    <thead>
      <tr>
        <th>Column</th>
        <th>Type</th>
        <th>Count</th>
        <th>Unique</th>
        <th>Main metric</th>
      </tr>
    </thead>

    <tbody></tbody>
  `;

  const tbody = table.querySelector("tbody");

  for (const row of rows) {
    const tr = document.createElement("tr");

    tr.innerHTML = `
      <td>
        <button
          type="button"
          class="statistics-expand"
          aria-expanded="false"
        >
          ${row.column}
        </button>
      </td>

      <td>${row.type}</td>

      <td>
        ${row.count.toLocaleString()}
      </td>

      <td>
        ${row.unique === "—" ? "—" : row.unique.toLocaleString()}
      </td>

      <td>
        <span class="statistics-main-label">
          ${row.mainLabel}
        </span>

        ${row.mainValue}
      </td>
    `;

    const detailRow = document.createElement("tr");
    detailRow.className = "statistics-detail hidden";

    const detailCell = document.createElement("td");
    detailCell.colSpan = 5;

    detailCell.innerHTML = `
      <div class="statistics-detail-content">
        ${row.details
          .map(
            ([label, value]) => `
              <div>
                <span>${label}</span>
                <strong>${value}</strong>
              </div>
            `,
          )
          .join("")}
      </div>
    `;

    detailRow.appendChild(detailCell);

    const expandButton = tr.querySelector(".statistics-expand");

    expandButton.addEventListener("click", () => {
      const expanded = !detailRow.classList.contains("hidden");

      detailRow.classList.toggle("hidden");
      expandButton.setAttribute("aria-expanded", String(!expanded));
    });

    tbody.appendChild(tr);
    tbody.appendChild(detailRow);
  }

  statistics.appendChild(table);
}

export function renderReport(report, rawReport) {
  rawReport.textContent = JSON.stringify(report, null, 2);
}

export function renderCleaningReport(report, cleaning) {
  cleaning.innerHTML = `
    <div class="cleaning-summary">
      <p>
        <strong>Rows:</strong>
        ${report.original_rows}
        →
        ${report.final_rows}
      </p>

      <p>
        <strong>Duplicates removed:</strong>
        ${report.duplicates_removed}
      </p>

      <p>
        <strong>Missing values filled:</strong>
        ${report.missing_values_filled}
      </p>

      <p>
        <strong>Missing rows removed:</strong>
        ${report.missing_rows_removed}
      </p>

      <p>
        <strong>Invalid values replaced:</strong>
        ${report.invalid_values_replaced}
      </p>

      <p>
        <strong>Type conversions:</strong>
        ${report.type_conversions}
      </p>

      <p>
        <strong>Outlier rows removed:</strong>
        ${report.outlier_rows_removed}
      </p>

      <p>
        <strong>Extreme rows removed:</strong>
        ${report.range_rows_removed}
      </p>
    </div>
  `;
}
