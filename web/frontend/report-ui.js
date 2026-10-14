function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function formatNumber(value) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "—";
  }

  if (Number.isInteger(value)) {
    return value.toLocaleString();
  }

  return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
}

function formatDecimal(value, digits = 2) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "—";
  }

  return value.toFixed(digits);
}

function kpiCard(label, value) {
  return `
    <div class="kpi-card">
      <strong>${escapeHtml(value)}</strong>
      <span>${escapeHtml(label)}</span>
    </div>
  `;
}

function columnDetailHtml(column, report) {
  const numeric = (report.numeric_statistics ?? []).find(
    (stat) => stat.column_name === column.name,
  );

  const categorical = (report.categorical_statistics ?? []).find(
    (stat) => stat.column_name === column.name,
  );

  const details = [];

  if (numeric) {
    details.push(
      ["Mean", formatDecimal(numeric.mean)],
      ["Median", formatDecimal(numeric.median)],
      ["Std", formatDecimal(numeric.std)],
      ["Min", formatDecimal(numeric.min)],
      ["Max", formatDecimal(numeric.max)],
      ["Skewness", formatDecimal(numeric.skewness)],
    );
  }

  if (categorical) {
    details.push(
      ["Dominant value", categorical.dominant_value ?? "—"],
      [
        "Dominant share",
        `${formatDecimal(categorical.dominant_percentage, 1)}%`,
      ],
      ["Unique values", formatNumber(categorical.n_unique)],
    );
  }

  details.push(
    ["Missing", `${formatDecimal(column.pct_missing, 1)}%`],
    ["Unique", formatNumber(column.n_unique)],
  );

  const notes = (column.notes ?? [])
    .map((note) => `<li>${escapeHtml(note)}</li>`)
    .join("");

  const samples = (column.sample_values ?? []).slice(0, 5);

  return `
    <div class="statistics-detail-content">
      ${details
        .map(
          ([label, value]) => `
            <div>
              <span>${escapeHtml(label)}</span>
              <strong>${escapeHtml(value)}</strong>
            </div>
          `,
        )
        .join("")}
    </div>

    ${
      samples.length
        ? `
          <div class="column-samples">
            <span>Sample values</span>
            <code>${samples.map(escapeHtml).join(", ")}</code>
          </div>
        `
        : ""
    }

    ${notes ? `<ul class="column-notes">${notes}</ul>` : ""}
  `;
}

export function renderOverview(report, overview) {
  const profile = report.profile;
  const columns = profile.columns ?? [];

  const missingCells = columns.reduce(
    (sum, column) => sum + (column.n_missing ?? 0),
    0,
  );

  const numericColumns = columns.filter(
    (column) => column.role === "numeric",
  ).length;

  const categoricalColumns = columns.filter(
    (column) => column.role === "categorical" || column.role === "boolean",
  ).length;

  if (columns.length === 0) {
    overview.innerHTML = "<p>No profile available.</p>";
    return;
  }

  const rows = columns
    .map((column, index) => {
      return `
        <tr>
          <td>
            <button
              type="button"
              class="statistics-expand"
              aria-expanded="false"
              data-detail="column-detail-${index}"
            >
              ${escapeHtml(column.name)}
            </button>
          </td>

          <td><code>${escapeHtml(column.dtype)}</code></td>

          <td>
            <span class="role-badge role-${escapeHtml(column.role)}">
              ${escapeHtml(column.role)}
            </span>
          </td>

          <td>${formatDecimal(column.pct_missing, 1)}%</td>

          <td>${formatNumber(column.n_unique)}</td>
        </tr>

        <tr id="column-detail-${index}" class="statistics-detail hidden">
          <td colspan="5">${columnDetailHtml(column, report)}</td>
        </tr>
      `;
    })
    .join("");

  overview.innerHTML = `
    <div class="kpi-grid">
      ${kpiCard("rows", formatNumber(profile.n_rows))}
      ${kpiCard("columns", formatNumber(profile.n_columns))}
      ${kpiCard("duplicate rows", formatNumber(profile.n_duplicates))}
      ${kpiCard("missing cells", formatNumber(missingCells))}
      ${kpiCard("numeric columns", formatNumber(numericColumns))}
      ${kpiCard(
        "categorical columns",
        formatNumber(categoricalColumns),
      )}
    </div>

    <div class="table-wrapper">
      <table class="statistics-table">
        <thead>
          <tr>
            <th>Column</th>
            <th>Dtype</th>
            <th>Role</th>
            <th>Missing</th>
            <th>Unique</th>
          </tr>
        </thead>

        <tbody>${rows}</tbody>
      </table>
    </div>
  `;

  overview.querySelectorAll(".statistics-expand").forEach((button) => {
    button.addEventListener("click", () => {
      const detail = document.getElementById(button.dataset.detail);
      const expanded = !detail.classList.contains("hidden");

      detail.classList.toggle("hidden");
      button.setAttribute("aria-expanded", String(!expanded));
    });
  });
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
          ${escapeHtml(insight.title)}
        </strong>
      </div>

      <p>
        ${escapeHtml(insight.message)}
      </p>
    `;

    quality.appendChild(card);
  }
}

export function renderPatterns(report, patternsContainer) {
  const groupPatterns = [...(report.group_difference_patterns ?? [])].sort(
    (a, b) =>
      Math.abs(b.standardized_difference ?? 0) -
      Math.abs(a.standardized_difference ?? 0),
  );

  const outlierPatterns = (report.numeric_outlier_patterns ?? [])
    .filter((pattern) => pattern.outlier_count > 0)
    .sort((a, b) => b.outlier_count - a.outlier_count);

  if (groupPatterns.length === 0 && outlierPatterns.length === 0) {
    patternsContainer.innerHTML = `
      <div class="pattern-block">
        <strong>No structural patterns detected</strong>

        <p>
          The dataset did not trigger any group-difference or
          outlier pattern template.
        </p>
      </div>
    `;

    return;
  }

  const groupRows = groupPatterns
    .map((pattern) => {
      return `
        <tr>
          <td>${escapeHtml(pattern.group_column)}</td>
          <td><strong>${escapeHtml(pattern.group_value)}</strong></td>
          <td>${formatNumber(pattern.group_mean)}</td>
          <td>${formatNumber(pattern.overall_mean)}</td>
          <td>${formatNumber(pattern.difference)}</td>
          <td>${formatDecimal(pattern.standardized_difference)}σ</td>
          <td>${formatNumber(pattern.group_size)}</td>
        </tr>
      `;
    })
    .join("");

  const outlierRows = outlierPatterns
    .map((pattern) => {
      const percentage =
        pattern.n_observations > 0
          ? (pattern.outlier_count / pattern.n_observations) * 100
          : 0;

      return `
        <tr>
          <td><strong>${escapeHtml(pattern.column)}</strong></td>
          <td>${formatNumber(pattern.outlier_count)}</td>
          <td>${formatDecimal(percentage, 1)}%</td>
          <td>${formatNumber(pattern.lower_bound)}</td>
          <td>${formatNumber(pattern.upper_bound)}</td>
        </tr>
      `;
    })
    .join("");

  patternsContainer.innerHTML = `
    ${
      groupPatterns.length
        ? `
          <div class="pattern-block">
            <h3>Group differences</h3>

            <p class="section-sub">
              Groups whose average differs noticeably from the overall
              average of the numeric column.
            </p>

            <div class="table-wrapper">
              <table>
                <thead>
                  <tr>
                    <th>Column</th>
                    <th>Group</th>
                    <th>Group mean</th>
                    <th>Overall mean</th>
                    <th>Difference</th>
                    <th>Std dev</th>
                    <th>Group size</th>
                  </tr>
                </thead>

                <tbody>${groupRows}</tbody>
              </table>
            </div>
          </div>
        `
        : ""
    }

    ${
      outlierPatterns.length
        ? `
          <div class="pattern-block">
            <h3>Numeric outliers</h3>

            <p class="section-sub">
              Columns with values outside the IQR fences (1.5 × IQR).
            </p>

            <div class="table-wrapper">
              <table>
                <thead>
                  <tr>
                    <th>Column</th>
                    <th>Outliers</th>
                    <th>% of observations</th>
                    <th>Lower bound</th>
                    <th>Upper bound</th>
                  </tr>
                </thead>

                <tbody>${outlierRows}</tbody>
              </table>
            </div>
          </div>
        `
        : ""
    }
  `;
}

export function renderReport(report, rawReport) {
  rawReport.textContent = JSON.stringify(report, null, 2);
}

export function renderCleaningReport(report, cleaning) {
  const rows = [
    ["Rows", `${report.original_rows} → ${report.final_rows}`],
    ["Duplicates removed", report.duplicates_removed],
    ["Missing values filled", report.missing_values_filled],
    ["Missing rows removed", report.missing_rows_removed],
    ["Invalid values replaced", report.invalid_values_replaced],
    ["Type conversions", report.type_conversions],
    ["Outlier rows removed", report.outlier_rows_removed],
    ["Extreme rows removed", report.range_rows_removed],
  ];

  cleaning.innerHTML = `
    <div class="cleaning-summary">
      <h3>Before → after</h3>

      <div class="table-wrapper">
        <table>
          <thead>
            <tr>
              <th>Action</th>
              <th>Result</th>
            </tr>
          </thead>

          <tbody>
            ${rows
              .map(
                ([label, value]) => `
                  <tr>
                    <td>${escapeHtml(label)}</td>
                    <td><strong>${escapeHtml(value)}</strong></td>
                  </tr>
                `,
              )
              .join("")}
          </tbody>
        </table>
      </div>
    </div>
  `;
}
