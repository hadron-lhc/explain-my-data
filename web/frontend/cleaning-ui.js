export function renderExtremeColumnOptions(report, extremeColumns) {
  extremeColumns.innerHTML = "";

  const columns = report.profile?.columns ?? [];
  const numericColumns = columns.filter((column) => column.role === "numeric");

  if (numericColumns.length === 0) {
    extremeColumns.innerHTML = `
      <p>No numeric columns are available for outlier removal.</p>
    `;
    return;
  }

  for (const column of numericColumns) {
    const qualityColumn = report.quality?.columns?.find(
      (quality) => quality.column_name === column.name,
    );

    const outlierCount = qualityColumn?.n_outliers ?? 0;
    const label = document.createElement("label");

    label.className = "extreme-column-option";
    label.innerHTML = `
      <input
        type="checkbox"
        name="remove-outlier-column"
        value="${column.name}"
      />
      <span>
        <strong>${column.name}</strong>
        ${outlierCount.toLocaleString()} statistical outliers
      </span>
    `;

    extremeColumns.appendChild(label);
  }
}

export function renderNumericRanges(report, numericRanges) {
  numericRanges.innerHTML = "";

  const numericStatistics = report.numeric_statistics ?? [];

  if (numericStatistics.length === 0) {
    numericRanges.innerHTML = `
      <p>No numeric columns are available.</p>
    `;
    return;
  }

  for (const stat of numericStatistics) {
    const min = Number(stat.min);
    const max = Number(stat.max);

    if (!Number.isFinite(min) || !Number.isFinite(max)) {
      continue;
    }

    const container = document.createElement("div");
    container.className = "numeric-range";
    container.dataset.column = stat.column_name;

    container.innerHTML = `
      <div class="numeric-range-header">
        <strong>${stat.column_name}</strong>
        <span class="numeric-range-values">
          <span class="range-value-min">${min}</span>
          <span>–</span>
          <span class="range-value-max">${max}</span>
        </span>
      </div>

      <div class="range-slider">
        <div class="range-track"></div>
        <div class="range-progress"></div>

        <input
          type="range"
          class="range-input range-min"
          min="${min}"
          max="${max}"
          value="${min}"
          step="any"
          aria-label="Minimum value for ${stat.column_name}"
        />

        <input
          type="range"
          class="range-input range-max"
          min="${min}"
          max="${max}"
          value="${max}"
          step="any"
          aria-label="Maximum value for ${stat.column_name}"
        />
      </div>
    `;

    numericRanges.appendChild(container);

    const minInput = container.querySelector(".range-min");
    const maxInput = container.querySelector(".range-max");
    const minLabel = container.querySelector(".range-value-min");
    const maxLabel = container.querySelector(".range-value-max");
    const progress = container.querySelector(".range-progress");

    function updateRange() {
      let currentMin = Number(minInput.value);
      let currentMax = Number(maxInput.value);

      if (currentMin > currentMax) {
        if (document.activeElement === minInput) {
          currentMin = currentMax;
          minInput.value = currentMin;
        } else {
          currentMax = currentMin;
          maxInput.value = currentMax;
        }
      }

      minLabel.textContent = currentMin;
      maxLabel.textContent = currentMax;

      const span = max - min;

      if (span > 0) {
        const minPercent = ((currentMin - min) / span) * 100;
        const maxPercent = ((currentMax - min) / span) * 100;

        progress.style.left = `${minPercent}%`;
        progress.style.width = `${maxPercent - minPercent}%`;
      }
    }

    minInput.addEventListener("input", updateRange);
    maxInput.addEventListener("input", updateRange);

    updateRange();
  }
}

export function openCleaningDrawer({
  cleaningOptions,
  cleaningOverlay,
  toggleCleaningButton,
}) {
  cleaningOptions.classList.add("open");
  cleaningOverlay.classList.remove("hidden");
  cleaningOptions.setAttribute("aria-hidden", "false");
  toggleCleaningButton.setAttribute("aria-expanded", "true");

  document.body.style.overflow = "hidden";
}

export function closeCleaningDrawer({
  cleaningOptions,
  cleaningOverlay,
  toggleCleaningButton,
}) {
  toggleCleaningButton.focus();

  cleaningOptions.classList.remove("open");
  cleaningOverlay.classList.add("hidden");
  cleaningOptions.setAttribute("aria-hidden", "true");
  toggleCleaningButton.setAttribute("aria-expanded", "false");

  document.body.style.overflow = "";
}
