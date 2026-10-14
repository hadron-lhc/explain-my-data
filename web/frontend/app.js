import { analyzeFile, cleanFile, fetchHtmlReport } from "./api.js";

import {
  closeCleaningDrawer,
  openCleaningDrawer,
  renderCleaningIssues,
  renderExtremeColumnOptions,
  renderNumericRanges,
} from "./cleaning-ui.js";

import {
  renderCleaningReport,
  renderOverview,
  renderPatterns,
  renderQualityInsights,
  renderReport,
} from "./report-ui.js";

import { renderExtraCharts, renderInsightsSection, reportChartIdentities } from "./charts.js";

const fileInput = document.getElementById("file-input");
const analyzeButton = document.getElementById("analyze-button");
const status = document.getElementById("status");
const results = document.getElementById("results");

const dropzone = document.getElementById("dropzone");
const selectedFileLabel = document.getElementById("selected-file");

const undoCleaningButton = document.getElementById("undo-cleaning-button");

const overview = document.getElementById("overview");
const quality = document.getElementById("quality");
const patterns = document.getElementById("patterns");
const insights = document.getElementById("insights");
const charts = document.getElementById("charts");
const rawReport = document.getElementById("raw-report");

const hiddenInsightsBar = document.getElementById("hidden-insights-bar");
const hiddenInsightsCount = document.getElementById("hidden-insights-count");
const restoreHiddenButton = document.getElementById("restore-hidden-button");

const hiddenChartsBar = document.getElementById("hidden-charts-bar");
const hiddenChartsCount = document.getElementById("hidden-charts-count");
const restoreHiddenChartsButton = document.getElementById(
  "restore-hidden-charts-button",
);

const toggleCleaningButton = document.getElementById("toggle-cleaning-button");
const cleaningOptions = document.getElementById("cleaning-options");
const cleanButton = document.getElementById("clean-button");

const removeDuplicates = document.getElementById("remove-duplicates");
const numericMissingStrategy = document.getElementById(
  "numeric-missing-strategy",
);
const categoricalMissingStrategy = document.getElementById(
  "categorical-missing-strategy",
);
const replaceInvalidValues = document.getElementById("replace-invalid-values");
const convertTypeMismatches = document.getElementById(
  "convert-type-mismatches",
);

const cleaning = document.getElementById("cleaning");
const extremeColumns = document.getElementById("extreme-columns");

const outlierStrategy = document.getElementById("outlier-strategy");
const numericRanges = document.getElementById("numeric-ranges");

let originalReport = null;
let cleanedReport = null;
let currentReport = null;
let cleanedRows = null;

let datasetSignature = null;
let hiddenInsightIds = new Set();
let hiddenChartIds = new Set();

function insightsStorageKey(signature) {
  return `emtd.hidden.v1.${signature.slice(0, 16)}`;
}

function chartsStorageKey(signature) {
  return `emtd.hidden-charts.v1.${signature.slice(0, 16)}`;
}

async function computeDatasetSignature(file) {
  try {
    const buffer = await file.arrayBuffer();

    if (globalThis.crypto && crypto.subtle) {
      const digest = await crypto.subtle.digest("SHA-256", buffer);

      return [...new Uint8Array(digest)]
        .map((byte) => byte.toString(16).padStart(2, "0"))
        .join("");
    }

    // Fallback determinista si no hay secure context (crypto.subtle).
    const bytes = new Uint8Array(buffer);
    let hash = 0x811c9dc5;

    for (const byte of bytes) {
      hash ^= byte;
      hash = Math.imul(hash, 0x01000193) >>> 0;
    }

    return hash.toString(16).padStart(8, "0");
  } catch {
    return "no-signature";
  }
}

function loadStoredSet(key) {
  try {
    const raw = localStorage.getItem(key);
    return new Set(raw ? JSON.parse(raw) : []);
  } catch {
    return new Set();
  }
}

function persistStoredSet(key, set) {
  try {
    if (set.size === 0) {
      localStorage.removeItem(key);
    } else {
      localStorage.setItem(key, JSON.stringify([...set]));
    }
  } catch {
    // localStorage puede no estar disponible; el estado en memoria sigue.
  }
}

function loadHiddenStores(signature) {
  hiddenInsightIds = loadStoredSet(insightsStorageKey(signature));
  hiddenChartIds = loadStoredSet(chartsStorageKey(signature));
}

function persistHiddenInsights() {
  if (datasetSignature) {
    persistStoredSet(insightsStorageKey(datasetSignature), hiddenInsightIds);
  }
}

function persistHiddenCharts() {
  if (datasetSignature) {
    persistStoredSet(chartsStorageKey(datasetSignature), hiddenChartIds);
  }
}

function hiddenCountForReport(report) {
  const all = [
    ...(report.executive_summary ?? []),
    ...(report.quality_insights ?? []),
    ...(report.relationship_insights ?? []),
    ...(report.distribution_insights ?? []),
    ...(report.pattern_insights ?? []),
    ...(report.structure_insights ?? []),
    ...(report.aggregation_insights ?? []),
    ...(report.comparative_insights ?? []),
  ];

  return all.filter((insight) => hiddenInsightIds.has(insight.id)).length;
}

function hiddenChartCount(report, consumed) {
  return reportChartIdentities(report).filter(
    (identity) => hiddenChartIds.has(identity) && !consumed.has(identity),
  ).length;
}

function renderHiddenBar(report) {
  const count = hiddenCountForReport(report);

  if (count === 0) {
    hiddenInsightsBar.classList.add("hidden");
    return;
  }

  hiddenInsightsBar.classList.remove("hidden");
  hiddenInsightsCount.textContent =
    count === 1 ? "1 insight hidden" : `${count} insights hidden`;
}

function renderHiddenChartsBar(report, consumed) {
  const count = hiddenChartCount(report, consumed);

  if (count === 0) {
    hiddenChartsBar.classList.add("hidden");
    return;
  }

  hiddenChartsBar.classList.remove("hidden");
  hiddenChartsCount.textContent =
    count === 1 ? "1 chart hidden" : `${count} charts hidden`;
}

function hideInsight(insightId) {
  if (!insightId) {
    return;
  }

  hiddenInsightIds.add(insightId);
  persistHiddenInsights();

  if (currentReport) {
    renderAll(currentReport);
  }
}

function restoreHiddenInsights() {
  hiddenInsightIds.clear();
  persistHiddenInsights();

  if (currentReport) {
    renderAll(currentReport);
  }
}

function hideChart(identity) {
  if (!identity) {
    return;
  }

  hiddenChartIds.add(identity);
  persistHiddenCharts();

  if (currentReport) {
    renderAll(currentReport);
  }
}

function restoreHiddenCharts() {
  hiddenChartIds.clear();
  persistHiddenCharts();

  if (currentReport) {
    renderAll(currentReport);
  }
}

restoreHiddenButton.addEventListener("click", restoreHiddenInsights);
restoreHiddenChartsButton.addEventListener("click", restoreHiddenCharts);

function formatFileSize(bytes) {
  if (bytes >= 1024 * 1024) {
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }

  if (bytes >= 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`;
  }

  return `${bytes} B`;
}

function showSelectedFile(file) {
  selectedFileLabel.textContent = `${file.name} · ${formatFileSize(file.size)}`;
  selectedFileLabel.classList.remove("hidden");
}

function isCsvFile(file) {
  return /\.csv$/i.test(file.name);
}

function useFile(file) {
  if (!file) {
    return;
  }

  if (!isCsvFile(file)) {
    status.textContent = "Please select a CSV file.";
    return;
  }

  showSelectedFile(file);
  analyzeDataset();
}

function pickFile() {
  fileInput.click();
}

dropzone.addEventListener("click", pickFile);

dropzone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    pickFile();
  }
});

fileInput.addEventListener("change", () => useFile(fileInput.files[0]));

for (const type of ["dragenter", "dragover"]) {
  dropzone.addEventListener(type, (event) => {
    event.preventDefault();
    event.stopPropagation();
    dropzone.classList.add("dragover");
  });
}

for (const type of ["dragleave", "drop"]) {
  dropzone.addEventListener(type, (event) => {
    event.preventDefault();
    event.stopPropagation();

    if (type === "dragleave" && dropzone.contains(event.relatedTarget)) {
      return;
    }

    dropzone.classList.remove("dragover");
  });
}

dropzone.addEventListener("drop", (event) => {
  const file = event.dataTransfer?.files?.[0];

  if (!file) {
    return;
  }

  if (!isCsvFile(file)) {
    status.textContent = "Please select a CSV file.";
    return;
  }

  // Copiamos el archivo al input para que el flujo de análisis
  // (fileInput.files[0]) funcione igual que con el explorador.
  const transfer = new DataTransfer();
  transfer.items.add(file);
  fileInput.files = transfer.files;

  useFile(file);
});

// Evita que el navegador abra el archivo si se suelta fuera del dropzone.
document.addEventListener("dragover", (event) => event.preventDefault());
document.addEventListener("drop", (event) => event.preventDefault());

const downloadPdfButton = document.getElementById("download-pdf-button");
const downloadReportButton = document.getElementById("download-report-button");
const downloadCsvButton = document.getElementById("download-csv-button");

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");

  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();

  URL.revokeObjectURL(url);
}

function rowsToCsv(rows) {
  if (!rows || rows.length === 0) {
    return "";
  }

  const headers = Object.keys(rows[0]);

  const escapeCell = (value) => {
    if (value === null || value === undefined) {
      return "";
    }

    const text = String(value);

    return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
  };

  const lines = [headers.join(",")];

  for (const row of rows) {
    lines.push(headers.map((header) => escapeCell(row[header])).join(","));
  }

  return lines.join("\n");
}

downloadReportButton.addEventListener("click", async () => {
  const file = fileInput.files[0];

  if (!file) {
    status.textContent = "Please select a CSV file.";
    return;
  }

  status.textContent = "Generating HTML report...";
  downloadReportButton.disabled = true;

  try {
    const blob = await fetchHtmlReport(
      file,
      [...hiddenInsightIds],
      [...hiddenChartIds],
    );

    downloadBlob(blob, "explain-my-data-report.html");

    status.textContent = "Report downloaded.";
  } catch (error) {
    status.textContent = `Error: ${error.message}`;
  } finally {
    downloadReportButton.disabled = false;
  }
});

downloadPdfButton.addEventListener("click", async () => {
  const file = fileInput.files[0];

  if (!file) {
    status.textContent = "Please select a CSV file.";
    return;
  }

  // Abrimos la ventana ANTES del await para no perder el gesto
  // del usuario (los bloqueadores de popups lo rechazarían luego).
  const printWindow = window.open("", "_blank");

  status.textContent = "Generating report for print...";
  downloadPdfButton.disabled = true;

  try {
    const blob = await fetchHtmlReport(
      file,
      [...hiddenInsightIds],
      [...hiddenChartIds],
    );
    const url = URL.createObjectURL(blob);

    if (!printWindow) {
      status.textContent =
        "Popup blocked. Allow popups for this site, or download the HTML report and open it in a new tab.";
      URL.revokeObjectURL(url);
    } else {
      printWindow.location = `${url}#print`;
      status.textContent =
        'Report opened — choose "Save as PDF" in the print dialog.';
      setTimeout(() => URL.revokeObjectURL(url), 60_000);
    }
  } catch (error) {
    if (printWindow) {
      printWindow.close();
    }

    status.textContent = `Error: ${error.message}`;
  } finally {
    downloadPdfButton.disabled = false;
  }
});

downloadCsvButton.addEventListener("click", () => {
  if (!cleanedRows) {
    status.textContent = "Clean a dataset first to download the cleaned CSV.";
    return;
  }

  const csv = rowsToCsv(cleanedRows);

  downloadBlob(new Blob([csv], { type: "text/csv;charset=utf-8" }), "cleaned.csv");

  status.textContent = "Cleaned CSV downloaded.";
});

analyzeButton.addEventListener("click", analyzeDataset);

undoCleaningButton.addEventListener("click", undoCleaning);

const cleaningOverlay = document.getElementById("cleaning-overlay");
const closeCleaningButton = document.getElementById("close-cleaning-button");

toggleCleaningButton.addEventListener("click", () => {
  openCleaningDrawer({
    cleaningOptions,
    cleaningOverlay,
    toggleCleaningButton,
  });
});

closeCleaningButton.addEventListener("click", () => {
  closeCleaningDrawer({
    cleaningOptions,
    cleaningOverlay,
    toggleCleaningButton,
  });
});

const cancelCleaningButton = document.getElementById(
  "cancel-cleaning-button",
);

cancelCleaningButton.addEventListener("click", () => {
  closeCleaningDrawer({
    cleaningOptions,
    cleaningOverlay,
    toggleCleaningButton,
  });
});

cleaningOverlay.addEventListener("click", () => {
  closeCleaningDrawer({
    cleaningOptions,
    cleaningOverlay,
    toggleCleaningButton,
  });
});

closeCleaningDrawer({
  cleaningOptions,
  cleaningOverlay,
  toggleCleaningButton,
});

cleanButton.addEventListener("click", cleanDataset);

function renderAll(report) {
  renderOverview(report, overview);
  renderQualityInsights(report, quality);
  renderCleaningIssues(report);
  renderExtremeColumnOptions(report, extremeColumns);
  renderNumericRanges(report, numericRanges);
  renderPatterns(report, patterns);
  renderHiddenBar(report);

  const consumed = renderInsightsSection(report, insights, {
    hiddenIds: hiddenInsightIds,
    onHide: hideInsight,
  });

  renderHiddenChartsBar(report, consumed);

  renderExtraCharts(report, charts, consumed, {
    hiddenCharts: hiddenChartIds,
    onHideChart: hideChart,
  });

  renderReport(report, rawReport);
}

async function analyzeDataset() {
  const file = fileInput.files[0];

  if (!file) {
    status.textContent =
      "No file selected — drop a CSV on the upload area or click to browse.";
    return;
  }

  status.textContent = "Analyzing dataset...";
  analyzeButton.disabled = true;

  try {
    const report = await analyzeFile(file);

    originalReport = report;
    cleanedReport = null;
    currentReport = report;
    cleanedRows = null;
    downloadCsvButton.disabled = true;
    undoCleaningButton.classList.add("hidden");

    datasetSignature = await computeDatasetSignature(file);
    loadHiddenStores(datasetSignature);

    renderAll(report);

    status.textContent = "Analysis completed.";
    results.classList.remove("hidden");
  } catch (error) {
    status.textContent = `Error: ${error.message}`;
  } finally {
    analyzeButton.disabled = false;
  }
}
async function cleanDataset() {
  const file = fileInput.files[0];

  if (!file) {
    status.textContent = "Please select a CSV file.";
    return;
  }

  status.textContent = "Cleaning dataset...";
  cleanButton.disabled = true;

  const selectedOutlierColumns = [
    ...document.querySelectorAll('input[name="remove-outlier-column"]:checked'),
  ].map((input) => input.value);

  const ranges = {};

  document.querySelectorAll(".numeric-range").forEach((rangeElement) => {
    const columnName = rangeElement.dataset.column;

    const minInput = rangeElement.querySelector(".range-min");
    const maxInput = rangeElement.querySelector(".range-max");

    ranges[columnName] = [Number(minInput.value), Number(maxInput.value)];
  });
  const options = {
    removeDuplicates: removeDuplicates.checked,
    numericMissingStrategy: numericMissingStrategy.value,
    categoricalMissingStrategy: categoricalMissingStrategy.value,
    replaceInvalidValues: replaceInvalidValues.checked,
    convertTypeMismatches: convertTypeMismatches.checked,
    outlierStrategy: outlierStrategy.value,
    outlierColumns: selectedOutlierColumns,
    numericRanges: ranges,
  };

  try {
    const result = await cleanFile(file, options);

    cleanedReport = result.analysis;
    currentReport = cleanedReport;
    cleanedRows = result.data;
    downloadCsvButton.disabled = !cleanedRows || cleanedRows.length === 0;

    renderCleaningReport(result.report, cleaning);

    renderAll(currentReport);

    undoCleaningButton.classList.remove("hidden");

    closeCleaningDrawer({
      cleaningOptions,
      cleaningOverlay,
      toggleCleaningButton,
    });

    status.textContent = "Cleaning completed.";
    results.classList.remove("hidden");
  } catch (error) {
    status.textContent = `Error: ${error.message}`;
  } finally {
    cleanButton.disabled = false;
  }
}

async function undoCleaning() {
  const file = fileInput.files[0];

  if (!file) {
    status.textContent = "Please select a CSV file.";
    return;
  }

  status.textContent = "Restoring original dataset...";

  try {
    const report = await analyzeFile(file);

    currentReport = report;
    cleanedReport = null;
    cleanedRows = null;
    downloadCsvButton.disabled = true;
    undoCleaningButton.classList.add("hidden");

    renderAll(currentReport);

    status.textContent = "Original dataset restored.";
  } catch (error) {
    status.textContent = `Error: ${error.message}`;
  }
}

window.addEventListener("resize", () => {
  document.querySelectorAll(".plotly-chart").forEach((chart) => {
    Plotly.Plots.resize(chart);
  });
});
