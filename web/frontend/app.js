import { analyzeFile, cleanFile } from "./api.js";

import {
  closeCleaningDrawer,
  openCleaningDrawer,
  renderExtremeColumnOptions,
  renderNumericRanges,
} from "./cleaning-ui.js";

import {
  renderCleaningReport,
  renderOverview,
  renderQualityInsights,
  renderReport,
  renderStatistics,
} from "./report-ui.js";

import { renderRelationshipInsights } from "./charts.js";

const fileInput = document.getElementById("file-input");
const analyzeButton = document.getElementById("analyze-button");
const status = document.getElementById("status");
const results = document.getElementById("results");

const undoCleaningButton = document.getElementById("undo-cleaning-button");

const overview = document.getElementById("overview");
const quality = document.getElementById("quality");
const statistics = document.getElementById("statistics");
const relationships = document.getElementById("relationships");
const charts = document.getElementById("charts");
const rawReport = document.getElementById("raw-report");

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

async function analyzeDataset() {
  const file = fileInput.files[0];

  if (!file) {
    status.textContent = "Please select a CSV file.";
    return;
  }

  status.textContent = "Analyzing dataset...";
  analyzeButton.disabled = true;

  try {
    const report = await analyzeFile(file);

    originalReport = report;
    cleanedReport = null;
    currentReport = report;

    renderOverview(report, overview);
    renderQualityInsights(report, quality);
    renderStatistics(report, statistics);
    renderExtremeColumnOptions(report, extremeColumns);
    renderNumericRanges(report, numericRanges);
    renderRelationshipInsights(report, relationships);
    renderReport(report, rawReport);

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

    renderCleaningReport(result.report, cleaning);

    renderOverview(currentReport, overview);
    renderQualityInsights(currentReport, quality);
    renderStatistics(currentReport, statistics);
    renderExtremeColumnOptions(currentReport, extremeColumns);
    renderNumericRanges(currentReport, numericRanges);
    renderRelationshipInsights(currentReport, relationships);
    renderReport(currentReport, rawReport);

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

    renderOverview(currentReport, overview);
    renderQualityInsights(currentReport, quality);
    renderStatistics(currentReport, statistics);
    renderExtremeColumnOptions(currentReport, extremeColumns);
    renderNumericRanges(currentReport, numericRanges);
    renderRelationshipInsights(currentReport, relationships);
    renderReport(currentReport, rawReport);

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
