const API_URL = "http://127.0.0.1:8000";

async function requestReport(endpoint, formData, errorMessage) {
  const response = await fetch(`${API_URL}${endpoint}`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || errorMessage);
  }

  return response.json();
}

export async function analyzeFile(file) {
  const formData = new FormData();
  formData.append("file", file);

  return requestReport("/analyze", formData, "Analysis failed.");
}

export async function cleanFile(file, options) {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("remove_duplicates", options.removeDuplicates);
  formData.append("missing_numeric_strategy", options.numericMissingStrategy);
  formData.append(
    "missing_categorical_strategy",
    options.categoricalMissingStrategy,
  );
  formData.append("replace_invalid_values", options.replaceInvalidValues);
  formData.append("convert_type_mismatches", options.convertTypeMismatches);
  formData.append("outlier_strategy", options.outlierStrategy);
  formData.append("outlier_columns", options.outlierColumns.join(","));
  formData.append("numeric_ranges", JSON.stringify(options.numericRanges));

  return requestReport("/clean", formData, "Cleaning failed.");
}

export async function fetchHtmlReport(file, hiddenIds = [], hiddenCharts = []) {
  const formData = new FormData();
  formData.append("file", file);

  const params = new URLSearchParams();

  if (hiddenIds.length > 0) {
    params.set("hide", hiddenIds.join(","));
  }

  if (hiddenCharts.length > 0) {
    params.set("hide_charts", hiddenCharts.join(","));
  }

  const query = params.toString() ? `?${params.toString()}` : "";

  const response = await fetch(`${API_URL}/report${query}`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || "Report download failed.");
  }

  return response.blob();
}
