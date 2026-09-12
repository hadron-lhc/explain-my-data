# Explain My Data

> An automated data analysis tool that explores, diagnoses, explains, and improves CSV datasets.

**Explain My Data** is a Python-based data analysis application designed to automatically extract as much useful information as possible from an uploaded CSV dataset.

The goal is not simply to generate statistics and charts.

The system attempts to understand the structure and quality of the data, identify relevant patterns and relationships, detect potential problems and anomalies, determine which statistical analyses are appropriate, identify potential predictive targets, recommend machine learning approaches when justified, and propose ways to clean or improve the dataset.

The guiding principle is:

> **Analyze everything that can be meaningfully inferred from the available data.**

---

## Goals

Given a CSV dataset, Explain My Data should be able to:

- Understand the structure of the dataset.
- Automatically infer variable types and roles.
- Detect data-quality problems.
- Calculate relevant descriptive statistics.
- Analyze distributions.
- Identify relationships between variables.
- Automatically select useful visualizations.
- Detect potential outliers and anomalies.
- Analyze temporal patterns when dates are available.
- Perform appropriate statistical tests.
- Identify potential target variables.
- Determine whether a predictive problem may exist.
- Recommend appropriate machine learning models.
- Evaluate candidate models when prediction is appropriate.
- Explain important findings in human-readable language.
- Suggest possible data-cleaning operations.
- Allow the user to review and apply selected cleaning operations.
- Produce a cleaned version of the dataset.
- Preserve the original dataset.

The system should adapt its analysis to the dataset instead of blindly applying every available technique.

---

# How it works

The general workflow is:

```text
CSV
 │
 ▼
Data Understanding
 │
 ▼
Data Quality
 │
 ▼
Exploratory Analysis
 │
 ▼
Relationships & Patterns
 │
 ├── Statistical Analysis
 ├── Anomaly Detection
 ├── Temporal Analysis
 │
 ▼
Target Detection
 │
 ├── No suitable target
 │
 └── Potential target
       │
       ▼
   Predictive Analysis
       │
       ▼
   Model Evaluation
 │
 ▼
Insights & Explanations
 │
 ▼
Cleaning Recommendations
 │
 ▼
Optional Data Cleaning
 │
 ▼
Cleaned Dataset
```

Not every stage needs to run for every dataset.

The available analyses depend on the characteristics of the data.

---

# What the system analyzes

## 1. Dataset structure

The system should identify:

- Number of rows.
- Number of columns.
- Column names.
- Data types.
- Possible identifiers.
- Numerical variables.
- Categorical variables.
- Boolean variables.
- Datetime variables.
- Text variables.
- Cardinality.
- Number of unique values.
- Constant columns.
- Near-constant columns.
- Potential target variables.
- Suspicious or incorrectly inferred types.

Example:

```text
customer_id    identifier
age            numerical
country        categorical
signup_date    datetime
churn          binary target candidate
```

---

## 2. Data quality

The system should detect potential quality problems such as:

- Missing values.
- Duplicate rows.
- Duplicate columns.
- Empty strings.
- Inconsistent categorical values.
- Suspicious data types.
- Invalid dates.
- Impossible or suspicious numerical values when they can be identified.
- Extremely high or low cardinality.
- Potential identifiers being treated as features.
- Columns containing almost no information.

Example:

```text
⚠ 31 duplicated rows detected.

⚠ 2.4% of `income` values are missing.

⚠ `country` contains potentially inconsistent category labels.

⚠ `signup_date` contains values that could not be parsed as dates.
```

The system should distinguish between a detected problem and an assumption.

It should not automatically declare that a value is incorrect unless there is sufficient evidence.

---

# 3. Descriptive statistics

For numerical variables, the system may calculate:

- Mean.
- Median.
- Mode when appropriate.
- Minimum.
- Maximum.
- Range.
- Variance.
- Standard deviation.
- Quantiles.
- Interquartile range.
- Skewness.
- Kurtosis.
- Other relevant statistics when justified.

For categorical variables:

- Frequency.
- Percentage.
- Number of unique values.
- Most frequent categories.
- Category concentration.

The application should prioritize statistics that help explain the dataset instead of displaying every possible statistic without context.

---

# 4. Distributions

The system should analyze the distribution of numerical variables when appropriate.

Possible analyses include:

- Histograms.
- Box plots.
- Distribution shape.
- Skewness.
- Heavy tails.
- Potential multimodality.
- Concentration.
- Extreme values.

Example insight:

> `income` is strongly right-skewed. The mean is considerably higher than the median, suggesting that a relatively small number of high-income observations are influencing the average.

---

# 5. Relationships between variables

The system should identify potentially meaningful relationships between variables.

## Numerical ↔ numerical

Possible analyses:

- Pearson correlation.
- Spearman correlation.
- Covariance.
- Scatter plots.
- Relationship strength.
- Relationship direction.

Example:

```text
income ↔ house_price

Pearson correlation: 0.74
Spearman correlation: 0.69

Strong positive association detected.
```

Correlation should never automatically be interpreted as causation.

## Categorical ↔ numerical

Depending on the data, the system may analyze:

- Group means.
- Group medians.
- Distributions by category.
- Differences between groups.
- Effect sizes.
- ANOVA.
- Non-parametric alternatives.

Example:

```text
Average house price by neighborhood
```

## Categorical ↔ categorical

Possible analyses include:

- Contingency tables.
- Chi-square tests.
- Cramér's V.
- Category associations.

The system should select appropriate techniques according to the characteristics of the variables and the available data.

---

# 6. Automatic visualizations

The application should not generate every possible graph.

Instead, it should identify relationships or patterns that are potentially useful and recommend or generate appropriate visualizations.

Examples:

```text
Numerical ↔ Numerical
→ Scatter plot

Category ↔ Numerical
→ Box plot / bar chart

Datetime ↔ Numerical
→ Time-series chart

Categorical distribution
→ Bar chart

Numerical distribution
→ Histogram / box plot
```

The goal is to answer:

> **Which visualization helps the user understand this dataset?**

rather than:

> **Which charts can we generate?**

---

# 7. Outliers and anomalies

The system should identify statistically unusual observations using appropriate methods.

Possible techniques include:

- IQR.
- Z-score.
- Modified Z-score.
- Isolation Forest.
- Local Outlier Factor.
- Other methods when justified.

An important distinction is:

```text
Statistical outlier
        ≠
Data error
        ≠
Business anomaly
```

The system should therefore report unusual observations as potential anomalies rather than automatically declaring them incorrect.

Example:

> Observation 4821 is unusually high compared with the distribution of `order_value`.

---

# 8. Temporal analysis

If the dataset contains a meaningful datetime variable, the system should attempt to identify:

- Time granularity.
- Trends.
- Growth or decline.
- Seasonality.
- Missing periods.
- Sudden changes.
- Temporal anomalies.
- Rolling statistics.
- Period-over-period changes.

Example:

> Sales increased by approximately 18% over the observed period.

The system should adapt the analysis to the available temporal resolution.

---

# 9. Statistical inference

When the dataset and analysis allow it, the system may perform statistical inference.

Possible techniques include:

- Confidence intervals.
- Hypothesis tests.
- P-values.
- Effect sizes.
- ANOVA.
- Chi-square tests.
- Non-parametric tests.
- Distribution tests.

The system should consider relevant assumptions and sample characteristics before applying a test.

Statistical significance should not automatically be interpreted as practical significance.

---

# 10. Target detection

The system should attempt to identify columns that may represent a predictive target.

A potential target may be:

- Numerical.
- Binary.
- Multiclass categorical.

The system should consider factors such as:

- Data type.
- Cardinality.
- Variability.
- Missingness.
- Whether the column appears to be an identifier.
- Whether other variables could reasonably explain it.

Example:

```text
Potential target detected:

churn

Problem type:
Binary classification

Confidence:
High
```

The user should always be able to manually select a different target or disable predictive analysis.

---

# 11. Predictive analysis

If a suitable target is identified or selected, the system should determine whether a predictive modeling problem is appropriate.

Possible problem types:

- Binary classification.
- Multiclass classification.
- Regression.

The system should exclude unsuitable variables such as identifiers when appropriate.

Example:

```text
Target:
house_price

Problem:
Regression

Potential features:

✓ income
✓ age
✓ experience
✓ location

Excluded:

✗ customer_id
```

---

# 12. Model recommendation and evaluation

When predictive modeling is appropriate, the system should recommend a reasonable set of candidate models.

For classification, possible models include:

- Logistic Regression.
- Random Forest.
- Gradient Boosting.
- Other suitable models.

For regression:

- Linear Regression.
- Random Forest Regressor.
- Gradient Boosting Regressor.
- Other suitable models.

Models should be evaluated using appropriate metrics and cross-validation.

Classification metrics may include:

- Accuracy.
- Precision.
- Recall.
- F1-score.
- ROC-AUC.
- Confusion matrix.

Regression metrics may include:

- MAE.
- RMSE.
- R².
- Residual analysis.

The application should not select models solely because they are more complex.

---

# 13. Feature importance and interpretation

When appropriate, the system should help explain which variables contribute most to the predictive model.

Possible techniques include:

- Model-specific feature importance.
- Permutation importance.
- Other interpretable methods.

Example:

```text
Most important features

income          ███████████
experience      ███████
age             ████
```

Model importance should not automatically be interpreted as causal importance.

---

# 14. Cleaning recommendations

The system should identify issues that may be fixable and propose possible solutions.

Examples:

```text
31 duplicate rows detected

Possible action:
Remove duplicate rows
```

```text
17 missing values in `income`

Possible actions:

- Keep missing values.
- Remove affected rows.
- Fill with median.
- Fill using a group-based strategy.
```

```text
Potentially inconsistent categories detected:

Argentina
argentina
ARG
Arg.

Possible action:
Review and normalize categories.
```

```text
3 invalid datetime values detected.

Possible actions:

- Review values.
- Remove affected rows.
- Attempt alternative parsing.
```

The original dataset should never be silently modified.

---

# 15. Cleaning pipeline

The user should be able to select proposed transformations and apply them.

Example:

```text
Selected actions:

✓ Remove duplicate rows
✓ Parse signup_date
✓ Normalize country labels
✓ Fill missing income values with median

[ Apply changes ]
```

The application should produce:

```text
Original dataset
       │
       ▼
Cleaning pipeline
       │
       ▼
Cleaned dataset
```

The system should also provide a cleaning log describing what was changed.

Example:

```text
Cleaning log

✓ Removed 31 duplicate rows.
✓ Parsed `signup_date` as datetime.
✓ Normalized 8 category labels.
✓ Filled 17 missing `income` values using the selected strategy.
```

---

# 16. Insights

The final objective is not just to show calculations.

The application should transform relevant analytical results into understandable findings.

Instead of:

```text
Pearson r = 0.74
```

the system should provide:

> `income` and `house_price` show a strong positive association in this dataset (r = 0.74). Higher-income observations tend to correspond to higher house prices.

And provide the supporting evidence when useful.

Each insight should ideally contain:

- Finding.
- Evidence.
- Relevant metric.
- Visualization when appropriate.
- Caveats or limitations.

---

# Analysis principles

Explain My Data follows several principles.

### 1. Analyze as much as possible, but only when meaningful

The system should be comprehensive without blindly applying every statistical technique.

### 2. Adapt to the dataset

Different datasets require different analyses.

A personal expense dataset should not be analyzed in exactly the same way as a medical study or a customer churn dataset.

### 3. Do not silently modify user data

Analysis and cleaning are separate operations.

### 4. Distinguish evidence from assumptions

The application should communicate uncertainty when an inference is not certain.

### 5. Correlation does not imply causation

Associations should not automatically be presented as causal relationships.

### 6. Statistical significance does not necessarily imply practical significance

A statistically significant result may still have little practical importance.

### 7. Outliers are not automatically errors

An unusual observation may be valid and meaningful.

### 8. Machine learning is optional

A dataset does not need a predictive model simply because machine learning is available.

### 9. Explanations should be based on the analysis

The system should explain analytical results rather than inventing conclusions.

---

# Architecture

The initial architecture is intentionally simple.

```text
Streamlit UI
      │
      ▼
Analysis Orchestrator
      │
      ├── Profiling
      ├── Data Quality
      ├── Statistics
      ├── Relationships
      ├── Anomalies
      ├── Temporal Analysis
      ├── Target Detection
      ├── Machine Learning
      ├── Insights
      └── Cleaning
```

The analysis engine should remain independent from the Streamlit interface.

The UI should be responsible for:

- File upload.
- User interaction.
- Displaying results.
- Selecting cleaning operations.
- Selecting targets/models.
- Downloading results.

The analysis core should be responsible for:

- Calculations.
- Detection.
- Statistical analysis.
- Model training/evaluation.
- Producing structured results.

---

# Technology

Initial technology stack:

| Component                  | Technology                |
| -------------------------- | ------------------------- |
| Language                   | Python 3.12+              |
| Data manipulation          | Pandas                    |
| Numerical computing        | NumPy                     |
| Statistics                 | SciPy                     |
| Machine learning           | scikit-learn              |
| Visualization              | Plotly                    |
| Web application            | Streamlit                 |
| Testing                    | pytest                    |
| Linting / formatting       | Ruff                      |
| Type checking              | Pyright                   |
| Environment / dependencies | uv                        |
| Version control            | Git + GitHub              |
| Deployment                 | Streamlit Community Cloud |

The project intentionally avoids unnecessary infrastructure in its initial versions.

Potential future technologies should only be introduced when they solve a real problem.

---

# Project structure

Initial structure:

```text
explain-my-data/
│
├── app.py
│
├── src/
│   ├── profiling.py
│   ├── quality.py
│   ├── statistics.py
│   ├── relationships.py
│   ├── insights.py
│   └── models.py
│
├── tests/
│   ├── test_profiling.py
│   └── test_statistics.py
│
├── data/
│   └── examples/
│
├── .streamlit/
│   └── config.toml
│
├── pyproject.toml
├── uv.lock
├── README.md
├── .gitignore
└── LICENSE
```

This structure is expected to evolve as the project grows.

---

# Development roadmap

## V1 — Understand

- CSV upload.
- Dataset validation.
- Dataset overview.
- Automatic type detection.
- Basic profiling.
- Missing values.
- Duplicates.
- Unique values.
- Basic descriptive statistics.
- Initial visualizations.

## V2 — Explore

- Distribution analysis.
- Automatic relationship detection.
- Correlations.
- Categorical/numerical relationships.
- Automatic visualization selection.
- Insight generation.
- More advanced data-quality detection.

## V3 — Diagnose

- Outlier detection.
- Anomaly detection.
- Temporal analysis.
- Statistical tests.
- Confidence intervals.
- Effect sizes.
- More sophisticated relationship analysis.

## V4 — Predict

- Target detection.
- Problem-type detection.
- Feature selection.
- Classification.
- Regression.
- Cross-validation.
- Model comparison.
- Metrics.
- Feature importance.
- Model interpretation.

## V5 — Improve

- Cleaning recommendations.
- Interactive cleaning decisions.
- Reproducible cleaning pipeline.
- Cleaning log.
- Cleaned CSV export.

## V6 — Report

- Complete analytical report.
- Executive summary.
- Key findings.
- Data-quality summary.
- Visualizations.
- Statistical evidence.
- Predictive results.
- Cleaning summary.

## Future possibilities

Potential future features may include:

- Additional anomaly-detection methods.
- Dimensionality reduction.
- Clustering.
- More advanced feature engineering.
- Explainable AI.
- Natural-language interaction.
- Optional local or external LLM integration.

These features should only be added when they provide meaningful value.

---

# Testing

The analytical engine should be tested independently from the Streamlit interface.

Tests should cover:

- Type detection.
- Missing-value detection.
- Duplicate detection.
- Statistical calculations.
- Correlation calculations.
- Outlier detection.
- Target detection.
- Cleaning transformations.
- Model evaluation.

Small controlled datasets should be used to verify analytical behavior.

---

# Example use cases

## Personal expenses

```text
date,category,description,amount
2026-01-01,food,lunch,8500
2026-01-02,transport,bus,1200
...
```

Potential analysis:

- Spending by category.
- Spending over time.
- Average daily spending.
- Largest expenses.
- Distribution of expenses.
- Unusual transactions.
- Monthly trends.
- Category concentration.

No predictive model may be necessary.

---

## Customer churn

```text
customer_id,age,income,contract_type,support_tickets,churn
...
```

Potential analysis:

- Data quality.
- Feature distributions.
- Feature relationships.
- Relationship with churn.
- Potential statistical differences.
- Target detection.
- Classification.
- Model comparison.
- Feature importance.

---

## Real estate

```text
area,rooms,bathrooms,location,age,price
...
```

Potential analysis:

- Price distribution.
- Price by location.
- Feature relationships.
- Correlations.
- Outliers.
- Potential target detection.
- Regression models.
- Feature importance.

---

# Project philosophy

Explain My Data is built around a simple idea:

> **A dataset should not require the user to already know what to look for.**

The application should help answer:

```text
What is in my data?

Is my data reliable?

What problems does it have?

What patterns exist?

Which variables are related?

What are the unusual observations?

Is there a meaningful target?

Can I predict it?

What should I investigate?

What can I safely fix?

What did the analysis actually tell me?
```

The final product should feel less like a collection of statistical functions and more like an **automated first-pass data analyst**.
