# Data Explorer

Data Explorer is a Streamlit application for turning an uploaded dataset into a clear exploratory data analysis (EDA) report. It accepts CSV, Excel, and JSON files, checks data quality, explains distributions and relationships, finds unusual values, recommends suitable charts, and exports a stakeholder-ready PDF report.

It is designed for analysts and non-technical users who need a useful first assessment of a dataset without writing code.

## What the project does

Upload a tabular dataset and Data Explorer will:

- Read CSV, Excel (`.xlsx`, `.xls`), and JSON files, including common nested JSON layouts and JSON files with trailing commas.
- Infer practical number, date/time, and category data types.
- Profile fields, missing values, repeated records, memory use, and sample values.
- Generate descriptive statistics and plain-language distribution interpretations.
- Identify unusual values per field and unusual records across several numeric fields.
- Display interactive charts for distributions, categories, relationships, group comparisons, and correlations.
- Calculate a transparent Data Health Score from 0 to 100.
- Run data-quality checks and allow a user-defined validation rule.
- Recommend a suitable chart based on selected field types and quality characteristics.
- Export the current findings as a PDF report.

## Features

### Data ingestion and profiling

- CSV encoding fallback for common real-world files.
- Excel and JSON loading.
- Safe JSON handling for nested record containers: `data`, `records`, `items`, and `results`.
- Automatic type inference for numeric and date-like columns.
- Dataset metadata, preview, field profile, memory use, and prepared CSV download.

### Data quality

- Missing-value counts and percentages.
- Missing-data chart with readable labels and tooltips.
- Exact duplicate-record detection.
- Built-in validation checks for negative values, non-positive prices, percentage ranges, date order, and duplicate IDs.
- Optional custom checks: not blank, unique, minimum value, or value range.

### Statistics and unusual-value analysis

- Count, average, median, spread, minimum, maximum, skewness, kurtosis, and normality signal.
- Missing-value count and percentage shown beside numeric summaries.
- Safe interpretation for special cases:
  - **Constant / No variation** for zero-variance fields.
  - **Insufficient data for distribution analysis** for sparse fields.
- IQR-based unusual-value detection per numeric field.
- Isolation Forest-based unusual-record detection across multiple numeric fields when PyOD is installed.

### Visual exploration

- Numeric distributions and box plots.
- Category count charts, limited to the most common values when a field is high-cardinality.
- Numeric-to-numeric scatter plots with an optional overall trend.
- Numeric-to-category group comparisons.
- Compact multi-field relationship maps.
- Pearson and Spearman correlation views with strong-relationship summaries.
- Constant numeric fields are excluded from misleading scatter plots and shown as `N/A` in correlation results.
- URL-like and ID-like category fields are detected and not plotted as unreadable category charts.

### Data Health Score

The health score is a 0–100 composite with an explainable weighted breakdown:

| Component | Weight | Meaning |
| --- | ---: | --- |
| Complete data | 35% | Fewer missing values increases the score. |
| Duplicate-free records | 20% | Fewer repeated rows increases the score. |
| Typical values | 20% | Lower IQR unusual-value rates increase the score. |
| Quality checks | 25% | More passed enabled checks increases the score. |

The dashboard shows the score, a radar chart, and each component value.

### Smart chart recommendation

The recommendation engine evaluates field type, unique-value count, variation, missing usable values, category cardinality, and URL/ID-like patterns.

| Selected fields | Recommended view |
| --- | --- |
| One varying numeric field | Histogram |
| One low-cardinality category | Bar chart |
| Two varying numeric fields | Scatter plot |
| One numeric + one low-cardinality category | Box plot or violin plot |
| Two low-cardinality categories | Count heatmap or stacked bar chart |
| Constant numeric, ID-like, or URL-like field | No chart; an explanation is shown |
| High-cardinality category | Top-N view or a recommendation to choose another field |

### PDF export

The PDF report includes dataset metadata, the Data Health Score, quality-check results, field profiling, numerical summaries, unusual-value information, and relationship charts.

WeasyPrint is used when its system libraries are available. On Windows environments where those native libraries are unavailable, the app automatically uses a Matplotlib PDF fallback so PDF export remains available.

## Tech stack

| Area | Libraries |
| --- | --- |
| Application | Streamlit |
| Data manipulation | pandas, NumPy |
| Statistics | SciPy |
| Outlier detection | PyOD Isolation Forest |
| Interactive charts | Plotly |
| Static charts / report charts | Matplotlib, Seaborn |
| Quality rules | Great Expectations, with a native fallback for compatible checks |
| PDF reports | Jinja2, WeasyPrint, Matplotlib fallback |

## Project structure

```text
app.py                         # Streamlit UI and feature integration
eda_tool/
  loaders.py                   # CSV, Excel, and resilient JSON ingestion
  profiling.py                 # Type inference, profiles, field suitability checks
  analysis.py                  # Statistics, correlations, and outliers
  visualizations.py            # Main report charts
  health_score.py              # Data Health Score and radar chart
  recommendations.py           # Smart chart recommendations and rendering
  quality_rules.py             # Built-in and custom quality checks
  pdf_report.py                # PDF report generation and fallback
requirements.txt               # Python dependencies
```

## How to run it

### 1. Use a supported Python version

Python **3.12 or 3.13** is recommended for the smoothest scientific-Python installation. Python 3.14 may attempt to compile older NumPy versions from source if compatible wheels are unavailable.

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If Python 3.13 is not installed:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Start the app

```powershell
streamlit run app.py
```

Open the local URL printed by Streamlit, then upload a CSV, Excel, or JSON dataset.


## Evaluation and metrics

This is an exploratory analysis tool, not a predictive model. Evaluation focuses on correct data handling and interpretable outputs.

| Area | Metric / expected behavior |
| --- | --- |
| File ingestion | CSV, Excel, standard JSON, nested JSON, and trailing-comma JSON load with a clear failure message for invalid files. |
| Missing values | Missing count and percentage match `pandas.isna()` results. |
| Duplicate records | Duplicate count matches `DataFrame.duplicated()`. |
| Constant numeric fields | One unique non-null value or zero standard deviation is labelled `Constant / No variation`. |
| Correlation | Constant-field correlations are `N/A`, not treated as a numerical relationship. |
| Scatter plots | No scatter chart is created if either selected number field has no variation. |
| Category suitability | URL/ID-like fields are not plotted as categories; high-cardinality fields are limited to Top-N or warned about. |
| Health score | Score remains bounded from 0 to 100 and exposes component scores and weights. |
| PDF export | Generated file starts with a valid PDF header and includes the current report inputs. |

## Future enhancements

- User-selectable primary keys and semantic field roles instead of name-based detection alone.
- Saved and shareable quality-rule sets.
- Data-cleaning actions with preview, undo, and export history.
- More JSON-flattening controls for deeply nested structures.
- Downloadable HTML and Excel report formats.
- Configurable chart themes and organization branding for exported reports.
- More advanced statistical tests and automatically written insight summaries.
- Dataset comparison across uploads and report history.
- Unit tests and CI/CD workflow for regression testing and deployment.
- Docker deployment configuration and cloud-hosting instructions.

## Privacy note

Uploaded datasets are processed in memory by the application and are not intentionally written to disk by the app itself. For sensitive data, run the project in an environment you control.
