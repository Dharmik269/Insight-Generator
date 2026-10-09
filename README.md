# Assignment 4 — Automated Insight Generation

A general-purpose Python dashboard that loads the **CSV sample supplied in the assignment PDF**, validates it, detects trends and outliers, computes Pearson correlations, and generates structured human-readable insights.


## Features

- Pandas CSV loading, preview, data types, and missing-value counts
- Live filters for district, month, and indicator
- Configurable significant trend threshold (default 10%)
- Outlier detection using IQR or Z-score, with configurable sensitivity
- Pearson correlation matrix and configurable `|r|` threshold (default 0.70)
- Data-driven insight records with IDs, type, indicator, entity, period, metric/change, severity, and explanation
- Severity count bar chart, correlation heatmap, and per-district line chart
- Downloadable insights CSV/JSON, correlation matrix CSV, and filtered data CSV

## Run locally

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

The app opens in your browser. Use the sidebar to change filters and thresholds.

## Output

- `insights.csv` — generated on demand from current filters and settings
- `insights.json` — generated on demand from current filters and settings
- `correlation_matrix.csv` — generated on demand from current filters and settings

These files are downloaded from the dashboard rather than saved automatically.

## Interpretation notes

- Trend insights compare available consecutive month rows within each district.
- Outliers are computed independently for each selected indicator over the currently filtered data.
- Correlation is computed across selected numerical indicators and filtered rows. Correlation does not imply causation.
- The assignment's sample has only 2 months × 6 districts = 12 rows. As the handout notes, correlations are fragile with such a small sample; findings should be interpreted cautiously.
- The app uses generic templates. District names and numeric findings are read from the CSV, not embedded as district-specific narratives.
