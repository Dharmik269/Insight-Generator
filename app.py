from pathlib import Path
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Automated Healthcare Insight Engine", page_icon="📊", layout="wide")
st.title("Automated Healthcare Insight Generation")
st.caption("District-level healthcare performance • data-driven trends, outliers, correlations, and readable insights")

DATA_PATH = Path(__file__).parent / "healthcare_performance.csv"
EXPECTED_COLUMNS = ["month", "district", "anc_coverage", "institutional_delivery", "immunization", "high_risk_cases"]
INDICATORS = ["anc_coverage", "institutional_delivery", "immunization", "high_risk_cases"]
PERCENT_INDICATORS = ["anc_coverage", "institutional_delivery", "immunization"]

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    missing_columns = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    df["month"] = df["month"].astype(str)
    for col in INDICATORS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

try:
    df = load_data(DATA_PATH)
except Exception as exc:
    st.error(f"Could not load the supplied CSV: {exc}")
    st.stop()

st.subheader("1. Data loading and validation")
c1, c2, c3 = st.columns(3)
c1.metric("Rows", len(df))
c2.metric("Districts", df["district"].nunique())
c3.metric("Months", df["month"].nunique())
with st.expander("Preview CSV (head)"):
    st.dataframe(df.head(), use_container_width=True)
with st.expander("DataFrame info and missing values"):
    info_df = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(t) for t in df.dtypes],
        "non_null": [int(df[c].notna().sum()) for c in df.columns],
        "missing": [int(df[c].isna().sum()) for c in df.columns],
    })
    st.dataframe(info_df, use_container_width=True)
    st.write("Missing values per column")
    st.dataframe(df.isna().sum().rename("missing_values").to_frame(), use_container_width=True)

st.sidebar.header("Live filters")
district_options = sorted(df["district"].dropna().unique().tolist())
month_options = sorted(df["month"].dropna().unique().tolist())
selected_districts = st.sidebar.multiselect("District", district_options, default=district_options)
selected_months = st.sidebar.multiselect("Month", month_options, default=month_options)
selected_indicators = st.sidebar.multiselect("Indicator", INDICATORS, default=INDICATORS)

st.sidebar.header("Detection thresholds")
trend_threshold = st.sidebar.slider("Significant trend change (%)", 1, 100, 10, 1)
correlation_threshold = st.sidebar.slider("Correlation |r| threshold", 0.10, 1.00, 0.70, 0.05)
outlier_method = st.sidebar.selectbox("Outlier method", ["IQR", "Z-score"])
outlier_sensitivity = st.sidebar.slider(
    "IQR multiplier" if outlier_method == "IQR" else "Absolute Z-score threshold",
    0.5 if outlier_method == "IQR" else 1.0,
    3.0 if outlier_method == "IQR" else 5.0,
    1.5 if outlier_method == "IQR" else 3.0,
    0.1
)

filtered = df[
    df["district"].isin(selected_districts)
    & df["month"].isin(selected_months)
].copy()
visible_indicators = selected_indicators

def severity_for_change(change_pct, threshold):
    magnitude = abs(change_pct)
    if magnitude >= 2 * threshold:
        return "High"
    if magnitude >= threshold:
        return "Medium"
    return "Low"

def generate_insights(data, indicators, trend_limit, corr_limit, method, sensitivity):
    insights = []
    insight_id = 1
    ordered = data.sort_values("month")
    # Trend detection: only consecutive available rows per district/indicator.
    for district, group in ordered.groupby("district"):
        group = group.sort_values("month")
        for indicator in indicators:
            pairs = group[["month", indicator]].dropna().sort_values("month")
            if len(pairs) < 2:
                continue
            for i in range(1, len(pairs)):
                previous = pairs.iloc[i - 1]
                current = pairs.iloc[i]
                prev_val, curr_val = float(previous[indicator]), float(current[indicator])
                if prev_val == 0:
                    continue
                change_pct = (curr_val - prev_val) / abs(prev_val) * 100
                if abs(change_pct) >= trend_limit:
                    severity = severity_for_change(change_pct, trend_limit)
                    direction = "increased" if change_pct > 0 else "decreased"
                    unit = "%" if indicator in PERCENT_INDICATORS else "cases"
                    explanation = (
                        f"{indicator.replace('_', ' ').title()} in {district} {direction} "
                        f"from {prev_val:g} to {curr_val:g} ({change_pct:+.1f}%) between "
                        f"{previous['month']} and {current['month']}, meeting the "
                        f"{trend_limit}% significant-change threshold."
                    )
                    insights.append({
                        "insight_id": f"INS-{insight_id:04d}", "type": "trend",
                        "indicator": indicator, "entity": district, "period": str(current["month"]),
                        "value": curr_val, "prev_value": prev_val, "change_pct": round(change_pct, 2),
                        "metric": curr_val, "change": round(change_pct, 2), "severity": severity,
                        "explanation": explanation
                    })
                    insight_id += 1

    # Outlier detection over the selected rows, separately for each indicator.
    for indicator in indicators:
        vals = pd.to_numeric(data[indicator], errors="coerce").dropna()
        if len(vals) < 4:
            continue
        if method == "IQR":
            q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
            spread = q3 - q1
            lower, upper = q1 - sensitivity * spread, q3 + sensitivity * spread
            mask = data[indicator].notna() & ((data[indicator] < lower) | (data[indicator] > upper))
            method_note = f"IQR bounds ({lower:.2f}, {upper:.2f})"
        else:
            mean, std = vals.mean(), vals.std(ddof=0)
            if std == 0:
                continue
            z = (data[indicator] - mean) / std
            mask = data[indicator].notna() & (z.abs() >= sensitivity)
            method_note = f"|z-score| ≥ {sensitivity:g}"
        for _, row in data.loc[mask].iterrows():
            val = float(row[indicator])
            insights.append({
                "insight_id": f"INS-{insight_id:04d}", "type": "outlier",
                "indicator": indicator, "entity": str(row["district"]), "period": str(row["month"]),
                "value": val, "prev_value": np.nan, "change_pct": np.nan,
                "metric": val, "change": np.nan, "severity": "High",
                "explanation": f"{indicator.replace('_', ' ').title()} for {row['district']} in {row['month']} "
                               f"was {val:g}, identified as an outlier using {method_note}."
            })
            insight_id += 1

    # Correlations use selected numerical indicators and selected rows.
    corr_data = data[indicators].corr(method="pearson") if len(indicators) >= 2 else pd.DataFrame()
    if not corr_data.empty:
        for i, left in enumerate(corr_data.columns):
            for right in corr_data.columns[i + 1:]:
                r = corr_data.loc[left, right]
                if pd.notna(r) and abs(r) >= corr_limit:
                    insights.append({
                        "insight_id": f"INS-{insight_id:04d}", "type": "correlation",
                        "indicator": f"{left}:{right}", "entity": "Selected districts",
                        "period": "Selected months", "value": float(r), "prev_value": np.nan,
                        "change_pct": np.nan, "metric": float(r), "change": float(r),
                        "severity": "High" if abs(r) >= min(0.9, corr_limit + 0.2) else "Medium",
                        "explanation": f"Pearson correlation between {left.replace('_', ' ')} and "
                                       f"{right.replace('_', ' ')} is r={r:.2f} in the filtered data "
                                       f"(flag threshold |r| ≥ {corr_limit:.2f}). Correlation is not proof of causation."
                    })
                    insight_id += 1
    return pd.DataFrame(insights), corr_data

if not filtered.empty and visible_indicators:
    insights, corr_matrix = generate_insights(
        filtered, visible_indicators, trend_threshold, correlation_threshold,
        outlier_method, outlier_sensitivity
    )
else:
    insights, corr_matrix = pd.DataFrame(), pd.DataFrame()

st.subheader("2. Automated insights")
st.caption("Insights are recalculated when filters or thresholds change. Trend, outlier, and correlation findings are kept as separate types.")
if insights.empty:
    st.info("No insights meet the current filters and thresholds. Try widening the filters or adjusting a threshold.")
else:
    severity_counts = insights["severity"].value_counts().reindex(["Low", "Medium", "High"], fill_value=0)
    m1, m2, m3 = st.columns(3)
    m1.metric("Low severity", int(severity_counts["Low"]))
    m2.metric("Medium severity", int(severity_counts["Medium"]))
    m3.metric("High severity", int(severity_counts["High"]))
    st.plotly_chart(px.bar(
        x=severity_counts.index, y=severity_counts.values,
        labels={"x": "Severity", "y": "Insight count"}, title="Insight counts by severity"
    ), use_container_width=True)
    display_cols = ["insight_id", "type", "indicator", "entity", "period", "value",
                    "prev_value", "change_pct", "severity", "explanation"]
    st.dataframe(insights[display_cols], use_container_width=True)
    csv_bytes = insights[display_cols].to_csv(index=False).encode("utf-8")
    st.download_button("Download insights CSV", data=csv_bytes, file_name="insights.csv", mime="text/csv")
    st.download_button(
        "Download insights JSON",
        data=insights[display_cols].replace({np.nan: None}).to_json(orient="records", indent=2).encode("utf-8"),
        file_name="insights.json", mime="application/json"
    )

st.subheader("3. Correlation matrix")
if not corr_matrix.empty:
    st.plotly_chart(px.imshow(
        corr_matrix, text_auto=".2f", aspect="auto",
        color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        title="Pearson correlation heatmap"
    ), use_container_width=True)
    st.download_button(
        "Download correlation matrix CSV",
        data=corr_matrix.to_csv(index=True).encode("utf-8"),
        file_name="correlation_matrix.csv", mime="text/csv"
    )
    st.caption(
        f"Pairs with |r| ≥ {correlation_threshold:.2f} are flagged. "
        f"This sample has only {df['month'].nunique()} months and {df['district'].nunique()} districts; "
        "correlations from this small dataset are fragile and should not be interpreted as causal."
    )
else:
    st.info("Select at least two indicators and enough non-missing rows to calculate a correlation matrix.")

st.subheader("4. District-level indicator trends")
if not filtered.empty and visible_indicators:
    chart_indicator = st.selectbox("Indicator to plot", visible_indicators, key="line_indicator")
    line_data = filtered[["month", "district", chart_indicator]].dropna().sort_values("month")
    if not line_data.empty:
        fig = px.line(line_data, x="month", y=chart_indicator, color="district", markers=True,
                      title=f"{chart_indicator.replace('_', ' ').title()} by district")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No values available for this chart under the current filters.")
else:
    st.info("Choose at least one district, month, and indicator to display the trend chart.")

st.subheader("5. Filtered data")
st.dataframe(filtered[["month", "district"] + visible_indicators], use_container_width=True)
st.download_button(
    "Download filtered data CSV",
    data=filtered[["month", "district"] + visible_indicators].to_csv(index=False).encode("utf-8"),
    file_name="filtered_healthcare_data.csv", mime="text/csv"
)
st.info("Dataset limitation: the provided sample contains only two months across six districts (12 rows). Trend comparisons are limited, and Pearson correlations may be unstable.")
