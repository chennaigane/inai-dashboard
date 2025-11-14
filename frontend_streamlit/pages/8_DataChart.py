# frontend_streamlit/pages/8_DataChart.py
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide", page_title="InaI — Data Chart", page_icon="📊")

# ---------------- helpers ----------------
def get_available_df():
    # Preferred keys first
    for k in ("active_df", "df_clean", "validated_df", "uploaded_df", "main_data"):
        v = st.session_state.get(k)
        if isinstance(v, pd.DataFrame) and not v.empty:
            return v.copy(), k
    # Data Preview store
    ds = st.session_state.get("datasets")
    an = st.session_state.get("active_name")
    if isinstance(ds, dict) and an in ds and isinstance(ds[an], pd.DataFrame) and not ds[an].empty:
        return ds[an].copy(), f"datasets[{an}]"
    return None, None

def apply_date_bucket(df: pd.DataFrame, col: str, bucket: str):
    s = pd.to_datetime(df[col], errors="coerce")
    if bucket == "(raw)":
        return s
    if bucket == "Day":
        return s.dt.date
    if bucket == "Week":
        return s.dt.to_period("W").dt.start_time
    if bucket == "Month":
        return s.dt.to_period("M").dt.to_timestamp()
    if bucket == "Quarter":
        return s.dt.to_period("Q").dt.to_timestamp()
    if bucket == "Year":
        return s.dt.year
    return s

# ---------------- load df ----------------
df, source = get_available_df()
if df is None:
    st.error("Please upload or clean data first (Data Preview / Data Cleaning).")
    st.stop()

# ensure canonical copy for other pages
st.session_state["active_df"] = df

if df.columns.duplicated().any():
    df = df.loc[:, ~df.columns.duplicated()].copy()

st.sidebar.header("Chart Builder")
chart_type = st.sidebar.selectbox("Chart type",
    ["Bar","Line","Area","Pie","Scatter","Box","Table","KPI"]
)

# Column choices
all_cols = df.columns.tolist()
num_cols = df.select_dtypes(include="number").columns.tolist()
cat_cols = [c for c in all_cols if c not in num_cols]

# X + date bucket (optional)
x_col = None
date_bucket = "(raw)"
if chart_type not in ["KPI","Table"]:
    x_col = st.sidebar.selectbox("X axis", all_cols)
    if "date" in x_col.lower() or "time" in x_col.lower() or pd.api.types.is_datetime64_any_dtype(df[x_col]):
        date_bucket = st.sidebar.selectbox("Date bucket", ["(raw)","Day","Week","Month","Quarter","Year"])

# Y columns & aggregation
agg = "(none)"
y_cols = []
if chart_type in ["Pie","KPI"]:
    if num_cols:
        y_cols = [st.sidebar.selectbox("Value", num_cols)]
else:
    if num_cols:
        y_cols = st.sidebar.multiselect("Y axis (one or more)", num_cols, default=num_cols[:1])
        agg = st.sidebar.selectbox("Aggregation", ["(none)","sum","mean","count","min","max"])

color_col = st.sidebar.selectbox("Color/Group (optional)", ["(none)"] + all_cols)

st.title("📊 Data Chart")
st.caption(f"Using source: **{source}**, shape: **{df.shape[0]:,} × {df.shape[1]}**")

# Build a working dataframe (aggregation if requested)
work = df.copy()

# Date bucketing
if x_col and date_bucket != "(raw)":
    work["_x_bucket"] = apply_date_bucket(work, x_col, date_bucket)
    x_plot = "_x_bucket"
else:
    x_plot = x_col

# Aggregation
if x_plot and y_cols and agg != "(none)":
    by = [x_plot]
    if color_col != "(none)":
        by.append(color_col)
    w = work[by + y_cols].copy()
    grouped = w.groupby(by, dropna=False).agg(agg).reset_index()
    work = grouped

# Render
fig = None
try:
    if chart_type == "Table":
        st.dataframe(work, use_container_width=True)
    elif chart_type == "KPI":
        if not y_cols:
            st.info("Pick a numeric column for KPI.")
        else:
            val = pd.to_numeric(work[y_cols[0]], errors="coerce").sum()
            st.metric(label=f"Total {y_cols[0]}", value=f"{val:,.2f}")
    elif chart_type == "Pie":
        names = x_plot if x_plot else (cat_cols[0] if cat_cols else None)
        vals = y_cols[0] if y_cols else None
        if names and vals:
            fig = px.pie(work, names=names, values=vals, color=None if color_col=="(none)" else color_col)
    elif chart_type == "Bar":
        fig = px.bar(work, x=x_plot, y=y_cols, color=None if color_col=="(none)" else color_col, barmode="group")
    elif chart_type == "Line":
        fig = px.line(work, x=x_plot, y=y_cols, color=None if color_col=="(none)" else color_col)
    elif chart_type == "Area":
        fig = px.area(work, x=x_plot, y=y_cols, color=None if color_col=="(none)" else color_col)
    elif chart_type == "Scatter":
        y = y_cols[0] if y_cols else None
        fig = px.scatter(work, x=x_plot, y=y, color=None if color_col=="(none)" else color_col)
    elif chart_type == "Box":
        y = y_cols[0] if y_cols else None
        fig = px.box(work, x=x_plot, y=y, color=None if color_col=="(none)" else color_col)

    if fig:
        fig.update_layout(template="plotly_white", margin=dict(l=8,r=8,t=40,b=8))
        st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.error(f"Could not render chart: {e}")

st.markdown("---")
st.subheader("Preview data used for this chart")
st.dataframe(work.head(200), use_container_width=True)
st.caption(f"Rows: {work.shape[0]:,} | Columns: {work.shape[1]:,}")
st.download_button("Download CSV", work.to_csv(index=False), "chart_data.csv")
