# frontend_streamlit/pages/7_EDA.py
import os, sys
import streamlit as st
import pandas as pd
import numpy as np

import pandas as pd
import streamlit as st

def get_active_df():
    for k in ("active_df", "validated_df", "df_clean", "uploaded_df"):
        v = st.session_state.get(k)
        if isinstance(v, pd.DataFrame) and not v.empty:
            return v, k
    return None, None

df, source = get_active_df()
if df is None:
    st.error("Please upload data first!")
    st.stop()


# --- Safe imports for plotting ---
try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    st.error("Matplotlib not found. Run:  pip install matplotlib")
    st.stop()

try:
    import seaborn as sns
except ModuleNotFoundError:
    st.error("Seaborn not found. Run:  pip install seaborn")
    st.stop()

import plotly.express as px  # optional for the widget preview at the end

st.title("📈 EDA (Matplotlib + Seaborn)")

# ------------------------ helpers ------------------------
def pick_df_from_session():
    for key in ("validated_df", "df_clean", "uploaded_df"):
        if key in st.session_state and isinstance(st.session_state[key], pd.DataFrame):
            return st.session_state[key], key
    return None, None

def safe_to_datetime(series: pd.Series) -> pd.Series:
    """Robust datetime conversion (avoids duplicate-key/odd input issues)."""
    s = series
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    if pd.api.types.is_numeric_dtype(s):
        for unit in ("s", "ms"):
            try:
                return pd.to_datetime(s, unit=unit, errors="coerce")
            except Exception:
                pass
    try:
        return pd.to_datetime(s, errors="coerce", infer_datetime_format=True, utc=False)
    except Exception:
        return pd.to_datetime(s.astype(str), errors="coerce", utc=False)

# ------------------------ get dataframe ------------------------
df, source = pick_df_from_session()
if df is None or df.empty:
    st.warning("No dataset found. Please upload and validate/clean your data first.")
    st.stop()

# remove duplicate-named columns (pandas can choke on these)
if df.columns.duplicated().any():
    df = df.loc[:, ~df.columns.duplicated()].copy()
    st.info("Duplicate column names were found and de-duplicated for analysis.")

st.caption(f"Using dataset from **{source}** — {len(df):,} rows × {df.shape[1]} columns")
st.dataframe(df.head(20), use_container_width=True)

# ------------------------ seaborn theme controls ------------------------
st.sidebar.header("Seaborn Theme")
sns_style   = st.sidebar.selectbox("Style", ["whitegrid", "darkgrid", "white", "dark", "ticks"], index=0)
sns_context = st.sidebar.selectbox("Context", ["notebook", "paper", "talk", "poster"], index=0)
sns_palette = st.sidebar.selectbox("Palette", ["deep","muted","pastel","bright","dark","colorblind"], index=0)
sns.set_theme(style=sns_style, context=sns_context, palette=sns_palette)

# ------------------------ tabs ------------------------
tab_summary, tab_dist, tab_corr, tab_time, tab_cat, tab_pairs = st.tabs(
    ["Summary", "Distribution", "Correlation", "Time Series", "Category", "Pair Plots"]
)

# ===================== Summary =====================
with tab_summary:
    st.subheader("Dataset Summary")
    st.write("Column Types:", {c: str(t) for c, t in df.dtypes.items()})
    st.write("Null Counts:", df.isna().sum().to_dict())

    st.subheader("Descriptive Statistics")
    try:
        desc = df.describe(include="all", datetime_is_numeric=True).T
    except Exception:
        desc = df.describe(include="all").T
    st.dataframe(desc, use_container_width=True)

    col_to_mode = st.selectbox("Column for Mode:", df.columns)
    m = df[col_to_mode].mode()
    st.write(f"Mode for **{col_to_mode}**: {m.iloc[0] if not m.empty else 'N/A'}")

# ===================== Distribution =====================
with tab_dist:
    st.subheader("Histogram & Box Plot (Seaborn + Matplotlib)")
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()

    if not num_cols:
        st.info("No numeric columns found.")
    else:
        col = st.selectbox("Numeric column", num_cols, key="dist_num")

        data = df[col].dropna()
        if data.empty:
            st.info(f"No valid numeric data in **{col}**.")
        else:
            # Seaborn histogram (with KDE toggle)
            kde = st.checkbox("Show KDE on histogram", value=True)
            fig1, ax1 = plt.subplots()
            sns.histplot(data, kde=kde, ax=ax1)
            ax1.set_title(f"Distribution of {col}")
            st.pyplot(fig1)

            # Matplotlib box plot
            fig2, ax2 = plt.subplots()
            ax2.boxplot(data, vert=False, showmeans=True)
            ax2.set_title(f"Box Plot of {col}")
            ax2.set_xlabel(col)
            st.pyplot(fig2)

# ===================== Correlation =====================
with tab_corr:
    st.subheader("Correlation Heatmap (Seaborn)")
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()

    if len(num_cols) < 2:
        st.info("Need at least two numeric columns.")
    else:
        corr = df[num_cols].corr(numeric_only=True)
        fig3, ax3 = plt.subplots(figsize=(min(10, 0.6*len(num_cols)+3), min(8, 0.6*len(num_cols)+2)))
        sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=ax3)
        ax3.set_title("Correlation Matrix")
        st.pyplot(fig3)

# ===================== Time Series =====================
with tab_time:
    st.subheader("Time Series (Seaborn lineplot / Matplotlib)")
    # detect datetime-like columns
    date_candidates = []
    for c in df.columns:
        try:
            s = safe_to_datetime(df[c])
            if s.notna().sum() >= max(5, int(0.2 * len(s))):
                date_candidates.append(c)
        except Exception:
            pass

    if not date_candidates:
        st.info("No datetime-like columns detected.")
    else:
        tcol = st.selectbox("Date/Time column", date_candidates)
        ycols = df.select_dtypes(include=["number"]).columns.tolist()
        if not ycols:
            st.info("No numeric columns for y-axis.")
        else:
            y = st.selectbox("Value column", ycols)


# ===================== Category =====================
with tab_cat:
    st.subheader("Category Analysis (Seaborn barplot)")
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()

    if not cat_cols:
        st.info("No categorical columns found.")
    elif not num_cols:
        st.info("No numeric columns to aggregate.")
    else:
        gcol = st.selectbox("Group by (category)", cat_cols)
        acol = st.selectbox("Aggregate (numeric)", num_cols)

        grp = df.groupby(gcol, dropna=False)[acol].agg(["count","mean","sum","min","max"]).reset_index()
        st.dataframe(grp, use_container_width=True)

        fig5, ax5 = plt.subplots()
        sns.barplot(data=grp, x=gcol, y="sum", ax=ax5)
        ax5.set_title(f"Sum of {acol} by {gcol}")
        ax5.tick_params(axis="x", rotation=45)
        st.pyplot(fig5)

# ===================== Pair Plots (optional, heavy) =====================
with tab_pairs:
    st.subheader("Pair Plot (Seaborn)")
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if len(num_cols) < 2:
        st.info("Need at least two numeric columns.")
    else:
        max_rows = st.slider("Sample rows (to keep it fast)", 100, 2000, 600, step=100)
        sample = df[num_cols].dropna().sample(min(max_rows, df.shape[0]), random_state=42)
        fig6 = sns.pairplot(sample)  # seaborn returns a FacetGrid
        st.pyplot(fig6.fig)

# ------------------------ Optional: plotly widget preview ------------------------
st.markdown("---")
st.header("🧩 Quick Plotly Widget Preview")
w_type = st.selectbox("Widget Type", ["Bar", "Line", "Area", "Scatter", "Pie"])
x = st.selectbox("X", df.columns)
y = st.selectbox("Y (numeric)", df.select_dtypes(include=["number"]).columns) if w_type != "Pie" else None

fig = None
try:
    if w_type == "Bar":
        fig = px.bar(df, x=x, y=y)
    elif w_type == "Line":
        fig = px.line(df, x=x, y=y)
    elif w_type == "Area":
        fig = px.area(df, x=x, y=y)
    elif w_type == "Scatter":
        fig = px.scatter(df, x=x, y=y)
    elif w_type == "Pie":
        # default to first numeric col if none picked
        vals = df.select_dtypes(include=["number"]).columns[0] if y is None else y
        fig = px.pie(df, names=x, values=vals)
except Exception as e:
    st.error(f"Plotly error: {e}")

if fig is not None:
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
# ---------- Layout Generator: create dashboard layout dicts ----------
def generate_dashboards(df: pd.DataFrame) -> dict:
    import plotly.express as px

    def basic_insights(df, roles):
        insights = []
        n_rows, n_cols = df.shape
        insights.append(f"The dataset contains {n_rows:,} rows and {n_cols} columns.")
        if roles.get("time"):
            insights.append(f"Time column detected: **{roles['time']}**.")
        if roles.get("category"):
            insights.append(f"Category column detected: **{roles['category']}**.")
        if roles.get("numeric"):
            insights.append(f"Numeric columns detected: {', '.join(roles['numeric'][:5])}" + ("..." if len(roles['numeric']) > 5 else ""))
        return insights

    def chart_time_series(df, t, n):
        fig = px.line(df, x=t, y=n, title=f"Time Series of {n}")
        return fig

    def chart_category_bar(df, cat, n):
        fig = px.bar(df, x=cat, y=n, title=f"Bar Chart of {n} by {cat}")
        return fig

    def chart_distribution(df, n):
        fig = px.histogram(df, x=n, nbins=30, title=f"Distribution of {n}")
        return fig

    def chart_correlation(df, nums):
        corr = df[nums].corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=True, title="Correlation Matrix")
        return fig

    # Identify roles
    roles = {"time": None, "category": None, "numeric": []}
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            roles["time"] = c
        elif pd.api.types.is_numeric_dtype(df[c]):
            roles["numeric"].append(c)
        elif pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_categorical_dtype(df[c]):
            if roles["category"] is None:
                roles["category"] = c

    nums = roles["numeric"]

    dashboards = {}

    # 1) Overview
    cards = []
    total_rows = len(df)
    cards.append({"type":"kpi","kpi":{"title":"Total Rows","value":total_rows,"avg":total_rows,"min":total_rows,"max":total_rows,"count":total_rows}})
    if roles["time"] and nums:
        cards.append({"type":"chart","fig":"time_series","params":{"t":roles["time"],"n":nums[0]}})
    dashboards["Overview"] = {"cards":cards, "insights": basic_insights(df, roles)}
    # 2) Category Analysis
    cards = []
    if roles["category"] and nums:
        cards.append({"type":"chart","fig":"category_bar","params":{"cat":roles["category"],"n":nums[0]}})
    if nums:
        cards.append({"type":"chart","fig":"distribution","params":{"n":nums[0]}})
    if len(nums) >= 2:
        cards.append({"type":"chart","fig":"correlation","params":{"nums":nums[:5]}})
    dashboards["Category Analysis"] = {"cards":cards, "insights": basic_insights(df, roles)}
    return {"roles": roles, "dashboards": dashboards}
    return {"roles": roles, "dashboards": dashboards}
    return {"roles": roles, "dashboards": dashboards}

def safe_to_datetime(series: pd.Series) -> pd.Series:
    """
    Robust datetime conversion that never uses the 'unit mapping' code path.
    Handles dict/list/string/numeric gracefully and returns pd.NaT on failure.
    """
    import numpy as np
    import pandas as pd
    from collections.abc import Mapping

    def parse_one(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return pd.NaT

        # Already datetime-like
        if isinstance(v, (pd.Timestamp, np.datetime64)):
            return pd.to_datetime(v, errors="coerce")

        # If it's a mapping like {"year": 2023, "month": 7, "day": 14}
        if isinstance(v, Mapping):
            y = v.get("year") or v.get("y")
            m = v.get("month") or v.get("m")
            d = v.get("day") or v.get("d")
            # If all present, try to build a Timestamp directly
            if y is not None and m is not None and d is not None:
                try:
                    return pd.Timestamp(int(y), int(m), int(d))
                except Exception:
                    return pd.NaT
            # If a single key contains an ISO string, try that
            for key in ("date", "datetime", "ts", "timestamp"):
                if key in v:
                    return pd.to_datetime(v[key], errors="coerce")
            # Fallback: stringify safely (avoids pandas' unit mapping)
            return pd.to_datetime(str(v), errors="coerce")

        # If it's a list/tuple like [year, month, day]
        if isinstance(v, (list, tuple)) and len(v) >= 3:
            try:
                return pd.Timestamp(int(v[0]), int(v[1]), int(v[2]))
            except Exception:
                # If it's not y/m/d, try generic parsing on the joined string
                return pd.to_datetime(" ".join(map(str, v)), errors="coerce")

        # Numeric epoch seconds / ms
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            # try seconds then ms
            for unit in ("s", "ms"):
                try:
                    ts = pd.to_datetime(v, unit=unit, errors="coerce")
                    if pd.notna(ts):
                        return ts
                except Exception:
                    pass

        # Generic parsing (string or anything else)
        return pd.to_datetime(v, errors="coerce")

    # Apply row-wise so pandas never routes to the unit-mapping code
    return series.apply(parse_one)

if df.columns.duplicated().any():
    df = df.loc[:, ~df.columns.duplicated()].copy()

st.caption(f"Using dataset from **{source}** — {len(df):,} rows × {df.shape[1]} columns")
st.dataframe(df.head(20), use_container_width=True)
