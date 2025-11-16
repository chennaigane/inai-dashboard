def plan_to_result(df: pd.DataFrame, plan: dict) -> pd.DataFrame:
    """
    Convert a parsed JSON plan into a pandas DataFrame result.
    Always returns a DataFrame (even for single KPI values).
    """
    intent = (plan.get("intent") or "").lower()
    cols = plan.get("columns", {})
    dim  = cols.get("dimension")
    meas = cols.get("measure")
    date = cols.get("date")
    agg  = (plan.get("aggregation") or "").lower() or None
    filters = plan.get("filters", []) or []
    topn = plan.get("topn")
    chart = plan.get("chart", "table")

    work = apply_filters(df, filters)

    # timeseries -> ensure date parsing
    if date and date in work.columns:
        work[date] = pd.to_datetime(work[date], errors="coerce")

    # Aggregation-style intents
    if intent in ("aggregate", "topn", "timeseries", "kpi") and meas:
        # pick group-by key
        group_key = None
        if intent == "timeseries" and date:
            work["_bucket"] = work[date].dt.to_period("M").dt.to_timestamp()
            group_key = "_bucket"
        elif dim and dim in work.columns:
            group_key = dim

        agg_fn = agg if agg in ("sum", "mean", "count", "max", "min") else "sum"

        # If there's a group key, do groupby -> series -> reset_index
        if group_key is not None:
            # ensure numeric conversion for measure column when needed
            if agg_fn != "count":
                # convert measure to numeric for safe aggregation
                work[meas] = pd.to_numeric(work[meas], errors="coerce")
            grouped = work.groupby(group_key, dropna=False)[meas]
            if agg_fn == "count":
                res_ser = grouped.count()
            else:
                res_ser = grouped.agg(agg_fn)
            res = res_ser.reset_index()
            # Rename columns to be predictable: [group_key, <agg>_<measure>]
            res.columns = [group_key, f"{agg_fn}_{meas}"]
            # Apply top-n if requested
            if topn and isinstance(topn, int) and topn > 0:
                res = res.sort_values(by=res.columns[-1], ascending=False).head(topn)
            return res.reset_index(drop=True)

        # No group key -> scalar KPI (single value)
        else:
            if agg_fn == "count":
                value = int(len(work))
            else:
                value = pd.to_numeric(work[meas], errors="coerce").agg(agg_fn)
                # If result is a numpy scalar, keep as Python scalar for nicer display
                try:
                    if hasattr(value, "item"):
                        value = value.item()
                except Exception:
                    pass
            # return a small dataframe with metric/value columns
            return pd.DataFrame([{"metric": f"{agg_fn}_{meas}", "value": value}])

    # describe -> descriptive table
    if intent == "describe":
        return work.describe(include="all").T.reset_index().rename(columns={"index": "column"})

    # fallback: a sample table
    return work.head(200).reset_index(drop=True)

# frontend_streamlit/pages/17__AskHYIA.py
"""
Ask HyIA — Decision Intelligence page (robust import handling + UI).
This file will try to import production components from frontend_streamlit/components/.
If some components are missing or error at import-time, safe minimal fallbacks are used so the page remains usable.
"""

import os
import sys
import traceback
import streamlit as st
import pandas as pd

# ---------------------------
# Ensure components path
# ---------------------------
THIS_DIR = os.path.dirname(__file__)  # .../frontend_streamlit/pages
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))  # .../frontend_streamlit
COMPONENTS_PATH = os.path.join(PROJECT_ROOT, "components")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if COMPONENTS_PATH not in sys.path:
    sys.path.insert(0, COMPONENTS_PATH)

print("DEBUG: PROJECT_ROOT =", PROJECT_ROOT)
print("DEBUG: COMPONENTS_PATH =", COMPONENTS_PATH)
print("DEBUG: components exists? ->", os.path.exists(COMPONENTS_PATH))

# ---------------------------
# Try imports (with graceful fallbacks)
# ---------------------------
# define fallback stubs first
def _fallback_generate_issue_insights(df):
    insights = []
    fixes = []
    if df is None or df.empty:
        return ["No data loaded."], []
    miss = df.isna().mean().sort_values(ascending=False)
    for col, pct in miss.items():
        if pct > 0.2:
            insights.append(f"Column '{col}' has {pct:.1%} missing values.")
            fixes.append(f"Consider filling or removing missing values in '{col}'.")
    dup = int(df.duplicated().sum())
    if dup > 0:
        insights.append(f"{dup} duplicate rows found.")
        fixes.append("Consider removing duplicates.")
    if not insights:
        insights.append("No major data issues detected.")
    return insights, fixes

def _fallback_root_cause_analysis(df):
    return ["Root cause analysis fallback: provide more domain info to get better results."]

def _fallback_detect_anomalies(df):
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    num_cols = out.select_dtypes(include="number").columns.tolist()
    out["_final_anomaly_flag"] = False
    for c in num_cols:
        out["_final_anomaly_flag"] = out["_final_anomaly_flag"] | (pd.to_numeric(out[c], errors="coerce") <= 0)
    out["_nan_ratio"] = out.isna().mean(axis=1)
    out["_final_anomaly_flag"] = out["_final_anomaly_flag"] | (out["_nan_ratio"] > 0.5)
    return out

def _fallback_auto_generate_dashboards(df, top_n=5):
    # return a simple dict spec compatible with render_dashboard below
    roles = {"numeric": df.select_dtypes(include="number").columns.tolist(),
             "datetime": [c for c in df.columns if "date" in c.lower()]}
    dashboards = {"Overview & KPIs": {"cards": [], "insights": []}}
    for n in roles["numeric"][:3]:
        s = pd.to_numeric(df[n], errors="coerce")
        dashboards["Overview & KPIs"]["cards"].append({
            "title": f"Total {n}",
            "chart": "kpi",
            "data": pd.DataFrame([{"metric": f"sum_{n}", "value": float(s.sum())}])
        })
    return {"roles": roles, "dashboards": dashboards}

def _fallback_render_dashboard(spec):
    # simple renderer for the fallback spec
    if not spec or "dashboards" not in spec:
        st.info("No auto-dashboard spec to render.")
        return
    for name, content in spec["dashboards"].items():
        st.header(name)
        for card in content.get("cards", []):
            st.subheader(card.get("title", "Card"))
            dfc = card.get("data")
            if isinstance(dfc, pd.DataFrame):
                if card.get("chart") == "kpi" and not dfc.empty:
                    try:
                        val = dfc.iloc[0]["value"]
                        st.metric(card.get("title", "KPI"), f"{val:,.2f}")
                    except Exception:
                        st.dataframe(dfc)
                else:
                    st.dataframe(dfc)
            else:
                st.write(card.get("data"))
        if content.get("insights"):
            st.markdown("**Insights:**")
            for it in content["insights"]:
                st.write("-", it)

def _fallback_monthly_aggregate(df, date_col, value_col):
    tmp = df.copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
    tmp = tmp.dropna(subset=[date_col])
    if tmp.empty:
        return pd.DataFrame()
    monthly = tmp.set_index(date_col).resample("M")[value_col].sum().reset_index()
    return monthly

def _fallback_linear_forecast(monthly_df, months_ahead=6):
    if monthly_df is None or monthly_df.empty:
        return pd.DataFrame(), {"error": "no data"}
    monthly_df = monthly_df.sort_values(by=monthly_df.columns[0])
    last_val = float(monthly_df.iloc[-1, 1])
    future_dates = [monthly_df.iloc[-1, 0] + pd.DateOffset(months=i) for i in range(1, months_ahead+1)]
    fut_vals = [last_val] * months_ahead
    fut_df = pd.DataFrame({monthly_df.columns[0]: future_dates, monthly_df.columns[1]: fut_vals})
    out = pd.concat([monthly_df, fut_df], ignore_index=True)
    return out, {"method": "naive_repeat_last"}

def _fallback_plot_forecast(forecast_df, title="Forecast"):
    try:
        import plotly.express as px
        fig = px.line(forecast_df, x=forecast_df.columns[0], y=forecast_df.columns[1], title=title)
        return fig
    except Exception:
        return None

# attempt real imports, otherwise use fallbacks
try:
    from components.decision_engine import generate_issue_insights, root_cause_analysis
except Exception as e:
    print("IMPORT WARNING: components.decision_engine failed - using fallback.", repr(e))
    traceback.print_exc()
    generate_issue_insights = _fallback_generate_issue_insights
    root_cause_analysis = _fallback_root_cause_analysis

try:
    from components.anomaly_detector import detect_anomalies
except Exception as e:
    print("IMPORT WARNING: components.anomaly_detector failed - using fallback.", repr(e))
    traceback.print_exc()
    detect_anomalies = _fallback_detect_anomalies

try:
    from components.auto_dash import auto_generate_dashboards, render_dashboard
except Exception as e:
    print("IMPORT WARNING: components.auto_dash failed - using fallback.", repr(e))
    traceback.print_exc()
    auto_generate_dashboards = _fallback_auto_generate_dashboards
    render_dashboard = _fallback_render_dashboard

try:
    from components.forecasting import monthly_aggregate, linear_forecast, plot_forecast
except Exception as e:
    print("IMPORT WARNING: components.forecasting failed - using fallback.", repr(e))
    traceback.print_exc()
    monthly_aggregate = _fallback_monthly_aggregate
    linear_forecast = _fallback_linear_forecast
    plot_forecast = _fallback_plot_forecast

# ---------------------------
# Page UI
# ---------------------------
st.set_page_config(page_title="Ask HyIA (Decision Intelligence)", layout="wide")
st.title("🤖 Ask HyIA — Decision Intelligence")

# get dataframe from session (validated -> cleaned -> uploaded)
def get_active_df():
    for k in ("validated_df", "df_clean", "uploaded_df", "main_data", "active_df"):
        v = st.session_state.get(k)
        if isinstance(v, pd.DataFrame) and not v.empty:
            return v, k
    # also check datasets dict
    ds = st.session_state.get("datasets")
    name = st.session_state.get("active_name")
    if isinstance(ds, dict) and name in ds and isinstance(ds[name], pd.DataFrame):
        return ds[name], f"datasets[{name}]"
    return None, None

df, src = get_active_df()
if df is None:
    st.warning("Please upload or select a dataset first (Data Upload / Data Preview).")
    st.stop()

st.caption(f"Using dataset from: {src} — shape: {df.shape[0]} × {df.shape[1]}")

# Buttons / actions
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("🔍 Run Decision Insights"):
        insights, fixes = generate_issue_insights(df)
        st.header("What went wrong")
        for i in insights:
            st.warning(i)
        st.header("Recommended fixes")
        for f in fixes:
            st.success(f)

with col2:
    if st.button("🧾 Root-cause analysis"):
        rc = root_cause_analysis(df)
        st.header("Root-cause hints")
        for r in rc:
            st.info(r)

with col3:
    if st.button("🚨 Run Anomaly Detection"):
        anom = detect_anomalies(df)
        st.header("Anomaly sample (top flagged rows)")
        if isinstance(anom, pd.DataFrame) and "_final_anomaly_flag" in anom.columns:
            st.dataframe(anom[anom["_final_anomaly_flag"]].head(50), use_container_width=True)
        else:
            st.dataframe(anom.head(50), use_container_width=True)

with col4:
    if st.button("📊 Auto-generate Dashboards"):
        spec = auto_generate_dashboards(df)
        render_dashboard(spec)

if st.button("Auto-generate now", key="autodash_small"):
    st.session_state["autodash_small_go"] = True

if st.session_state.get("autodash_small_go"):
    spec = auto_generate_dashboards(df)
    render_dashboard(spec)


# Quick Forecast card (choose date & value)
st.markdown("---")
st.header("📈 Quick Forecast Card")
date_cols = [c for c in df.columns if "date" in c.lower() or pd.api.types.is_datetime64_any_dtype(df[c])]
num_cols = df.select_dtypes(include=["number"]).columns.tolist()
if date_cols and num_cols:
    dc = st.selectbox("Date column", date_cols, index=0)
    yc = st.selectbox("Numeric column", num_cols, index=0)
    horizon = st.number_input("Forecast months ahead", min_value=1, max_value=24, value=6)
    if st.button("Run Forecast"):
        monthly = monthly_aggregate(df, dc, yc)
        if monthly is None or monthly.empty:
            st.info("No valid monthly data after parsing selected date column.")
        else:
            fc, meta = linear_forecast(monthly, months_ahead=int(horizon))
            fig = plot_forecast(fc, title=f"{yc} forecast ({horizon} months)")
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            st.json(meta)
else:
    st.info("Need at least one date column and one numeric column to run forecast.")

# -----------------------------
# Interactive Data Explorer UI
# -----------------------------
st.markdown("---")
st.header("🔎 Interactive Data Explorer")

def show_column_info(df, col):
    st.subheader(f"Column: `{col}`")
    st.write("Type:", str(df[col].dtype))
    st.write("Unique values:", int(df[col].nunique(dropna=True)))
    st.write("Missing values:", int(df[col].isna().sum()))
    st.write("Sample values:")
    st.write(df[[col]].head(10))
    if pd.api.types.is_numeric_dtype(df[col]):
        st.write(df[col].describe().to_frame().T)

def safe_query(df, q):
    if not q or len(q.strip()) == 0:
        return df
    if "__" in q:
        raise ValueError("Unsafe tokens in query.")
    return df.query(q, engine="python")

# 1) Column selector and info
cols = df.columns.tolist()
c1, c2 = st.columns([2,3])
with c1:
    sel_col = st.selectbox("Select column to inspect", ["(none)"] + cols)
    if sel_col and sel_col != "(none)":
        show_column_info(df, sel_col)

with c2:
    st.subheader("Quick column stats (top 10)")
    stats_df = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(t) for t in df.dtypes],
        "missing": df.isna().sum().values,
        "unique": df.nunique(dropna=True).values
    }).sort_values(by="missing", ascending=False).head(10)
    st.dataframe(stats_df, use_container_width=True)

# 2) Filter builder
st.markdown("### 🔎 Filter builder (visual)")
filter_col = st.selectbox("Filter column", ["(none)"] + cols, key="filter_col")
if filter_col and filter_col != "(none)":
    if pd.api.types.is_numeric_dtype(df[filter_col]):
        minv = float(df[filter_col].min(skipna=True)) if df[filter_col].notna().any() else 0.0
        maxv = float(df[filter_col].max(skipna=True)) if df[filter_col].notna().any() else 0.0
        lo, hi = st.slider("Value range", min_value=minv, max_value=maxv, value=(minv, maxv))
        if st.button("Apply filter (visual)"):
            df_filtered = df[df[filter_col].between(lo, hi)]
            st.write(f"Filtered rows: {len(df_filtered):,}")
            st.dataframe(df_filtered.head(200), use_container_width=True)
    else:
        vals = df[filter_col].astype(str).value_counts().index.tolist()[:200]
        sel_vals = st.multiselect("Pick values to keep (top 200 shown)", vals)
        contains_text = st.text_input("OR search text contains (substring)")
        if st.button("Apply filter (visual)"):
            df_tmp = df
            if sel_vals:
                df_tmp = df_tmp[df_tmp[filter_col].astype(str).isin(sel_vals)]
            if contains_text:
                df_tmp = df_tmp[df_tmp[filter_col].astype(str).str.contains(contains_text, case=False, na=False)]
            st.write(f"Filtered rows: {len(df_tmp):,}")
            st.dataframe(df_tmp.head(200), use_container_width=True)

# 3) Distribution / frequency
st.markdown("### 📊 Distribution / Frequency")
dist_col = st.selectbox("Choose column for distribution", ["(none)"] + cols, key="dist_col")
if dist_col and dist_col != "(none)":
    if pd.api.types.is_numeric_dtype(df[dist_col]):
        fig = None
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.hist(pd.to_numeric(df[dist_col], errors="coerce").dropna(), bins=30)
            ax.set_title(f"Histogram: {dist_col}")
            st.pyplot(fig)
        except Exception:
            st.write(df[dist_col].describe())
    else:
        freq = df[dist_col].astype(str).value_counts().reset_index().rename(columns={"index": dist_col, dist_col: "count"})
        st.dataframe(freq.head(50), use_container_width=True)
        try:
            import plotly.express as px
            fig = px.bar(freq.head(20), x=dist_col, y="count", title=f"Top values of {dist_col}")
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass

# 4) Correlation / scatter
st.markdown("### 🔗 Correlation / Pair selection")
num_cols = df.select_dtypes(include=["number"]).columns.tolist()
if len(num_cols) >= 2:
    xcol = st.selectbox("X axis (numeric)", ["(none)"] + num_cols, key="xcol")
    ycol = st.selectbox("Y axis (numeric)", ["(none)"] + num_cols, key="ycol")
    if xcol != "(none)" and ycol != "(none)":
        try:
            import plotly.express as px
            fig = px.scatter(df, x=xcol, y=ycol, trendline="ols", title=f"{ycol} vs {xcol}")
            st.plotly_chart(fig, use_container_width=True)
            corr = df[[xcol, ycol]].corr().iloc[0, 1]
            st.write(f"Pearson correlation: {corr:.3f}")
        except Exception as e:
            st.write("Could not render correlation plot:", e)
else:
    st.info("Need at least two numeric columns for correlation.")

# 5) Pivot table builder
st.markdown("### 🔁 Pivot table builder")
pivot_index = st.selectbox("Index (rows)", ["(none)"] + cols, key="p_index")
pivot_columns = st.selectbox("Columns", ["(none)"] + cols, key="p_columns")
pivot_values = st.selectbox("Values (aggregate)", ["(none)"] + cols, key="p_values")
aggfunc = st.selectbox("Aggregation", ["sum", "mean", "count", "min", "max"], index=0)
if st.button("Generate pivot"):
    try:
        if pivot_index == "(none)":
            st.error("Choose an index for pivot.")
        else:
            table = pd.pivot_table(df, index=pivot_index,
                                   columns=(pivot_columns if pivot_columns != "(none)" else None),
                                   values=(pivot_values if pivot_values != "(none)" else None),
                                   aggfunc=aggfunc, fill_value=0)
            st.dataframe(table.reset_index().head(200), use_container_width=True)
    except Exception as e:
        st.error("Pivot failed: " + str(e))

# 6) Safe pandas query (advanced)
st.markdown("### 🧪 Run a safe query (use `df.query()` syntax)")
st.caption("Examples: `Sales > 1000 and Region == \"India\"`, `Quantity == 1`")
user_q = st.text_input("Enter query (df.query syntax)", key="user_query")
if st.button("Run query"):
    try:
        dfq = safe_query(df, user_q)
        st.write(f"Query result: {len(dfq):,} rows")
        st.dataframe(dfq.head(200), use_container_width=True)
    except Exception as e:
        st.error(f"Query failed: {e}")

# 7) Ask HyIA to recommend next actions
st.markdown("### 🤝 Ask HyIA to recommend next actions for this dataset")
ask_choice = st.radio("What type of recommendation?", ["Cleaning", "Modeling", "Dashboard", "Anomalies"])
if st.button("Get recommendation from HyIA"):
    if ask_choice == "Cleaning":
        insights_list, fixes_list = generate_issue_insights(df)
        st.header("Suggested cleaning steps (auto)")
        for s in insights_list:
            st.warning(s)
        for f in fixes_list:
            st.success(f)
    elif ask_choice == "Modeling":
        st.info("Suggested modeling: pick a target (numeric) and run a quick train/validation (AutoML).")
        numeric = df.select_dtypes(include=["number"]).columns.tolist()
        st.write("Numeric columns available:", numeric[:20])
    elif ask_choice == "Dashboard":
        st.info("Suggested dashboards: Overview KPIs, Time trends (by date), Category Top-N.")
        if st.button("Auto-generate now", key="autodash_small"):
            spec = auto_generate_dashboards(df, top_n=5)
            render_dashboard(spec)
    else:
        st.info("Run anomaly detection and then review flagged rows.")

# End of page
import streamlit as st
import pandas as pd

# --- Smart DataFrame fetch: always gets currently loaded data
def get_active_df():
    for k in ("validated_df", "df_clean", "uploaded_df", "main_data", "active_df"):
        v = st.session_state.get(k)
        if isinstance(v, pd.DataFrame) and not v.empty:
            return v
    return None

df = get_active_df()
st.header("🧠 Ask HyIA — Interactive Analytics Assistant")

if df is None:
    st.warning("Please upload/select your data first!")
    st.stop()

# Show current dataset preview (optional, for user confidence)
with st.expander("Preview loaded data"):
    st.dataframe(df.head(10), use_container_width=True)

user_question = st.text_input("Ask a question about your loaded data:")

def answer_query(query, df):
    query = str(query).lower()
    cols = [c.lower() for c in df.columns]
    if "top" in query and "state" in query and "sales" in query and "state" in cols and "sales" in cols:
        out = df.groupby("State")["Sales"].sum().sort_values(ascending=False).head(5)
        return "Top states by sales:", out.to_frame()
    if "fraud" in query and ("fraudflag" in cols or "fraud_flag" in cols):
        fraud_col = "FraudFlag" if "fraudflag" in df.columns else "Fraud_Flag"
        frauds = df[df[fraud_col].astype(str) == "True"]
        return f"Total fraud transactions: {len(frauds)}", frauds.head(10)
    if ("negative profit" in query or "loss" in query) and "profit" in cols:
        neg = df[df['Profit'] < 0]
        return f"Transactions with negative profit:", neg.head(10)
    if "customer type" in query and "customertype" in cols:
        out = df.groupby("CustomerType")[["Sales","Profit"]].sum()
        return "Sales and profit by customer type:", out
    if "payment method" in query and "paymentmethod" in cols:
        out = df["PaymentMethod"].value_counts()
        return "Payment method counts:", out.to_frame()
    # Add more intelligent or generic logic here
    sample = df.head(1)
    return f"Sorry, I can't answer that automatically yet. Here are your columns: {list(df.columns)}", None

if user_question:
    a, out = answer_query(user_question, df)
    st.markdown(f"**{a}**")
    if out is not None:
        st.dataframe(out)
        if st.button("Download this result"):
            st.download_button("Download as CSV", out.to_csv(), "hyia_result.csv")
    else:
        st.info("Try another question like: 'top states by sales', 'fraud transactions', 'negative profit', 'customer type', 'payment method'.")
else:
    st.caption("Examples: top states by sales, fraud transactions, negative profit, customer type, or payment method.")

raw = """6956.351330.221202.181914.151838.98164.394179.09 ..."""

# Split on "." to separate numbers (may need manual review if input format changes)
parts = raw.split(".")
numbers = []
i = 0

while i < len(parts) - 1:
    # Pair integer part with next two digits as decimal
    integer = parts[i]
    decimal = parts[i+1][:2]
    try:
        value = float(f"{integer}.{decimal}")
        numbers.append(f"{value:.2f}")
    except ValueError:
        pass
    # Move to next integer (skipping the decimal digits just taken)
    if len(parts[i+1]) > 2:
        # If decimal string is longer than 2 (overlap), next integer starts after 2 digits
        parts[i+1] = parts[i+1][2:]
    else:
        i += 1
    i += 1



