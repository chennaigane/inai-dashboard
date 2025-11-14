# frontend_streamlit/pages/9_DashboardBuilder.py
import os, json
import streamlit as st
import pandas as pd
import plotly.express as px

# ---------- Optional predictor (will continue without it) ----------
PRED_AVAILABLE = True
try:
    import sys, os as _os
    sys.path.append(_os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")))
    from components.predictor import train_auto_model
except Exception:
    PRED_AVAILABLE = False

# ---------- Helpers ----------
DASH_PATH = "data/dashboards.json"
os.makedirs("data", exist_ok=True)

def load_dashboards():
    if os.path.exists(DASH_PATH):
        return json.load(open(DASH_PATH, "r", encoding="utf-8"))
    return []  # [{name, cards: [card,...]}]

def save_dashboards(dbs):
    json.dump(dbs, open(DASH_PATH, "w", encoding="utf-8"), indent=2, default=str)

def get_active_df():
    """Pick the first available df from session."""
    for k in ("active_df", "validated_df", "df_clean", "uploaded_df", "main_data"):
        v = st.session_state.get(k)
        if isinstance(v, pd.DataFrame) and not v.empty:
            return v, k
    return None, None

def render_card(df: pd.DataFrame, card: dict):
    """Render a single card spec with Plotly/Streamlit."""
    typ = card.get("type", "kpi")
    title = card.get("title", typ.capitalize())
    st.caption(title)

    if typ == "kpi":
        col = card.get("value_col")
        if col and col in df.columns:
            val = pd.to_numeric(df[col], errors="coerce").sum()
            st.metric(label=title, value=f"{val:,.2f}")
        else:
            st.metric(label=title, value="—")
        return

    if typ == "map":
        lat, lon = card.get("lat"), card.get("lon")
        color = card.get("color")
        size = card.get("size")
        if lat in df.columns and lon in df.columns:
            fig = px.scatter_mapbox(
                df, lat=lat, lon=lon,
                color=color if color in df.columns else None,
                size=size if size in df.columns else None,
                mapbox_style="carto-positron", zoom=4, height=500
            )
            fig.update_layout(title=title)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Map needs Latitude/Longitude columns.")
        return

    # standard XY charts
    x = card.get("x")
    y = card.get("y")
    color = card.get("color")
    if x not in df.columns:
        st.warning("X column missing.")
        return
    if typ != "pie" and (y is None or y not in df.columns):
        st.warning("Y column missing.")
        return

    try:
        if typ == "bar":
            fig = px.bar(df, x=x, y=y, color=color if color in df.columns else None)
        elif typ == "line":
            fig = px.line(df, x=x, y=y, color=color if color in df.columns else None)
        elif typ == "area":
            fig = px.area(df, x=x, y=y, color=color if color in df.columns else None)
        elif typ == "scatter":
            fig = px.scatter(df, x=x, y=y, color=color if color in df.columns else None)
        elif typ == "box":
            fig = px.box(df, x=x, y=y, color=color if color in df.columns else None)
        elif typ == "pie":
            values = card.get("values") or (y if y in df.columns else None)
            if values is None:
                # choose first numeric col if not provided
                num_cols = df.select_dtypes(include=["number"]).columns
                values = num_cols[0] if len(num_cols) else None
            fig = px.pie(df, names=x, values=values, color=color if color in df.columns else None)
        else:
            st.info(f"Unknown card type: {typ}")
            return
        fig.update_layout(title=title, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Failed to render chart: {e}")

# ---------- Page ----------
st.title("📊 Dashboard Builder")

df, source = get_active_df()
if df is None:
    st.info("Please upload data in Data Preview first.")
    st.stop()

# Clean duplicate column names if any
if df.columns.duplicated().any():
    df = df.loc[:, ~df.columns.duplicated()].copy()

# Session state for building a dashboard
st.session_state.setdefault("current_cards", [])

# -------------------- Builder: add cards --------------------
with st.sidebar:
    st.header("Add a Widget")
    chart_type = st.selectbox(
        "Type",
        ["kpi", "bar", "line", "area", "pie", "scatter", "box", "map"],
        index=1
    )

    title = st.text_input("Title", value=f"{chart_type.upper()}")

    if chart_type == "kpi":
        value_col = st.selectbox("Value column (sum)", df.columns)
        if st.button("➕ Add to dashboard"):
            st.session_state["current_cards"].append({
                "type":"kpi", "title": title, "value_col": value_col
            })
            st.success("Added.")
    elif chart_type == "map":
        lat = st.selectbox("Latitude", df.columns)
        lon = st.selectbox("Longitude", df.columns)
        color = st.selectbox("Color (optional)", ["(none)"]+list(df.columns))
        size = st.selectbox("Size (optional)", ["(none)"]+list(df.columns))
        if st.button("➕ Add to dashboard"):
            st.session_state["current_cards"].append({
                "type":"map", "title": title,
                "lat": lat, "lon": lon,
                "color": None if color=="(none)" else color,
                "size": None if size=="(none)" else size
            })
            st.success("Added.")
    else:
        x = st.selectbox("X", df.columns)
        y = None if chart_type=="pie" else st.selectbox("Y", df.columns)
        color = st.selectbox("Color (optional)", ["(none)"]+list(df.columns))
        values = None
        if chart_type == "pie":
            values = st.selectbox("Values", df.columns)
        if st.button("➕ Add to dashboard"):
            st.session_state["current_cards"].append({
                "type": chart_type, "title": title,
                "x": x, "y": y, "color": None if color=="(none)" else color,
                "values": values
            })
            st.success("Added.")

# -------------------- Current dashboard preview --------------------
st.subheader("Current Dashboard (session)")
if not st.session_state["current_cards"]:
    st.info("Use the sidebar to add widgets.")
else:
    for card in st.session_state["current_cards"]:
        render_card(df, card)

st.markdown("---")

# -------------------- Save / Load dashboards --------------------
st.subheader("Save / Load")
dashboards = load_dashboards()
existing_names = [d["name"] for d in dashboards]

c1, c2, c3 = st.columns([2,1,1])
with c1:
    dash_name = st.text_input("Dashboard name", value="My Dashboard")
with c2:
    if st.button("💾 Save"):
        # upsert by name
        new = {"name": dash_name, "cards": st.session_state["current_cards"]}
        dashboards = [d for d in dashboards if d["name"] != dash_name] + [new]
        save_dashboards(dashboards)
        st.success("Saved.")
with c3:
    if st.button("🧹 Clear current"):
        st.session_state["current_cards"] = []
        st.experimental_rerun()

st.markdown("### Saved Dashboards")
sel = st.selectbox("Choose", ["(none)"] + existing_names)
if sel != "(none)":
    chosen = next(d for d in dashboards if d["name"] == sel)
    st.session_state["current_cards"] = chosen["cards"]
    st.success(f"Loaded **{sel}** into editor above.")

    # show view with current df
    st.markdown(f"#### View: {sel}")
    for card in chosen["cards"]:
        render_card(df, card)

    if st.button("🗑️ Delete this dashboard"):
        dashboards = [d for d in dashboards if d["name"] != sel]
        save_dashboards(dashboards)
        st.experimental_rerun()

# -------------------- Auto-generated dashboards (if any) --------------------
if "dashboards" in st.session_state and isinstance(st.session_state["dashboards"], dict):
    st.markdown("---")
    st.subheader("⚡ Auto-generated Dashboards")
    from components.ai_assistant import render_dashboards as _render_auto
    try:
        _render_auto(df, st.session_state["dashboards"])
    except Exception as e:
        st.info(f"Auto-dashboards available but could not render: {e}")

# -------------------- Predictive Analysis --------------------
st.markdown("---")
st.subheader("🔮 Predictive Analysis")
if not PRED_AVAILABLE:
    st.info("`components/predictor.py` not found. Add it to enable predictive analysis.")
else:
    target = st.selectbox("Target column", df.columns)
    if st.button("Run Predictive Analysis"):
        with st.spinner("Training model..."):
            model, metrics = train_auto_model(df, target)
        st.success(f"Done. Task: **{metrics['problem']}**")
        st.json(metrics)

        # Optional: feature importances (tree-based)
        try:
            # Pull importances from inner model if available
            clf = model.named_steps["model"]
            import numpy as np
            if hasattr(clf, "feature_importances_"):
                fi = clf.feature_importances_
                # Pull feature names from preprocessor
                pre = model.named_steps["pre"]
                num_names = pre.transformers_[0][2]
                cat_names = pre.transformers_[1][1]["onehot"].get_feature_names_out(pre.transformers_[1][2])
                names = list(num_names) + list(cat_names)
                imp_df = pd.DataFrame({"feature": names[:len(fi)], "importance": fi})
                fig = px.bar(imp_df.sort_values("importance", ascending=False).head(20),
                             x="feature", y="importance", title="Top Feature Importances")
                fig.update_layout(xaxis_tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass

import streamlit as st
import pandas as pd
import plotly.express as px

# make sure Python can find /components
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from components.predictor import train_auto_model, forecast_series

df = (
    st.session_state.get("active_df")
    or st.session_state.get("df_clean")
    or st.session_state.get("validated_df")
    or st.session_state.get("uploaded_df")
)

st.header("🔮 Predictive Analysis")

if not isinstance(df, pd.DataFrame) or df.empty:
    st.info("Upload/clean data first.")
    st.stop()

# ---- Tabular AutoML ----
st.subheader("1) AutoML (tabular)")
target = st.selectbox("Target column", df.columns, key="target_auto")
if st.button("Run AutoML"):
    with st.spinner("Training models..."):
        model, art = train_auto_model(df, target)
    st.success(f"Done. Problem: **{art['metrics']['problem']}**, Best: **{art['metrics']['best_model']}**")
    st.json(art["metrics"])
    if "importances" in art and art["importances"]:
        imp = pd.DataFrame(art["importances"]).sort_values("importance", ascending=False).head(20)
        st.plotly_chart(px.bar(imp, x="feature", y="importance", title="Top Feature Importances"), use_container_width=True)
    if "holdout_fig" in art:
        st.plotly_chart(art["holdout_fig"], use_container_width=True)
    elif "holdout_df" in art:
        st.dataframe(art["holdout_df"].head(100), use_container_width=True)

# ---- Forecasting ----
st.subheader("2) Forecast (time series)")
date_candidates = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()] + \
                  [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
date_candidates = list(dict.fromkeys(date_candidates)) or [None]

date_col = st.selectbox("Date column", date_candidates)
value_col = st.selectbox("Value to forecast (numeric)", df.select_dtypes("number").columns)
periods = st.slider("Future periods", 6, 60, 12)

if st.button("Generate forecast"):
    if date_col is None:
        st.warning("Pick a date/time column.")
    else:
        with st.spinner("Building forecast..."):
            out = forecast_series(df, date_col, value_col, periods)
        st.plotly_chart(out["fig"], use_container_width=True)
        st.dataframe(out["forecast_df"].tail(periods), use_container_width=True)

import streamlit as st
import pandas as pd

def get_current_df():
    for k in ("active_df","df_clean","validated_df","uploaded_df","main_data"):
        v = st.session_state.get(k)
        if isinstance(v, pd.DataFrame) and not v.empty:
            return v, k
    ds, an = st.session_state.get("datasets"), st.session_state.get("active_name")
    if isinstance(ds, dict) and an in ds and isinstance(ds[an], pd.DataFrame) and not ds[an].empty:
        return ds[an], f"datasets[{an}]"
    return None, None

# --- call this when you want to (re)build the auto dashboard ---
def build_auto_dashboard():
    df, source = get_current_df()
    if df is None:
        st.warning("No data loaded.")
        return

    # Store the df as the canonical dataset
    st.session_state["active_df"] = df

    # === Your generator (example) ===
    from components.ai_assistant import generate_dashboards  # your function that makes cards from df
    auto = generate_dashboards(df)  # returns {"roles":..., "dashboards": {...}}
    st.session_state["dashboards"] = auto  # <- consumed by Dashboard Builder view

    st.success(f"Auto dashboard generated from **{source}** (shape {df.shape[0]:,}×{df.shape[1]})")

# Example UI control
if st.button("⚡ Regenerate Auto Dashboard from Loaded Data"):
    build_auto_dashboard()

def render_card(df, card, st):
    typ = card["type"]
    title = card.get("title", typ.capitalize())

    if typ == "kpi":
        kcol = card.get("value_col")
        st.caption(title)
        if kcol in df.columns:
            val = pd.to_numeric(df[kcol], errors="coerce").sum()
            st.metric(title, f"{val:,.2f}")
        else:
            st.metric(title, "—")
        return

    if typ == "map":
        import plotly.express as px
        lat, lon = card.get("lat"), card.get("lon")
        color = card.get("color")
        size = card.get("size")
        st.caption(title)
        if lat in df.columns and lon in df.columns:
            fig = px.scatter_mapbox(
                df, lat=lat, lon=lon,
                color=(color if color in df.columns else None),
                size=(size if size in df.columns else None),
                mapbox_style="carto-positron", zoom=4, height=500
            )
            fig.update_layout(title=title, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Map needs Latitude/Longitude columns.")
        return

    # ---------- NEW: Forecast card ----------
    if typ == "forecast":
        st.caption(title)
        try:
            from components.predictor import forecast_series
            date_col = card["date_col"]
            value_col = card["value_col"]
            periods = int(card.get("periods", 12))
            out = forecast_series(df, date_col, value_col, periods)
            st.plotly_chart(out["fig"], use_container_width=True)
            if card.get("show_table"):
                st.dataframe(out["forecast_df"].tail(periods), use_container_width=True)
        except Exception as e:
            st.warning(f"Forecast error: {e}")
        return
    # ---------------------------------------

    # standard XY charts
    import plotly.express as px
    x = card.get("x")
    y = card.get("y")
    color = card.get("color")
    st.caption(title)
    try:
        if typ == "bar":
            fig = px.bar(df, x=x, y=y, color=None if color is None or color not in df.columns else color)
        elif typ == "line":
            fig = px.line(df, x=x, y=y, color=None if color is None or color not in df.columns else color)
        elif typ == "area":
            fig = px.area(df, x=x, y=y, color=None if color is None or color not in df.columns else color)
        elif typ == "scatter":
            fig = px.scatter(df, x=x, y=y, color=None if color is None or color not in df.columns else color)
        elif typ == "box":
            fig = px.box(df, x=x, y=y, color=None if color is None or color not in df.columns else color)
        elif typ == "pie":
            values = card.get("values") or y
            fig = px.pie(df, names=x, values=values, color=None if color is None or color not in df.columns else color)
        else:
            st.info(f"Unknown card type: {typ}")
            return
        fig.update_layout(title=title, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Failed to render card: {e}")

# ---------- Auto Dashboard + Forecast ----------
st.markdown("---")
st.subheader("⚡ Auto Dashboard + Forecast")

# Use the same df already loaded earlier in the page
df_use = st.session_state.get("active_df") or df

# simple role inference
num_cols = df_use.select_dtypes(include="number").columns.tolist()
cat_cols = [c for c in df_use.columns if c not in num_cols]

# choose date & value for forecast
date_candidates = [c for c in df_use.columns if "date" in c.lower() or "time" in c.lower()] + \
                  [c for c in df_use.columns if pd.api.types.is_datetime64_any_dtype(df_use[c])]
date_candidates = list(dict.fromkeys(date_candidates))

date_col = st.selectbox("Date column for forecast", date_candidates or ["(none)"])
value_col = st.selectbox("Value to forecast", num_cols or ["(none)"])
periods = st.slider("Forecast periods", 6, 36, 12)

if st.button("Build Auto Dashboard + Forecast"):
    cards = []

    # 1) KPIs (top 3 numerics)
    for n in num_cols[:3]:
        cards.append({"type":"kpi", "title":f"Total {n}", "value_col": n})

    # 2) Quick overview chart
    if cat_cols and num_cols:
        cards.append({"type":"bar", "title":f"{num_cols[0]} by {cat_cols[0]}", "x": cat_cols[0], "y": num_cols[0], "color": None})

    # 3) Forecast card (only if date+value valid)
    if date_col and date_col != "(none)" and value_col and value_col in num_cols:
        cards.append({
            "type":"forecast",
            "title": f"Forecast: {value_col}",
            "date_col": date_col,
            "value_col": value_col,
            "periods": periods,
            "show_table": False
        })
        st.success("Added forecast card.")
    else:
        st.info("No date column detected; added KPI/overview cards only.")

    # Drop into the live editor
    st.session_state["current_cards"] = cards
    st.success("Auto dashboard created. Scroll up to preview, tweak, and Save.")

