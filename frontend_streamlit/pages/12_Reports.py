# frontend_streamlit/pages/11_Reports.py
import os, json
import streamlit as st
import pandas as pd
import plotly.express as px

SAVE_PATH = "data/reports.json"
os.makedirs("data", exist_ok=True)

def load_reports():
    if os.path.exists(SAVE_PATH):
        return json.load(open(SAVE_PATH, "r", encoding="utf-8"))
    return []

def save_reports(items):
    json.dump(items, open(SAVE_PATH, "w", encoding="utf-8"), indent=2, default=str)

st.title("📑 Reports & Dashboards")

reports = load_reports()
names = [r["name"] for r in reports] if reports else []
sel = st.selectbox("Saved reports", ["(none)"] + names)

if sel != "(none)":
    rep = next(r for r in reports if r["name"] == sel)
    st.write("**Description**:", rep.get("desc",""))
    # cards: simple list of plotly spec
    for i, card in enumerate(rep.get("cards", []), start=1):
        st.write(f"### Card {i}: {card['type']} — {card.get('title','')}")
        df = pd.DataFrame(card["data"]) if "data" in card else None
        if card["type"] == "bar":
            fig = px.bar(df, x=card["x"], y=card["y"], color=card.get("color"))
            st.plotly_chart(fig, use_container_width=True)
        elif card["type"] == "line":
            fig = px.line(df, x=card["x"], y=card["y"], color=card.get("color"))
            st.plotly_chart(fig, use_container_width=True)
        elif card["type"] == "table":
            st.dataframe(df, use_container_width=True)

with st.expander("➕ Create a quick report from last query result"):
    df = st.session_state.get("last_query_result")
    if isinstance(df, pd.DataFrame):
        name = st.text_input("Report name", value="My Report")
        desc = st.text_input("Description", value="")
        x = st.selectbox("X", df.columns)
        y = st.selectbox("Y", df.select_dtypes(include=["number"]).columns)
        color = st.selectbox("Color", ["(none)"] + list(df.columns))
        if st.button("Save report"):
            card = {
                "type": "bar",
                "title": f"{y} by {x}",
                "x": x, "y": y, "color": None if color=="(none)" else color,
                "data": df[[x,y] + ([] if color=="(none)" else [color])].to_dict("records")
            }
            new = {"name": name, "desc": desc, "cards": [card]}
            reports = [r for r in reports if r["name"] != name] + [new]
            save_reports(reports)
            st.success("Saved.")
    else:
        st.info("Run a query in Query Builder first to create a report.")
# frontend_streamlit/components/auto_analyst.py
import pandas as pd
import plotly.express as px
import numpy as np
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype, is_categorical_dtype, is_object_dtype
from datetime import datetime
# ---------- Charting Functions ----------
def chart_time_series(df, t, n):
    fig = px.line(df, x=t, y=n, title=f"Time Series of {n} over {t}")
    fig.update_layout(template="plotly_white")
    return fig
def chart_category_bar(df, cat, n):
    fig = px.bar(df, x=cat, y=n, title=f"Bar Chart of {n} by {cat}")
    fig.update_layout(template="plotly_white")
    return fig
def chart_distribution(df, n):
    fig = px.histogram(df, x=n, nbins=30, title=f"Distribution of {n}")
    fig.update_layout(template="plotly_white")
    return fig
def chart_correlation(df, nums):
    corr = df[nums].corr()
    fig = px.imshow(corr, text_auto=True, title="Correlation Matrix")
    fig.update_layout(template="plotly_white")
    return fig
# ---------- Dashboard Generation Logic ----------
def generate_dashboards(df: pd.DataFrame):
    roles = {
        "time_col": None,
        "num_cols": [],
        "cat_cols": []
    }
    for col in df.columns:
        if is_datetime64_any_dtype(df[col]):
            roles["time_col"] = col
        elif is_numeric_dtype(df[col]):
            roles["num_cols"].append(col)
        elif is_categorical_dtype(df[col]) or is_object_dtype(df[col]):
            roles["cat_cols"].append(col)

    dashboards = {}

    # 1) Overview Dashboard
    cards = []
    for n in roles["num_cols"][:4]:
        kpi = {
            "title": n,
            "value": df[n].mean(),
            "avg": df[n].mean(),
            "min": df[n].min(),
            "max": df[n].max(),
            "count": df[n].count()
        }
        cards.append({"type":"kpi","kpi":kpi})
    dashboards["Overview"] = {"cards":cards, "insights": [f"Dataset has {len(df)} rows and {len(df.columns)} columns."]}

    # 2) Categorical Analysis
    cards = []
    for cat in roles["cat_cols"][:2]:
        for n in roles["num_cols"][:2]:
            grp = df.groupby(cat)[n].mean().reset_index()
            cards.append({"type":"chart","title":f"Avg {n} by {cat}","fig":"cat_bar","params":{"cat":cat,"n":n}})
    dashboards["Categorical Analysis"] = {"cards":cards, "insights": [f"Analyzed top categorical columns: {roles['cat_cols'][:2]}"]}

    # 3) Time Series Analysis
    cards = []
    if roles["time_col"]:
        t = roles["time_col"]
        for n in roles["num_cols"][:2]:
            grp = df.groupby(pd.Grouper(key=t, freq="M"))[n].mean().reset_index()
            cards.append({"type":"chart","title":f"{n} by Month","fig":"line_raw","data":grp.to_dict("records"),"x":"month","y":n})
        dashboards["Time & Seasonality"] = {"cards":cards, "insights": [f"Time series analysis based on {t}."]}

    # 4) Data Quality & Correlation
    cards = []
    if roles["num_cols"]:
       cards.append({"type":"chart","title":"Correlation Matrix","fig":"correlation","params":{"nums":roles["num_cols"]}})
    dashboards["Data Quality"] = {"cards":cards, "insights": [f"Correlation analysis on numeric columns."]}
    return {"roles": roles, "dashboards": dashboards}
def basic_insights(df: pd.DataFrame, roles: dict):
    insights = []
    insights.append(f"The dataset contains {len(df)} rows and {len(df.columns)} columns.")
    if roles["time_col"]:
        insights.append(f"Time column detected: {roles['time_col']}.")
    if roles["num_cols"]:
        insights.append(f"Numeric columns: {', '.join(roles['num_cols'][:5])}...")
    if roles["cat_cols"]:
        insights.append(f"Categorical columns: {', '.join(roles['cat_cols'][:5])}...")
    return insights
def kpi_cards(df: pd.DataFrame, nums: list):
    cards = []
    for n in nums:
        kpi = {
            "title": n,
            "value": df[n].mean(),
            "avg": df[n].mean(),
            "min": df[n].min(),
            "max": df[n].max(),
            "count": df[n].count()
        }
        cards.append(kpi)
    return cards
# ---------- Chart Renderers ----------
def render_card(df: pd.DataFrame, card: dict, st_module):
    typ = card["type"]
    if typ == "kpi":
        k = card["kpi"]
        st_module.metric(k["title"], f"{k['value']:,.2f}", help=f"avg={k['avg']:.2f} min={k['min']:.2f} max={k['max']:.2f} n={k['count']}")
    elif typ == "chart":
        fig_type = card.get("fig")
        if fig_type == "time":
            fig = chart_time_series(df, card["params"]["t"], card["params"]["n"])
            st_module.plotly_chart(fig, use_container_width=True)
        elif fig_type == "cat_bar":
            fig = chart_category_bar(df, card["params"]["cat"], card["params"]["n"])
            st_module.plotly_chart(fig, use_container_width=True)
        elif fig_type == "dist":
            fig = chart_distribution(df, card["params"]["n"])
            st_module.plotly_chart(fig, use_container_width=True)
        elif fig_type == "corr":
            fig = chart_correlation(df, card["params"]["nums"])
            st_module.plotly_chart(fig, use_container_width=True)
        elif fig_type == "line_raw":
            data = pd.DataFrame(card["data"])
            fig = px.line(data, x=card["x"], y=card["y"], title=card.get("title",""))
            fig.update_layout(template="plotly_white")
            st_module.plotly_chart(fig, use_container_width=True)
    elif typ == "table":
        data = pd.DataFrame(card["data"])
        st_module.dataframe(data, use_container_width=True)
# ---------- Main Auto-Analyst Function ----------
def auto_analyst(df: pd.DataFrame, st_module):
    layout = generate_dashboards(df)
    dashboards = layout["dashboards"]

    for dname, dcontent in dashboards.items():
        st_module.header(dname)
        cols = st_module.columns(2)
        cards = dcontent["cards"]
        for i, card in enumerate(cards):
            with cols[i % 2]:
                render_card(df, card, st_module)
        if dcontent.get("insights"):
            st_module.subheader("Key Insights")
            for ins in dcontent["insights"]:
                st_module.write(f"- {ins}")
        st_module.markdown("---")
