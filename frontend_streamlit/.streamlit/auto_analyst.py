# frontend_streamlit/components/auto_analyst.py
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

# ---------- Helpers: detect column roles ----------
DATE_HINTS = ["date", "time", "dt", "day", "month", "year"]

def sniff_types(df: pd.DataFrame, max_cats: int = 30):
    cols = df.columns.tolist()
    # time col
    time_col = None
    for c in cols:
        lc = c.lower()
        if any(h in lc for h in DATE_HINTS):
            try:
                pd.to_datetime(df[c], errors="raise")
                time_col = c
                break
            except Exception:
                pass
    if time_col is None:
        # try parse any datetime-like
        for c in cols:
            try:
                pd.to_datetime(df[c], errors="raise")
                time_col = c
                break
            except Exception:
                continue

    # numeric & categorical
    num_cols = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    cat_cols = [c for c in cols if c not in num_cols]
    # keep only cats with manageable cardinality
    cat_cols = [c for c in cat_cols if df[c].nunique(dropna=True) <= max_cats]

    # strip id-like from cats if too unique
    id_like = [c for c in cat_cols if df[c].nunique() > max(10, len(df)*0.5)]
    cat_cols = [c for c in cat_cols if c not in id_like]

    return {
        "time_col": time_col,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "id_like": id_like
    }

# ---------- KPI summaries ----------
def kpi_cards(df: pd.DataFrame, num_cols: list[str]):
    kpis = []
    for c in num_cols[:4]:  # up to 4 KPIs
        s = df[c].dropna()
        if not len(s):
            continue
        kpis.append({
            "title": c,
            "value": float(s.sum()),
            "avg": float(s.mean()),
            "min": float(s.min()),
            "max": float(s.max()),
            "count": int(s.count()),
        })
    return kpis

# ---------- Insight rules ----------
def basic_insights(df: pd.DataFrame, roles: dict):
    insights = []
    t = roles["time_col"]
    nums = roles["num_cols"]
    cats = roles["cat_cols"]

    if t and nums:
        s = df[[t] + nums].copy()
        s[t] = pd.to_datetime(s[t], errors="coerce")
        s = s.dropna(subset=[t])
        for n in nums[:2]:
            by = s.groupby(pd.Grouper(key=t, freq="D"))[n].sum().dropna()
            if len(by) >= 2:
                delta = (by.iloc[-1] - by.iloc[-2])
                pct = (delta / (by.iloc[-2] if by.iloc[-2] != 0 else 1)) * 100
                direction = "↑" if delta >= 0 else "↓"
                insights.append(f"{n}: {direction} {abs(pct):.1f}% vs previous period.")
    if cats and nums:
        c = cats[0]; n = nums[0]
        top = df.groupby(c)[n].sum().sort_values(ascending=False).head(3)
        if not top.empty:
            items = ", ".join([f"{k}: {v:,.0f}" for k,v in top.items()])
            insights.append(f"Top {c} by {n}: {items}.")
    if not insights:
        insights = ["No strong trends found. Try filtering or changing date granularity."]
    return insights

# ---------- Chart builders ----------
def chart_time_series(df, t, n):
    d = df[[t, n]].copy()
    d[t] = pd.to_datetime(d[t], errors="coerce")
    d = d.dropna(subset=[t])
    d = d.groupby(pd.Grouper(key=t, freq="D"))[n].sum().reset_index()
    return px.line(d, x=t, y=n, title=f"{n} over time")

def chart_category_bar(df, cat, n, topn=10):
    d = df.groupby(cat)[n].sum().sort_values(ascending=False).head(topn).reset_index()
    return px.bar(d, x=cat, y=n, title=f"Top {topn} {cat} by {n}")

def chart_distribution(df, n):
    return px.histogram(df, x=n, nbins=30, title=f"Distribution of {n}")

def chart_correlation(df, nums):
    corr = df[nums].corr(numeric_only=True)
    fig = px.imshow(corr, text_auto=True, title="Correlation (numeric)")
    return fig

# ---------- Dashboard generators (return pure Python layout dicts) ----------
def generate_dashboards(df: pd.DataFrame):
    roles = sniff_types(df)
    t, nums, cats = roles["time_col"], roles["num_cols"], roles["cat_cols"]

    dashboards = {}

    # 1) Executive KPI
    cards = []
    for k in kpi_cards(df, nums):
        cards.append({"type":"kpi","title":k["title"],"kpi":k})
    # add 1–2 trend charts
    if t and nums:
        cards.append({"type":"chart","title":f"{nums[0]} Trend","fig":"time", "params":{"t":t,"n":nums[0]}})
        if len(nums) > 1:
            cards.append({"type":"chart","title":f"{nums[1]} Trend","fig":"time", "params":{"t":t,"n":nums[1]}})
    dashboards["Executive Overview"] = {"cards":cards, "insights": basic_insights(df, roles)}

    # 2) Category Breakdown
    cards = []
    if cats and nums:
        c = cats[0]
        for n in nums[:2]:
            cards.append({"type":"chart","title":f"Top {c} by {n}","fig":"cat_bar","params":{"cat":c,"n":n}})
    if nums:
        cards.append({"type":"chart","title":f"Distribution of {nums[0]}","fig":"dist","params":{"n":nums[0]}})
    dashboards["Category Breakdown"] = {"cards":cards, "insights": basic_insights(df, roles)}

    # 3) Time & Seasonality
    cards = []
    if t and nums:
        for n in nums[:2]:
            cards.append({"type":"chart","title":f"{n} by Day","fig":"time","params":{"t":t,"n":n}})
        # month seasonality
        d = df.copy()
        d[t] = pd.to_datetime(d[t], errors="coerce")
        d = d.dropna(subset=[t])
        d["month"] = d[t].dt.to_period("M").dt.to_timestamp()
        for n in nums[:1]:
            grp = d.groupby("month")[n].sum().reset_index()
            cards.append({"type":"chart","title":f"{n} by Month","fig":"line_raw","data":grp.to_dict("records"),"x":"month","y":n})
    dashboards["Time & Seasonality"] = {"cards":cards, "insights": basic_insights(df, roles)}

    # 4) Data Quality & Correlation
    cards = []
    if len(nums) >= 2:
        cards.append({"type":"chart","title":"Correlation","fig":"corr","params":{"nums":nums[:6]}})
    # missingness
    miss = df.isna().mean().sort_values(ascending=False).head(10).reset_index()
    miss.columns = ["column","missing_ratio"]
    miss["missing_ratio"] = (miss["missing_ratio"]*100).round(1)
    cards.append({"type":"table","title":"Missing Values % (top 10)","data":miss.to_dict("records")})
    dashboards["Data Quality"] = {"cards":cards, "insights": ["Review columns with high missing ratios."]}

    return {"roles": roles, "dashboards": dashboards}

# ---------- Renderer: turn layout dicts into Plotly/Streamlit renders ----------
def render_card(df, card, st):
    typ = card["type"]
    if typ == "kpi":
        k = card["kpi"]
        st.metric(k["title"], f"{k['value']:,.2f}", help=f"avg={k['avg']:.2f} min={k['min']:.2f} max={k['max']:.2f} n={k['count']}")
    elif typ == "chart":
        fig_code = card["fig"]
        if fig_code == "time":
            t, n = card["params"]["t"], card["params"]["n"]
            st.plotly_chart(chart_time_series(df, t, n), use_container_width=True)
        elif fig_code == "cat_bar":
            cat, n = card["params"]["cat"], card["params"]["n"]
            st.plotly_chart(chart_category_bar(df, cat, n), use_container_width=True)
        elif fig_code == "dist":
            n = card["params"]["n"]
            st.plotly_chart(chart_distribution(df, n), use_container_width=True)
        elif fig_code == "corr":
            nums = card["params"]["nums"]
            st.plotly_chart(chart_correlation(df, nums), use_container_width=True)
        elif fig_code == "line_raw":
            # when chart data was pre-aggregated into card["data"]
            x, y = card["x"], card["y"]
            data = pd.DataFrame(card["data"])
            st.plotly_chart(px.line(data, x=x, y=y, title=f"{y} by {x}"), use_container_width=True)
    elif typ == "table":
        st.dataframe(pd.DataFrame(card["data"]), use_container_width=True)
# ---------- Main Auto-Analyst Function ----------
def auto_analyst(df: pd.DataFrame, st):
    layout = generate_dashboards(df)
    dashboards = layout["dashboards"]

    for dname, dcontent in dashboards.items():
        st.header(dname)
        cols = st.columns(2)
        cards = dcontent["cards"]
        for i, card in enumerate(cards):
            with cols[i % 2]:
                render_card(df, card, st)
        if dcontent.get("insights"):
            st.subheader("Key Insights")
            for ins in dcontent["insights"]:
                st.write(f"- {ins}")
        st.markdown("---")