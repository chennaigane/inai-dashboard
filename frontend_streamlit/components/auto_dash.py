# frontend_streamlit/components/auto_dash.py
import pandas as pd
import plotly.express as px
from typing import Dict, Any, List

# -------------------------
# render_chart_from_plan
# -------------------------
def render_chart_from_plan(df_res: pd.DataFrame, plan: dict, use_plotly: bool = True):
    """
    Render a small chart/table given the result dataframe and plan.
    This function is self-contained and does NOT import streamlit
    so it can be reused in non-streamlit contexts (but in pages you
    will call streamlit to actually display).
    Returns a plotly Figure or the dataframe for display.
    """
    if df_res is None:
        return None

    chart = plan.get("chart", "table") if isinstance(plan, dict) else "table"

    # Table or empty
    if chart == "table" or df_res.empty:
        return df_res

    # KPI: single value
    if chart == "kpi" or (df_res.shape[0] == 1 and df_res.shape[1] == 2):
        # return df_res so caller can show metric
        return df_res

    # fallback to x,y
    if df_res.shape[1] >= 2:
        x = df_res.columns[0]
        y = df_res.columns[1]
    else:
        return df_res

    try:
        if chart == "bar":
            fig = px.bar(df_res, x=x, y=y)
        elif chart == "line":
            fig = px.line(df_res, x=x, y=y)
        elif chart == "area":
            fig = px.area(df_res, x=x, y=y)
        elif chart == "pie":
            fig = px.pie(df_res, names=x, values=y)
        elif chart == "scatter":
            fig = px.scatter(df_res, x=x, y=y)
        else:
            fig = px.bar(df_res, x=x, y=y)
        fig.update_layout(template="plotly_white", margin=dict(l=6, r=6, t=36, b=6))
        return fig
    except Exception:
        return df_res


# -------------------------
# auto_generate_dashboards
# -------------------------
def auto_generate_dashboards(df: pd.DataFrame, max_dashboards: int = 4) -> Dict[str, Any]:
    """
    Create a simple auto-dashboard SPEC from dataframe.
    The spec is a dict like:
    {
      "roles": {"numeric": [...], "category": [...], "time": [...]},
      "dashboards": {
          "Overview": {"cards": [ ... ]},
          ...
      }
    }
    This is purely in-memory spec; rendering is done by render_dashboard.
    """
    roles = {"time": [], "numeric": [], "category": []}
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            roles["time"].append(c)
        elif pd.api.types.is_numeric_dtype(df[c]):
            roles["numeric"].append(c)
        else:
            roles["category"].append(c)

    nums = roles["numeric"][:6]
    cats = roles["category"][:6]
    times = roles["time"]

    dashboards: Dict[str, Any] = {}

    # 1. Overview KPIs
    kpi_cards = []
    for n in nums[:3]:
        col_data = pd.to_numeric(df[n], errors="coerce").dropna()
        if not col_data.empty:
            kpi_cards.append({
                "type": "kpi",
                "title": f"Total {n}",
                "value": float(col_data.sum()),
                "metric": n
            })
    dashboards["Overview & KPIs"] = {"cards": kpi_cards, "notes": "KPI summary"}

    # 2. Time trend (if time exists)
    time_cards = []
    if times and nums:
        d = df.copy()
        d[times[0]] = pd.to_datetime(d[times[0]], errors="coerce")
        d = d.dropna(subset=[times[0]])
        if not d.empty:
            month = d[times[0]].dt.to_period("M").dt.to_timestamp()
            grp = d.assign(_m=month).groupby("_m")[nums[0]].sum().reset_index()
            time_cards.append({
                "type": "chart",
                "title": f"{nums[0]} by month",
                "chart": "line",
                "data": grp.to_dict("records"),
                "x": "_m",
                "y": nums[0]
            })
    dashboards["Time & Seasonality"] = {"cards": time_cards, "notes": "Time based trends"}

    # 3. Category breakdowns
    cat_cards = []
    if cats and nums:
        for c in cats[:2]:
            grp = df.groupby(c)[nums[0]].sum().reset_index().sort_values(nums[0], ascending=False).head(8)
            cat_cards.append({
                "type": "chart",
                "title": f"{nums[0]} by {c}",
                "chart": "bar",
                "data": grp.to_dict("records"),
                "x": c,
                "y": nums[0]
            })
    dashboards["Category Breakdown"] = {"cards": cat_cards, "notes": "Top category charts"}

    # 4. Data Quality
    dq_cards = []
    miss = df.isna().mean().sort_values(ascending=False).head(10).reset_index()
    miss.columns = ["column", "missing_ratio"]
    if not miss.empty:
        dq_cards.append({
            "type": "chart",
            "title": "Missing values",
            "chart": "bar",
            "data": miss.to_dict("records"),
            "x": "column",
            "y": "missing_ratio"
        })
    dashboards["Data Quality"] = {"cards": dq_cards, "notes": "Missingness & health"}

    spec = {"roles": roles, "dashboards": dashboards}
    return spec


# -------------------------
# render_dashboard
# -------------------------
def render_dashboard(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert spec -> a structure that the Streamlit page can iterate and render.
    This returns the same spec; actual plotting must be done by the page to
    present figures (we avoid streamlit calls here to prevent import-time side-effects).
    """
    # No side effects here
    return spec


# -------------------------
# generate_insights
# -------------------------
def generate_insights(df: pd.DataFrame, max_items: int = 6) -> List[str]:
    insights = []
    # high missing
    miss = df.isna().mean().sort_values(ascending=False)
    high = miss[miss > 0.2]
    for col, pct in high.items():
        insights.append(f"⚠️ Column `{col}` has {pct:.1%} missing values")

    # correlation
    nums = df.select_dtypes(include="number")
    if nums.shape[1] >= 2:
        corr = nums.corr().abs().max().max()
        if corr > 0.9:
            insights.append("🔍 Very high correlation (>0.9) among numeric columns")

    if not insights:
        insights.append("👍 No major issues detected (first-pass).")

    return insights[:max_items]
