# frontend_streamlit/components/ai_utils.py
from __future__ import annotations
import os, json
import pandas as pd
import streamlit as st

# Optional: OpenAI, but run safely without it
def _get_openai_client():
    try:
        from openai import OpenAI  # >=1.0 lib
    except Exception:
        return None, "OpenAI SDK not installed"
    api_key = st.secrets.get("openai_api_key", os.getenv("OPENAI_API_KEY"))
    if not api_key:
        return None, "Missing OPENAI_API_KEY"
    try:
        return OpenAI(api_key=api_key), None
    except Exception as e:
        return None, str(e)

SYSTEM_PROMPT = """You are a senior data analyst. 
Given a dataframe profile (columns, dtypes, sample rows) and a user goal,
return:
- 5-10 bullet insights
- a JSON array 'charts' of chart specs (plotly-like) each with: type(bar|line|area|pie|scatter|box), 
  x, y (list or str), color(optional), agg(optional: sum|mean|count), title.
- a brief 'executive_summary' (2-3 lines).
ONLY return valid JSON object with keys: insights, charts, executive_summary.
"""

def _profile_df(df: pd.DataFrame, n=5) -> dict:
    info = {
        "shape": list(df.shape),
        "columns": {c: str(df[c].dtype) for c in df.columns},
        "sample": df.head(n).to_dict(orient="records"),
    }
    return info

def analyze_with_llm(df: pd.DataFrame, goal: str = "Find key trends and KPIs") -> dict:
    """
    Returns dict: {insights: [..], executive_summary: str, charts: [specs]}
    If OpenAI is unavailable, returns a heuristic fallback with simple chart specs.
    """
    client, err = _get_openai_client()
    payload = {
        "goal": goal,
        "profile": _profile_df(df, n=5),
    }

    if client is None:
        # Fallback: heuristics
        numeric = df.select_dtypes(include="number").columns.tolist()
        cat = df.select_dtypes(include=["object", "category"]).columns.tolist()
        charts = []
        if cat and numeric:
            charts.append({"type":"bar", "x":cat[0], "y":numeric[0], "agg":"sum", "title":f"{numeric[0]} by {cat[0]}"})
        if len(numeric) >= 2:
            charts.append({"type":"scatter", "x":numeric[0], "y":numeric[1], "title":f"{numeric[0]} vs {numeric[1]}"})
        if numeric:
            charts.append({"type":"box", "x":numeric[0], "title":f"Distribution of {numeric[0]}"})
        return {
            "insights":[
                "Heuristic insights (LLM not configured).",
                f"Rows: {len(df)}, Cols: {df.shape[1]}",
                f"Numeric columns: {len(numeric)}, Categorical columns: {len(cat)}"
            ],
            "executive_summary":"Basic auto-analysis produced draft visuals. Add OPENAI_API_KEY for richer insights.",
            "charts": charts
        }

    # Using OpenAI responses (JSON mode)
    try:
        msg = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            response_format={"type":"json_object"},
            messages=[
                {"role":"system","content":SYSTEM_PROMPT},
                {"role":"user","content":json.dumps(payload)}
            ],
        )
        content = msg.choices[0].message.content
        data = json.loads(content)
        # sanity
        data.setdefault("insights", [])
        data.setdefault("charts", [])
        data.setdefault("executive_summary", "")
        return data
    except Exception as e:
        # Fail safe
        return {
            "insights":[f"LLM error: {e}", "Fallback to heuristics."],
            "executive_summary": "LLM call failed; showing minimal charts.",
            "charts":[]
        }
