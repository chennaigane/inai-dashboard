# frontend_streamlit/pages/16_AskHyIA_OpenAI.py
import os
import json
import traceback
import requests
import streamlit as st
import pandas as pd
import plotly.express as px

# ---------- Page ----------
st.set_page_config(page_title="Ask HyIA (OpenAI)", page_icon="🤖", layout="wide")
st.title("🤖 Ask HyIA — ChatGPT-powered Conversational Analytics (OpenAI)")

# ---------- Dataset discovery ----------
def get_current_df():
    for k in ("active_df", "df_clean", "validated_df", "uploaded_df", "main_data"):
        v = st.session_state.get(k)
        if isinstance(v, pd.DataFrame) and not v.empty:
            return v, k
    ds, an = st.session_state.get("datasets"), st.session_state.get("active_name")
    if isinstance(ds, dict) and an in ds and isinstance(ds[an], pd.DataFrame) and not ds[an].empty:
        return ds[an], f"datasets[{an}]"
    return None, None

df, source = get_current_df()
if df is None:
    st.warning("Please upload / validate a dataset first (Data Preview / Data Cleaning).")
    st.stop()

# avoid duplicated columns interfering
if df.columns.duplicated().any():
    df = df.loc[:, ~df.columns.duplicated()].copy()

st.caption(f"Using dataset from **{source}** — shape: **{df.shape[0]:,} × {df.shape[1]}**")

# ---------- OpenAI key & model ----------
OPENAI_KEY = st.secrets.get("OPENAI_API_KEY") if isinstance(st.secrets, dict) else os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    st.error("OpenAI API key not found. Set OPENAI_API_KEY in Streamlit secrets or environment variables.")
    st.stop()

# Model choice — change to whichever model your account supports.
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # low-cost dev model; change if unavailable

# ---------- Planner system prompt ----------
SYS_PLAN_PROMPT = """You are a concise data-analysis planner. Given (1) a dataset schema and (2) a user question,
return ONLY a compact JSON plan following this schema (no extra text):

{
  "intent": "aggregate|filter|describe|timeseries|topn|kpi",
  "columns": {"dimension": "<col or null>", "measure": "<col or null>", "date": "<col or null>"},
  "aggregation": "sum|mean|count|max|min|null",
  "filters": [{"column":"<col>","op":"==|!=|>|>=|<|<=","value":"<value>"}],
  "topn": <integer or null>,
  "chart": "bar|line|area|pie|scatter|table|kpi",
  "notes": "optional short hint"
}

Rules:
- Use exact column names from the schema.
- If question asks for time trend, set intent='timeseries' and set 'date'.
- For top-N requests set topn and chart='bar' or 'table'.
- For single overall numbers, intent='kpi' and chart='kpi'.
- Keep JSON as small as possible. Do not return prose.
"""

# ---------- OpenAI call helper ----------
def call_openai_for_plan(question: str, df: pd.DataFrame) -> str:
    """Call OpenAI ChatCompletion to get a JSON plan string."""
    schema = {"columns": {c: str(t) for c, t in df.dtypes.items()}}
    user_text = f"Dataset schema: {json.dumps(schema)}\nUser question: {question}\nReturn ONLY the JSON plan."
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"}

    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYS_PLAN_PROMPT},
            {"role": "user", "content": user_text}
        ],
        "temperature": 0.0,
        "max_tokens": 512,
        "top_p": 1.0,
    }

    resp = requests.post(url, headers=headers, json=body, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"OpenAI API error {resp.status_code}: {resp.text}")
    data = resp.json()
    # new-style access: data['choices'][0]['message']['content']
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    return content.strip()

# ---------- Plan execution helpers ----------
def apply_filters(df: pd.DataFrame, filters):
    if not filters:
        return df
    out = df.copy()
    for f in filters:
        col, op, val = f.get("column"), f.get("op"), f.get("value")
        if col not in out.columns:
            continue
        s = out[col]
        if op in [">", ">=", "<", "<="]:
            left = pd.to_numeric(s, errors="coerce")
            try:
                right = float(val)
            except Exception:
                right = pd.to_numeric(pd.Series([val]), errors="coerce").iloc[0]
        else:
            left, right = s, val
        try:
            if op == "==": out = out[left == right]
            elif op == "!=": out = out[left != right]
            elif op == ">": out = out[left > right]
            elif op == ">=": out = out[left >= right]
            elif op == "<": out = out[left < right]
            elif op == "<=": out = out[left <= right]
        except Exception:
            pass
    return out

def plan_to_result(df: pd.DataFrame, plan: dict) -> pd.DataFrame:
    intent = (plan.get("intent") or "").lower()
    cols = plan.get("columns", {})
    dim = cols.get("dimension")
    meas = cols.get("measure")
    date = cols.get("date")
    agg = (plan.get("aggregation") or "").lower() or None
    filters = plan.get("filters", [])
    topn = plan.get("topn")
    chart = plan.get("chart", "table")

    work = apply_filters(df, filters)

    if date and date in work.columns:
        work[date] = pd.to_datetime(work[date], errors="coerce")

    if intent in ("aggregate", "topn", "timeseries", "kpi") and meas:
        group_key = None
        if intent == "timeseries" and date:
            work["_bucket"] = work[date].dt.to_period("M").dt.to_timestamp()
            group_key = "_bucket"
        elif dim and dim in work.columns:
            group_key = dim

        agg_fn = agg if agg in ("sum", "mean", "count", "max", "min") else "sum"
        if agg_fn == "count":
            g = work.groupby(group_key, dropna=False)[meas].count() if group_key else pd.Series({"count": len(work)})
        else:
            if group_key:
                g = work.groupby(group_key, dropna=False)[meas].apply(lambda s: pd.to_numeric(s, errors="coerce")).agg(agg_fn)
            else:
                g = pd.Series({agg_fn: pd.to_numeric(work[meas], errors="coerce").agg(agg_fn)})

        res = g.reset_index()
        if group_key is None:
            res.columns = ["metric", "value"]
            return res
        res.columns = [(date if group_key == "_bucket" else dim), f"{agg_fn}_{meas}"]
        if topn and isinstance(topn, int) and topn > 0:
            res = res.sort_values(res.columns[-1], ascending=False).head(topn)
        return res

    if intent == "describe":
        return work.describe(include="all").T.reset_index().rename(columns={"index":"column"})

    return work.head(200)

# ---------- Chart rendering ----------
def render_chart_from_plan(df_res: pd.DataFrame, plan: dict):
    chart = plan.get("chart", "table")
    if chart == "table" or df_res.empty:
        st.dataframe(df_res, use_container_width=True)
        return

    if chart == "kpi" or (df_res.shape[0] == 1 and df_res.shape[1] == 2):
        label = df_res.columns[-2] if df_res.shape[1] == 2 else "Value"
        val = df_res.iloc[0, -1]
        if isinstance(val, (int, float)):
            st.metric(label=str(label), value=f"{val:,.2f}")
        else:
            st.metric(label=str(label), value=str(val))
        return

    if df_res.shape[1] >= 2:
        x = df_res.columns[0]; y = df_res.columns[1]
    else:
        st.dataframe(df_res, use_container_width=True)
        return

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
            st.dataframe(df_res, use_container_width=True); return
        fig.update_layout(template="plotly_white", margin=dict(l=8,r=8,t=40,b=8))
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.info(f"Could not render chart ({chart}): {e}")
        st.dataframe(df_res, use_container_width=True)

# ---------- UI ----------
st.markdown("Ask natural questions like: *Top 5 States by Sales this year*, *Monthly revenue trend*, *Total profit*")
examples = ["Total Sales", "Top 5 State by Sales", "Monthly Sales trend by OrderDate", "Average Profit by Region", "Show Orders where Country == India"]

question = st.chat_input("Ask your question about this data…")
if not question:
    st.caption("Examples:")
    st.write(" • " + "\n • ".join(examples))

if question:
    with st.spinner("Thinking…"):
        raw = ""
        try:
            raw = call_openai_for_plan(question, df)
            raw = (raw or "").strip()
            # defensive clean
            if raw.startswith("```"):
                raw = raw.strip("`")
                if raw[:4].lower().startswith("json"):
                    raw = raw[4:].strip()
            if not raw:
                st.error("OpenAI returned empty response.")
                st.stop()
            # show raw for debugging (collapsible)
            with st.expander("Raw model output (debug)"):
                st.code(raw)
            plan = json.loads(raw)
        except Exception as e:
            st.error(f"Model did not return a valid JSON plan: {e}")
            with st.expander("Raw output / traceback"):
                st.code(raw or "(empty)")
                st.code(traceback.format_exc())
            st.stop()

        st.subheader("Plan")
        st.json(plan, expanded=False)

        try:
            result_df = plan_to_result(df, plan)
        except Exception as e:
            st.error(f"Could not execute plan: {e}")
            st.stop()

        st.subheader("Result")
        render_chart_from_plan(result_df, plan)
        with st.expander("Result table"):
            st.dataframe(result_df, use_container_width=True)

import openai, os, json
openai.api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")

def call_openai_for_plan(prompt):
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":SYS_PLAN_PROMPT},
                      {"role":"user","content": prompt}],
            temperature=0.0,
            max_tokens=400,
        )
        return resp.choices[0].message["content"].strip()
    except openai.error.RateLimitError as e:
        # 429-ish rate/quota problems
        st.error("OpenAI rate/quota limit reached. Please check your OpenAI billing/usage.")
        # optional fallback: return "" or call local LLM or cached result
        return ""
    except openai.error.AuthenticationError:
        st.error("OpenAI API key invalid. Set OPENAI_API_KEY in env or Streamlit secrets.")
        return ""
    except Exception as e:
        st.error(f"OpenAI request failed: {e}")
        return ""
