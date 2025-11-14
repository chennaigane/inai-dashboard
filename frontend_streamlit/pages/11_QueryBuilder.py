# frontend_streamlit/pages/10_QueryBuilder.py
import os, json, time
import streamlit as st
import pandas as pd
from components.database import get_duckdb

SAVE_PATH = "data/saved_queries.json"
os.makedirs("data", exist_ok=True)

def load_saved():
    if os.path.exists(SAVE_PATH):
        return json.load(open(SAVE_PATH, "r", encoding="utf-8"))
    return []

def save_all(items):
    json.dump(items, open(SAVE_PATH, "w", encoding="utf-8"), indent=2, default=str)

st.title("🧮 Query Builder (SQL)")

st.caption("Tip: your active dataset is registered as table **dataset** in DuckDB.")

# Saved queries list
saved = load_saved()
names = [q["name"] for q in saved] if saved else []
col1, col2 = st.columns([3,1])
with col1:
    selected = st.selectbox("Saved queries", ["(new)"] + names)
with col2:
    if st.button("🗑️ Delete selected", disabled=(selected == "(new)")):
        saved = [q for q in saved if q["name"] != selected]
        save_all(saved)
        st.experimental_rerun()

# Editor
default_sql = "SELECT * FROM dataset LIMIT 100;"
if selected != "(new)":
    default_sql = next((q["sql"] for q in saved if q["name"] == selected), default_sql)

sql = st.text_area("SQL", value=default_sql, height=200, placeholder="SELECT * FROM dataset LIMIT 100;")
params = st.text_input("Parameters (JSON dict)", value="{}", help='Example: {"country":"India","limit":200}')

# Run
if st.button("▶️ Run"):
    try:
        con = get_duckdb()
        # simple param substitution using named parameters
        p = json.loads(params or "{}")
        for k,v in p.items():
            con.execute(f"SET ${k} = '{v}'")
        res = con.execute(sql).fetchdf()
        st.success(f"Rows: {len(res)}")
        st.dataframe(res, use_container_width=True)
        st.download_button("Download CSV", res.to_csv(index=False), "query_result.csv")
        st.session_state["last_query_result"] = res
    except Exception as e:
        st.error(str(e))

# Save
with st.expander("💾 Save query"):
    new_name = st.text_input("Name", value=(selected if selected!="(new)" else "My Query"))
    desc = st.text_input("Description", value="")
    if st.button("Save"):
        entry = {"name": new_name, "sql": sql, "params": params, "desc": desc, "ts": int(time.time())}
        saved = [q for q in saved if q["name"] != new_name] + [entry]
        save_all(saved)
        st.success("Saved.")
