# frontend_streamlit/pages/6_DataValidation.py

import os, sys
import streamlit as st
import pandas as pd

# Make sure we can import from ../components
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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


st.title("🔍 Data Validation & Auto-Fix Assistant")

# 1) Get a DataFrame uploaded earlier (by 3_DataUpload.py)
df = st.session_state.get("uploaded_df")
if df is None:
    st.warning("Please upload a dataset first (see **Data Upload** page).")
    st.stop()

st.write(f"**Dataset:** {len(df):,} rows × {df.shape[1]} columns")
st.dataframe(df.head(20), use_container_width=True)

# 2) Define validation rules (edit these or load from a JSON later)
expected_schema = {
    "CustomerID": "int",
    "Age": "int",
    "Revenue": "float",
    "Cost": "float",
    "Email": "str",
}
range_rules = {"Age": (0, 120), "Revenue": (0, None), "Cost": (0, None)}
patterns    = {"Email": r"^[\w._%+-]+@[\w.-]+\.[a-zA-Z]{2,}$"}

with st.expander("📘 View / Edit Validation Rules"):
    st.json({"Schema": expected_schema, "Range Rules": range_rules, "Patterns": patterns})
