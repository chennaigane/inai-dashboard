import streamlit as st  # ✅ imports always first
import streamlit as st
import plotly.express as px
import pandas as pd
from PIL import Image
from streamlit_option_menu import option_menu

import streamlit as st
import duckdb

# ✅ MUST be the first Streamlit command
st.set_page_config(page_title="INAI", layout="wide")

# 🔽 All other Streamlit commands come after this
st.sidebar.title("INAI Navigation")
st.sidebar.page_link("Home.py", label="Home", icon="🏠")
st.sidebar.page_link("pages/3_DataUpload.py", label="Upload", icon="⤴️")
st.sidebar.page_link("pages/10_QueryBuilder.py", label="SQL Editor", icon="🧮")

st.write("Welcome to INAI — upload data, run SQL, build dashboards.")


st.set_page_config(page_title="InaI – AI Data Scientist", page_icon="InaI-Logo.jpg", layout="wide")

# frontend_streamlit/pages/6_DataValidation.py

import os, sys
import streamlit as st
import pandas as pd

# Make sure we can import from ../components
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from components.data_validation import run_data_validation, render_validation_report

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

# 3) Run validation on click
if st.button("Run Validation"):
    with st.spinner("Running checks..."):
        results = run_data_validation(df, expected_schema, range_rules, patterns)
    st.success("Validation complete.")
else:
    st.stop()

# 4) Render report + interactive fixes (returns cleaned_df when you click Apply)
cleaned_df, fix_opts = render_validation_report(df, results)

# 5) Summary + Next step
st.markdown("---")
missing_total = int(results["missing"]["MissingCount"].sum()) if not results["missing"].empty else 0
dup_total = int(results["duplicates"])
summary = pd.DataFrame({
    "Rows": [len(df)],
    "Columns": [df.shape[1]],
    "Missing Values": [missing_total],
    "Duplicate Rows": [dup_total],
    "Schema Issues": [len(results["schema"])],
    "Range Issues": [len(results["ranges"])],
    "Format Issues": [len(results["patterns"])],
})
st.subheader("📊 Validation Summary")
st.dataframe(summary, use_container_width=True)

# Proceed only if fixes were applied (render_validation_report sets this)
if st.session_state.get("validated_df") is not None:
    st.success("Data cleaned and stored. Proceed to EDA.")
    if st.button("Proceed to EDA →"):
        st.switch_page("pages/7_EDA.py")
else:
    st.info("Apply fixes (if any) to continue.")
