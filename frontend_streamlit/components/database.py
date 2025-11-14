import streamlit as st, duckdb

# frontend_streamlit/components/database.py
import os, duckdb, pandas as pd, streamlit as st

@st.cache_resource
def duck():
    return duckdb.connect(database=":memory:")

@st.cache_resource
def api_url():
    return st.secrets["api_url"]



_DUCK = None

def get_duckdb():
    """
    Return a singleton in-memory DuckDB connection, and (re)register any session DataFrames.
    """
    global _DUCK
    if _DUCK is None:
        _DUCK = duckdb.connect(database=':memory:')
    # register main df (pick best available)
    df = st.session_state.get("validated_df") or st.session_state.get("df_clean") or st.session_state.get("uploaded_df")
    if isinstance(df, pd.DataFrame):
        _DUCK.register("dataset", df)      # main user dataset
    return _DUCK
