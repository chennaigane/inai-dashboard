import streamlit as st
import pandas as pd
from io import BytesIO

st.title("🔎 Data Preview")

# ---------- Helpers ----------
def read_any(upload):
    if upload.name.lower().endswith(".csv"):
        return pd.read_csv(upload)
    return pd.read_excel(upload)

def init_state():
    st.session_state.setdefault("datasets", {})      # {name: DataFrame}
    st.session_state.setdefault("active_name", None) # current dataset key
    st.session_state.setdefault("toolbar_mode", None)

init_state()

# ---------- Upload area ----------
if not st.session_state["datasets"]:
    up = st.file_uploader("Upload MAIN dataset (CSV/Excel)", type=["csv","xlsx"], key="main_up")
    if not up:
        st.info("Upload a main dataset to begin.")
        st.stop()
    main_df = read_any(up)
    st.session_state["datasets"]["Main Data"] = main_df
    st.session_state["active_name"] = "Main Data"

st.success(f"Loaded {len(st.session_state['datasets'])} dataset(s)")

with st.expander("➕ Import additional datasets (optional)"):
    ups = st.file_uploader("Upload one or more datasets", type=["csv","xlsx"], accept_multiple_files=True, key="more_up")
    if ups:
        for f in ups:
            st.session_state["datasets"][f.name] = read_any(f)
        st.success(f"Imported {len(ups)} additional file(s).")

# ---------- Choose active dataset ----------
names = list(st.session_state["datasets"].keys())
active = st.selectbox("Active dataset", names, index=names.index(st.session_state["active_name"]) if st.session_state["active_name"] in names else 0)
st.session_state["active_name"] = active
df = st.session_state["datasets"][active].copy()

# ---------- Toolbar (real Streamlit buttons) ----------
st.markdown("### Toolbar")

btn_cols = st.columns(5)
with btn_cols[0]:
    if st.button("Filter", use_container_width=True):
        st.session_state["toolbar_mode"] = "filter"
with btn_cols[1]:
    if st.button("Sort", use_container_width=True):
        st.session_state["toolbar_mode"] = "sort"
with btn_cols[2]:
    if st.button("Add Column", use_container_width=True):
        st.session_state["toolbar_mode"] = "add"
with btn_cols[3]:
    if st.button("Delete Column", use_container_width=True):
        st.session_state["toolbar_mode"] = "delete"
with btn_cols[4]:
    if st.button("More", use_container_width=True):
        st.session_state["toolbar_mode"] = "more"

mode = st.session_state.get("toolbar_mode")

# ---------- Actions ----------
def commit(new_df: pd.DataFrame):
    # write changes back to current dataset and clear mode
    st.session_state["datasets"][st.session_state["active_name"]] = new_df
    st.session_state["toolbar_mode"] = None
    st.success("✅ Changes applied.")

if mode == "filter":
    st.subheader("Filter rows")
    col = st.selectbox("Column", df.columns)
    op = st.selectbox("Operator", ["contains", "==", "!=", ">", ">=", "<", "<="])
    val = st.text_input("Value (compared as string for 'contains')", "")
    if st.button("Apply Filter"):
        try:
            if op == "contains":
                new_df = df[df[col].astype(str).str.contains(val, na=False, case=False)]
            else:
                # evaluate safely
                if op == "==": new_df = df[df[col] == pd.to_numeric(val, errors="ignore")]
                elif op == "!=": new_df = df[df[col] != pd.to_numeric(val, errors="ignore")]
                elif op == ">":  new_df = df[pd.to_numeric(df[col], errors="coerce") >  pd.to_numeric(val, errors="coerce")]
                elif op == ">=": new_df = df[pd.to_numeric(df[col], errors="coerce") >= pd.to_numeric(val, errors="coerce")]
                elif op == "<":  new_df = df[pd.to_numeric(df[col], errors="coerce") <  pd.to_numeric(val, errors="coerce")]
                elif op == "<=": new_df = df[pd.to_numeric(df[col], errors="coerce") <= pd.to_numeric(val, errors="coerce")]
            commit(new_df)
        except Exception as e:
            st.error(f"Filter failed: {e}")

elif mode == "sort":
    st.subheader("Sort rows")
    col = st.selectbox("Sort by", df.columns)
    asc = st.checkbox("Ascending", value=True)
    na_pos = st.selectbox("NaNs position", ["last","first"])
    if st.button("Apply Sort"):
        try:
            new_df = df.sort_values(by=col, ascending=asc, na_position=na_pos)
            commit(new_df)
        except Exception as e:
            st.error(f"Sort failed: {e}")

elif mode == "add":
    st.subheader("Add column")
    new_col = st.text_input("New column name")
    fill_type = st.selectbox("Fill with", ["", "Empty string", "0", "NaN", "Copy from another column", "Expression (eval)"])
    copy_src = st.selectbox("Copy from", ["(none)"] + list(df.columns)) if fill_type == "Copy from another column" else None
    expr = st.text_input("Expression (e.g., price * qty)", "") if fill_type == "Expression (eval)" else None

    if st.button("Add"):
        try:
            new_df = df.copy()
            if fill_type == "Empty string":
                new_df[new_col] = ""
            elif fill_type == "0":
                new_df[new_col] = 0
            elif fill_type == "NaN":
                new_df[new_col] = pd.NA
            elif fill_type == "Copy from another column" and copy_src and copy_src != "(none)":
                new_df[new_col] = new_df[copy_src]
            elif fill_type == "Expression (eval)" and expr:
                # pandas.eval uses column names as variables
                new_df[new_col] = pd.eval(expr, engine="python", parser="pandas", target=new_df)
            else:
                new_df[new_col] = pd.NA
            commit(new_df)
        except Exception as e:
            st.error(f"Add column failed: {e}")

elif mode == "delete":
    st.subheader("Delete column")
    del_col = st.selectbox("Column to delete", df.columns)
    if st.button("Delete"):
        try:
            new_df = df.drop(columns=[del_col])
            commit(new_df)
        except Exception as e:
            st.error(f"Delete failed: {e}")

elif mode == "more":
    st.subheader("More actions")
    action = st.selectbox("Choose", [
        "Rename Column",
        "Change Data Type",
        "Find & Replace",
        "Split Column by Delimiter",
        "Show/Hide Columns",
        "Drop Duplicates"
    ])

    if action == "Rename Column":
        src = st.selectbox("Column", df.columns)
        new = st.text_input("New name", src + "_new")
        if st.button("Rename"):
            new_df = df.rename(columns={src:new})
            commit(new_df)

    elif action == "Change Data Type":
        src = st.selectbox("Column", df.columns)
        dtype = st.selectbox("To dtype", ["string","int","float","datetime"])
        if st.button("Convert"):
            new_df = df.copy()
            try:
                if dtype == "string":   new_df[src] = new_df[src].astype("string")
                elif dtype == "int":    new_df[src] = pd.to_numeric(new_df[src], errors="coerce").astype("Int64")
                elif dtype == "float":  new_df[src] = pd.to_numeric(new_df[src], errors="coerce").astype("Float64")
                elif dtype == "datetime": new_df[src] = pd.to_datetime(new_df[src], errors="coerce", infer_datetime_format=True)
                commit(new_df)
            except Exception as e:
                st.error(f"Conversion failed: {e}")

    elif action == "Find & Replace":
        src = st.selectbox("Column", df.columns)
        find = st.text_input("Find")
        repl = st.text_input("Replace with")
        if st.button("Apply Replace"):
            new_df = df.copy()
            new_df[src] = new_df[src].astype(str).str.replace(find, repl, regex=False)
            commit(new_df)

    elif action == "Split Column by Delimiter":
        src = st.selectbox("Column", df.columns)
        delim = st.text_input("Delimiter", ",")
        limit = st.number_input("Max splits (0 = all)", min_value=0, value=0)
        if st.button("Split"):
            new_df = df.copy()
            expanded = new_df[src].astype(str).str.split(delim, n=(None if limit==0 else limit), expand=True)
            for i, c in enumerate(expanded.columns):
                new_df[f"{src}_{i+1}"] = expanded[c]
            commit(new_df)

    elif action == "Show/Hide Columns":
        keep = st.multiselect("Visible columns", list(df.columns), default=list(df.columns))
        if st.button("Apply visibility"):
            commit(df[keep].copy())

    elif action == "Drop Duplicates":
        subset = st.multiselect("Subset columns (optional)", list(df.columns))
        if st.button("Drop"):
            new_df = df.drop_duplicates(subset=subset if subset else None)
            commit(new_df)

# ---------- Current preview ----------
cur = st.session_state["datasets"][st.session_state["active_name"]]
st.markdown("### Current Data")
st.dataframe(cur, use_container_width=True, height=420)
st.caption(f"Rows: {cur.shape[0]:,} | Columns: {cur.shape[1]:,}")

# ---------- Download cleaned ----------
def df_to_csv_bytes(df_: pd.DataFrame) -> bytes:
    return df_.to_csv(index=False).encode("utf-8")

st.download_button(
    "⬇️ Download cleaned CSV",
    data=df_to_csv_bytes(cur),
    file_name=f"{st.session_state['active_name'].replace(' ','_').lower()}_cleaned.csv",
    mime="text/csv",
)
st.markdown("✅ Data preview complete.")

# after: df = st.session_state["datasets"][active].copy()
st.session_state["active_df"] = df          # <— canonical
st.session_state["uploaded_df"] = df        # <— for older pages that expect this

def commit(new_df: pd.DataFrame):
    st.session_state["datasets"][st.session_state["active_name"]] = new_df
    # keep global copies in sync so other pages see the data
    st.session_state["active_df"] = new_df
    st.session_state["uploaded_df"] = new_df
    st.session_state["toolbar_mode"] = None
    st.success("✅ Changes applied.")
