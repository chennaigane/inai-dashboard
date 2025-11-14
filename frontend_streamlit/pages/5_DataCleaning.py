# frontend_streamlit/pages/5_DataCleaning.py (example name)
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout="wide", page_title="InaI Data Cleaning & Pre-Processing", page_icon="🧼")

# ------------------------ common helpers ------------------------
def get_active_df():
    """
    Return (df, source_key) from the first available location.
    Supports Data Preview storage as well.
    """
    # preferred keys
    for k in ("active_df", "validated_df", "df_clean", "uploaded_df", "main_data"):
        v = st.session_state.get(k)
        if isinstance(v, pd.DataFrame) and not v.empty:
            return v.copy(), k

    # Data Preview structure (datasets + active_name)
    ds = st.session_state.get("datasets")
    an = st.session_state.get("active_name")
    if isinstance(ds, dict) and an in ds and isinstance(ds[an], pd.DataFrame) and not ds[an].empty:
        return ds[an].copy(), f"datasets[{an}]"

    return None, None


def set_active_df(df: pd.DataFrame):
    """Persist the latest cleaned/validated dataframe for other pages."""
    st.session_state["df_clean"] = df
    st.session_state["active_df"] = df  # keep a canonical copy
    # optional: also mirror to uploaded_df so older pages work
    st.session_state["uploaded_df"] = df


# ------------------------ layout ------------------------
tab1, tab2 = st.tabs(["Data Validation", "Data Cleaning & Pre-Processing"])

df, source = get_active_df()
if df is None:
    with tab1:
        st.error("Please upload data first!")
    with tab2:
        st.error("Please upload data first!")
    st.stop()

# de-duplicate column names if any
if df.columns.duplicated().any():
    df = df.loc[:, ~df.columns.duplicated()].copy()

# ======================== TAB 1: VALIDATION ========================
with tab1:
    st.header("Data Validation")
    st.caption(f"Source: **{source}**  ·  Shape: **{df.shape[0]:,} × {df.shape[1]}**")

    st.subheader("Preview")
    st.dataframe(df.head(100), use_container_width=True)

    st.subheader("Missing Values")
    miss = df.isna().sum()
    miss = miss[miss > 0]
    if miss.empty:
        st.success("✅ No missing values detected.")
    else:
        st.table(miss.to_frame("MissingCount"))

    st.subheader("Duplicate Rows")
    dup_count = int(df.duplicated().sum())
    if dup_count == 0:
        st.success("✅ No duplicate rows found.")
    else:
        st.warning(f"⚠️ Duplicate rows: **{dup_count}**")
        if st.checkbox("Show duplicate rows"):
            st.dataframe(df[df.duplicated(keep=False)], use_container_width=True, height=300)

    st.subheader("Data Types")
    dtypes_df = pd.DataFrame({"Column": df.columns, "Data Type": df.dtypes.astype(str)})
    st.dataframe(dtypes_df, use_container_width=True, height=280)

    st.subheader("Outlier Check (IQR) — numeric only")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        st.info("No numeric columns available.")
    else:
        col = st.selectbox("Column", num_cols, key="outlier_col")
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lb, ub = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        out_n = int(((df[col] < lb) | (df[col] > ub)).sum())
        st.write(f"Bounds: [{lb:.3f}, {ub:.3f}] — Outliers: **{out_n}**")
        if out_n and st.checkbox("Show outliers"):
            st.dataframe(df[(df[col] < lb) | (df[col] > ub)][[col]], use_container_width=True, height=260)

# =================== TAB 2: CLEANING & PRE-PROCESS ===================
with tab2:
    st.header("Data Cleaning & Pre-Processing")
    work = df.copy()

    # ---- 1. Missing value handling ----
    st.subheader("Missing Value Handling")
    mv_opt = st.selectbox(
        "Choose strategy",
        ["(no change)", "Drop rows with any NA", "Impute numeric (median)", "Fill all NA with value"]
    )
    fill_val = None
    if mv_opt == "Fill all NA with value":
        fill_val = st.text_input("Fill value (as text; numeric cols will coerce if possible)", "")

    # ---- 2. Normalize/standardize text ----
    st.subheader("Normalize Text Columns")
    txt_cols = work.select_dtypes(include=["object", "string"]).columns.tolist()
    norm_cols = st.multiselect("Columns to normalize (lowercase/trim/ascii)", txt_cols)

    # ---- 3. Drop duplicates ----
    st.subheader("Duplicates")
    drop_dups = st.checkbox("Drop duplicate rows")

    # ---- 4. Encode categorical ----
    st.subheader("Categorical Encoding")
    cat_cols = work.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    enc_col = st.selectbox("Column to encode", ["(none)"] + cat_cols)
    enc_type = st.radio("Type", ["Label", "One-hot"], horizontal=True, disabled=(enc_col == "(none)"))

    # ---- 5. Dates & numerics format ----
    st.subheader("Parse Dates & Coerce Numerics")
    auto_dates = st.checkbox("Try parse date-like columns (contains 'date')", value=True)
    coerce_nums = st.checkbox("Coerce numerics where possible", value=True)

    # ---- 6. Merge/Append ----
    st.subheader("Merge/Append data (optional)")
    merge_file = st.file_uploader("CSV/Excel to merge/append", type=["csv", "xlsx"])
    merge_kind = st.radio("Operation", ["Append rows", "Merge on key"], horizontal=True, disabled=(merge_file is None))
    merge_key = None

    if merge_file and merge_kind == "Merge on key":
        # read once to show keys
        if merge_file.name.lower().endswith(".csv"):
            merge_df = pd.read_csv(merge_file)
        else:
            merge_df = pd.read_excel(merge_file)
        common_cols = sorted(list(set(work.columns) & set(merge_df.columns)))
        if common_cols:
            merge_key = st.selectbox("Join key", common_cols)
        else:
            st.warning("No common columns to join on.")

    # ---- APPLY all ----
    if st.button("✅ Apply Cleaning"):
        # 1) missing values
        if mv_opt == "Drop rows with any NA":
            work = work.dropna()
        elif mv_opt == "Impute numeric (median)":
            for c in work.select_dtypes(include=[np.number]).columns:
                work[c] = work[c].fillna(work[c].median())
        elif mv_opt == "Fill all NA with value":
            work = work.fillna(fill_val)

        # 2) normalize selected text
        for c in norm_cols:
            work[c] = (
                work[c]
                .astype(str)
                .str.strip()
                .str.lower()
                .str.normalize("NFKD")
                .str.encode("ascii", "ignore")
                .str.decode("ascii")
            )

        # 3) duplicates
        if drop_dups:
            work = work.drop_duplicates()

        # 4) encoding
        if enc_col != "(none)":
            if enc_type == "Label":
                work[enc_col] = pd.factorize(work[enc_col])[0]
            else:
                work = pd.get_dummies(work, columns=[enc_col])

        # 5) dates + numerics
        if auto_dates:
            for c in [c for c in work.columns if "date" in c.lower()]:
                try:
                    work[c] = pd.to_datetime(work[c], errors="coerce", infer_datetime_format=True)
                except Exception:
                    pass
        if coerce_nums:
            for c in work.columns:
                if work[c].dtype == object:
                    # try a gentle numeric coerce, keep if majority numeric
                    tmp = pd.to_numeric(work[c], errors="coerce")
                    if tmp.notna().mean() > 0.8:
                        work[c] = tmp

        # 6) merge/append
        if merge_file:
            if merge_file.name.lower().endswith(".csv"):
                add_df = pd.read_csv(merge_file)
            else:
                add_df = pd.read_excel(merge_file)
            if merge_kind == "Append rows":
                work = pd.concat([work, add_df], ignore_index=True)
            elif merge_kind == "Merge on key" and merge_key:
                work = pd.merge(work, add_df, on=merge_key, how="left")

        # save for other pages
        set_active_df(work)
        st.success("Cleaning applied. Data saved to session (df_clean & active_df).")

    # Preview current working copy (either original df or last-applied clean)
    current = st.session_state.get("df_clean", work)
    st.markdown("### Cleaned Data Preview")
    st.dataframe(current.head(500), use_container_width=True)
    st.caption(f"Rows: {current.shape[0]:,} | Columns: {current.shape[1]:,}")


    