import streamlit as st
import pandas as pd
import io
import json
import xml.etree.ElementTree as ET

st.title("Data Upload")


# Always initialize df from session (or None)
df = st.session_state.get("uploaded_df", None)

# ---- CSS FOR WHOLE PAGE AND GRID ----

st.markdown("""
            
<style>
.upload-section {
    margin-left: 60px;
    margin-top: 20px;
    margin-bottom: 18px;
}

.source-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 40px;
    margin-top: 28px;
    margin-bottom: 18px;
    justify-items: center;
    align-items: center;
    width: 100%;
    margin-left: 40px;  /* <-- This adds a left margin */
}
.square-box {
    background: linear-gradient(129deg,#253e77 80%, #f5f7fb 100%);
    border-radius: 15px;
    box-shadow: 0 2px 12px #22339718;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    width: 130px;
    height: 130px;
    transition: background 0.22s, box-shadow 0.2s;
    cursor: pointer;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
body, .stApp {
    background: #f3f4f7 !important;
}
.stApp, .block-container, .main {
    max-width: 100vw !important;
    width: 100vw !important;
    padding-left: 0 !important;
    padding-right: 0 !important;
}           
.data-catalog-tabs {
    display: flex;
    flex-wrap: wrap;
    justify-content: flex-start;
    gap: 36px;
    font-size: 1.13rem;
    font-weight: 900;
    margin: 18px 0 4px 0;
    padding: 0 14px 7px 4px;
    color: #223975;
}
.data-catalog-tabs span {
    cursor: pointer;
    transition: color 0.22s, border-bottom 0.22s;
    padding-bottom: 3px;
    margin-right: 6px;
}
.data-catalog-tabs span.selected, .data-catalog-tabs span:hover {
    color: #d32f2f;
    border-bottom: 2px solid #d32f2f;
}
.catalog-desc {
    font-size: 1.07rem;
    font-weight: 600;
    color: #436ae6;
    margin-bottom: 18px;
    margin-left: 3px;
}
.source-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 28px;
    margin-top: 28px;
    margin-bottom: 18px;
    margin-left: 60px;
    margin-right: 30px;
    justify-items: center;
    align-items: center;
    width: auto;
}
.square-box {
    background: linear-gradient(129deg,#253e77 80%, #f5f7fb 100%);
    border-radius: 15px;
    box-shadow: 0 2px 12px #22339718;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    width: 130px;
    height: 130px;
    transition: background 0.22s, box-shadow 0.2s;
    cursor: pointer;
}
.square-box:hover {
    background: linear-gradient(120deg,#1e2755 60%, #d32f2f 100%);
    box-shadow: 0 8px 30px #21366f50;
}
.square-icon {
    font-size: 2.18rem;
    margin-bottom: 5px;
}
.square-label {
    font-size: 13.5px;
    color: #fff;
    font-weight: 700;
    text-align: center;
}
.coming-soon {
    font-size: 24px;
    font-weight: 800;
    color: #d32f2f !important;
    background: none;
    text-align: center;
    margin-top: 28px;
    margin-bottom: 20px;
}
header, .data-catalog-tabs, .coming-soon {
    background: linear-gradient(90deg, #20396e 60%, #d32f2f 100%);
    color: #fff !important;
    border-radius: 14px;
    padding: 14px 30px 16px 30px;
    box-shadow: 0 2px 10px #24478812;
    margin-bottom: 22px;
}
.glossy-card {
    background: linear-gradient(109deg, #20396e 90%, #e3e7f7 100%);
    border-radius: 16px;
    box-shadow: 0 4px 20px #24478830;
    padding: 22px 34px 20px 34px;
    margin-left: 40px;
    margin-top: 14px;
    margin-right: 20px;
}
</style>
""", unsafe_allow_html=True)

# ---- HEADER ----
st.markdown("<h2 style='margin-bottom:4px;'>Import Your Data</h2>", unsafe_allow_html=True)
st.markdown(
    "<div class='data-catalog-tabs'>"
    "<span class='selected'>All</span>"
    "<span>Files & Feeds</span>"
    "<span>Databases</span>"
    "<span>Apps</span>"
    "<span>Sales</span>"
    "<span>Marketing</span>"
    "<span>Finance</span>"
    "<span>Help Desk</span>"
    "<span>Project Management</span>"
    "<span>Human Resources</span>"
    "<span>Social Media</span>"
    "<span>Sports</span>"
    "<span>Entertainment</span>"
    "</div>",
    unsafe_allow_html=True
)
st.markdown(
    "<div class='catalog-desc'>Import your data from a wide variety of sources to create insightful reports & dashboards.</div>",
    unsafe_allow_html=True
)

# ---- DATA SOURCE SQUARES ----

# ---- COMING SOON ----
st.markdown("<div class='coming-soon'>Coming soon: Connect directly to Business Applications, Cloud databases, Cloud Storage, CRMs, and Databases — catering to every operational and analytical requirement.</div>", unsafe_allow_html=True)

import streamlit as st
import pandas as pd
import json
import xml.etree.ElementTree as ET



# ---- Plan Selector in Sidebar ----
plan = st.sidebar.selectbox(
    "Choose your plan",
    ["Free","Basic (Starter)","Standard (Plus)" "Intermediate (Professional)", "Pro (Enterprise / AI Copilot)"],
    index=0
)

# ---- Ensure MEMORY is ALWAYS initialized at the top! ----
if "multi_file_memory" not in st.session_state:
    st.session_state["multi_file_memory"] = {}

if plan == "Free":
    max_files = 1
    max_size_mb = 10
    support_types = ["csv", "xls"]
    plan_msg = "Free plan: Up to 10 MB per file, supports CSV, Excel."
elif plan == "Basic (Starter)":
    max_files = 1
    max_size_mb = 50
    support_types = ["csv", "xls", "xlsx", "json", "zip"]
    plan_msg = "Basic plan: Up to 50 MB per file, supports CSV, Excel, JSON."
elif plan == "Standard (Plus)":
    max_files = 1
    max_size_mb = 150
    support_types = ["csv", "xls", "xlsx", "json", "zip", "tsv", "html", "ods"]
    plan_msg = "StarterStandard plan: Up to 150 MB per file, supports CSV, Excel, JSON."
elif plan == "Intermediate (Professional)":
    max_files = 3
    max_size_mb = 500
    support_types = ["csv", "xls", "xlsx", "json", "tsv", "html", "ods", "xml", "pdf", "txt"]
    plan_msg = "Professional plan: Up to 500 MB per file, supports all major office/text formats."
else:
    max_files = 5
    max_size_mb = 5000
    support_types = ["csv", "xls", "xlsx", "json", "tsv", "html", "ods", "xml", "pdf", "txt",
                    "sav", "sas7bdat", "rdata", "zip", "htm"]
    plan_msg = "Enterprise plan: Up to 5GB per file, all formats and live app connections."

st.markdown(f"<div style='margin-left:24px; color:#555; font-size:1.12rem;'>Maximum file size: <b>{max_size_mb} MB</b> for your plan.</div>", unsafe_allow_html=True)



# ---- FILE UPLOADER ----
if max_files == 1:
    files = [st.file_uploader("Upload your data file:", type=support_types, key="one_upload")]
    files = [f for f in files if f is not None]
else:
    files = st.file_uploader(
        f"Upload up to {max_files} files:",
        type=support_types,
        accept_multiple_files=True,
        key="multi_upload")
    if files is None:
        files = []

# ---- Loader Function ----
def load_df(file):
    ext = file.name.split(".")[-1].lower()
    try:
        if ext == "csv":
            return pd.read_csv(file)
        elif ext in ["xlsx", "xls", "ods"]:
            return pd.read_excel(file)
        elif ext == "tsv":
            return pd.read_csv(file, sep="\t")
        elif ext == "json":
            return pd.read_json(file)
        elif ext == "xml":
            tree = ET.parse(file)
            root = tree.getroot()
            return pd.DataFrame([{child.tag: child.text for child in item} for item in root])
        elif ext in ["html", "htm", "txt"]:
            st.write(f"{file.name} preview (first 1000 chars):")
            st.code(file.read(1000).decode(errors="ignore"))
        elif ext in ["pdf", "sav", "sas7bdat", "rdata", "zip"]:
            st.info(f"Preview for {ext.upper()} not supported; process in analytics.")
    except Exception as e:
        st.error(f"{file.name}: {e}")
    return None

# ---- Add files to session memory ----
new_mem = {}
for f in files[:max_files]:
    if f is not None:
        df = load_df(f)
        if isinstance(df, pd.DataFrame):
            new_mem[f.name] = df
            st.success(f"Loaded: {f.name} ({df.shape[0]} rows, {df.shape[1]} cols)")
            st.dataframe(df.head(30))  # Preview

if new_mem:
    # Add/replace, only up to max_files
    all_files = list(st.session_state["multi_file_memory"].keys()) + list(new_mem.keys())
    if len(all_files) > max_files:
        all_files = all_files[-max_files:]
    st.session_state["multi_file_memory"] = {k: v for k, v in {**st.session_state["multi_file_memory"], **new_mem}.items() if k in all_files}

# ---- SHOW MEMORY FILES + CLEAR BUTTON ----
if st.session_state["multi_file_memory"]:
    st.markdown("<div style='margin-left:20px; color:#254876; font-weight:600;'>Currently loaded files in session:</div>", unsafe_allow_html=True)
    for fname, df in st.session_state["multi_file_memory"].items():
        with st.expander(f"{fname} ({df.shape[0]} rows, {df.shape[1]} cols)", expanded=False):
            st.dataframe(df.head(50))
    if st.button("Clear All Data", key="clear_mem_uploads"):
        st.session_state["multi_file_memory"] = {}
        st.experimental_rerun()


if df is not None:
    st.session_state['uploaded_df'] = df
    st.success("File uploaded! Click below to preview and validate your data.")

    if st.button("Go to Data Preview ➡️"):
        st.switch_page("pages/4_DataPreview.py")




