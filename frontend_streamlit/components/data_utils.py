import streamlit as st, pandas as pd, io, requests

@st.cache_data(show_spinner=False)
def read_local(file_bytes:bytes, filename:str)->pd.DataFrame:
    if filename.lower().endswith(".xlsx"):
        return pd.read_excel(io.BytesIO(file_bytes))
    return pd.read_csv(io.BytesIO(file_bytes))

def backend_upload(name:str, file)->dict:
    r = requests.post(st.secrets["api_url"]+"/datasets/upload",
                      files={"file": (file.name, file.getvalue(), file.type)},
                      data={"name": name, "owner_id": 1})
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=30)
def get_catalog():
    return requests.get(st.secrets["api_url"]+"/datasets/catalog").json()

import streamlit as st, pandas as pd, io, requests

def _api():
    # Safe: returns None if secret missing
    return st.secrets.get("api_url")

@st.cache_data(show_spinner=False)
def read_local(file_bytes:bytes, filename:str)->pd.DataFrame:
    import pandas as pd, io
    if filename.lower().endswith(".xlsx"):
        return pd.read_excel(io.BytesIO(file_bytes))
    return pd.read_csv(io.BytesIO(file_bytes))

def backend_upload(name:str, file):
    api = _api()
    if not api:
        # Local-only fallback (no backend): keep df in session
        df = read_local(file.getvalue(), file.name)
        st.session_state["df"] = df
        st.session_state["active_table"] = name
        return {"ok": True, "dataset_id": -1, "mode": "local"}
    r = requests.post(api + "/datasets/upload",
                      files={"file": (file.name, file.getvalue(), file.type)},
                      data={"name": name, "owner_id": 1})
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=30)
def get_catalog():
    api = _api()
    if not api:
        # Local mode: show only the session dataset if present
        if "df" in st.session_state and "active_table" in st.session_state:
            return [{"id": -1,
                     "name": st.session_state["active_table"],
                     "rows": len(st.session_state["df"]),
                     "bytes": 0,
                     "storage_path": "(memory)",
                     "schema_json": {c:str(t) for c,t in st.session_state["df"].dtypes.items()},
                     "lineage": {"source":"local"}}]
        return []
    return requests.get(api + "/datasets/catalog").json()

import streamlit as st, pandas as pd, io, requests

def _api():
    # returns None when secrets.toml not present
    return st.secrets.get("api_url") if hasattr(st, "secrets") else None

@st.cache_data(show_spinner=False)
def read_local(file_bytes:bytes, filename:str)->pd.DataFrame:
    if filename.lower().endswith(".xlsx"):
        return pd.read_excel(io.BytesIO(file_bytes))
    return pd.read_csv(io.BytesIO(file_bytes))

def backend_upload(name:str, file):
    api = _api()
    if not api:
        # local mode: keep in memory
        df = read_local(file.getvalue(), file.name)
        st.session_state["df"] = df
        st.session_state["active_table"] = name
        return {"ok": True, "dataset_id": -1, "mode": "local"}
    r = requests.post(api + "/datasets/upload",
                      files={"file": (file.name, file.getvalue(), file.type)},
                      data={"name": name, "owner_id": 1})
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=30)
def get_catalog():
    api = _api()
    if not api:
        # show only the session dataset when no backend
        if "df" in st.session_state and "active_table" in st.session_state:
            df = st.session_state["df"]
            return [{
                "id": -1,
                "name": st.session_state["active_table"],
                "rows": len(df),
                "bytes": 0,
                "storage_path": "(memory)",
                "schema_json": {c: str(t) for c, t in df.dtypes.items()},
                "lineage": {"source": "local"}
            }]
        return []
    return requests.get(api + "/datasets/catalog").json()
