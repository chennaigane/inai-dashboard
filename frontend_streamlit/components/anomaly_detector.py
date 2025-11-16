# frontend_streamlit/components/anomaly_detector.py
"""
Minimal safe stub for anomaly detection so imports succeed.
Replace with the full implementation when ready.
"""

import pandas as pd
import numpy as np

def detect_anomalies(df: pd.DataFrame, numeric_cols: list = None, if_params: dict = None) -> pd.DataFrame:
    """
    Simple rule-based anomaly flags:
      - _flag_neg_values: numeric cols <= 0
      - _flag_nan_ratio: rows with many NaNs
    Returns a DataFrame same index as df with added _final_anomaly_flag column.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    X = df.copy()
    # pick numeric columns if not provided
    if numeric_cols is None:
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    # initialize flags
    X["_flag_neg_values"] = False
    for c in numeric_cols:
        try:
            X["_flag_neg_values"] = X["_flag_neg_values"] | (pd.to_numeric(X[c], errors="coerce") <= 0)
        except Exception:
            continue

import pandas as pd
import numpy as np
from typing import Optional

def detect_anomalies(df: pd.DataFrame, numeric_cols: Optional[list] = None, if_params: dict = None) -> pd.DataFrame:
    """
    Rule-based anomaly flags:
     - _flag_neg_values: True for numeric cols <= 0
     - _flag_many_nans: True if row has >50% NaNs
     - _final_anomaly_flag: OR of flags
    Returns the input dataframe with added boolean flag columns (keeps original index).
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()

    out = df.copy()
    if numeric_cols is None:
        numeric_cols = out.select_dtypes(include=[np.number]).columns.tolist()

    # negative or zero numeric flag
    out["_flag_neg_values"] = False
    for c in numeric_cols:
        try:
            out["_flag_neg_values"] = out["_flag_neg_values"] | (pd.to_numeric(out[c], errors="coerce") <= 0)
        except Exception:
            continue

    # row-level missingness
    out["_nan_ratio"] = out.isna().mean(axis=1)
    out["_flag_many_nans"] = out["_nan_ratio"] > 0.5

    # final flag
    out["_final_anomaly_flag"] = out[["_flag_neg_values", "_flag_many_nans"]].any(axis=1)
    return out

# frontend_streamlit/components/anomaly_detector.py
"""
Minimal safe stub for anomaly detection to ensure imports succeed.
Replace with full logic later.
"""

import pandas as pd
import numpy as np
from typing import Optional

def detect_anomalies(df: pd.DataFrame, numeric_cols: Optional[list] = None, if_params: dict = None) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if numeric_cols is None:
        numeric_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    out["_flag_neg_values"] = False
    for c in numeric_cols:
        try:
            out["_flag_neg_values"] = out["_flag_neg_values"] | (pd.to_numeric(out[c], errors="coerce") <= 0)
        except Exception:
            continue
    out["_nan_ratio"] = out.isna().mean(axis=1)
    out["_flag_many_nans"] = out["_nan_ratio"] > 0.5
    out["_final_anomaly_flag"] = out[["_flag_neg_values", "_flag_many_nans"]].any(axis=1)
    return out
