# components/decision_engine.py

import pandas as pd

def generate_issue_insights(df: pd.DataFrame):
    """
    Returns simple data issue insights.
    """
    insights = []

    # Missing values
    missing = df.isna().mean().sort_values(ascending=False)
    for col, pct in missing.items():
        if pct > 0.2:
            insights.append(f"⚠️ Column '{col}' has {pct:.1%} missing values.")

    # Duplicate rows
    dup = df.duplicated().sum()
    if dup > 0:
        insights.append(f"⚠️ Dataset contains {dup} duplicate rows.")

    if not insights:
        insights.append("👍 No major data issues detected.")
    return insights


def root_cause_analysis(df: pd.DataFrame):
    """
    Dummy placeholder — expand later.
    """
    return ["Root-cause analysis not implemented yet."]

# --- DIAGNOSTIC: ensure components folder is visible to Python ---
import os, sys, traceback, streamlit as st

THIS_DIR = os.path.dirname(__file__)                      # .../frontend_streamlit/pages
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))  # .../frontend_streamlit
COMPONENTS = os.path.join(PROJECT_ROOT, "components")

# add to path if missing
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if COMPONENTS not in sys.path:
    sys.path.insert(0, COMPONENTS)

print("DEBUG: PROJECT_ROOT =", PROJECT_ROOT)
print("DEBUG: COMPONENTS exists? ->", os.path.exists(COMPONENTS))
print("DEBUG: anomaly_detector.py exists? ->", os.path.exists(os.path.join(COMPONENTS, "anomaly_detector.py")))

# try import with helpful traceback if it fails
try:
    from components.anomaly_detector import detect_anomalies
except Exception as e:
    st.error("❌ Failed to import components.anomaly_detector — check terminal for traceback.")
    print("IMPORT ERROR (components.anomaly_detector):", repr(e))
    traceback.print_exc()
    raise

def generate_issue_insights(df):
    insights = []
    fixes = []
    # Check for missing data
    for col in df.columns:
        missing = df[col].isna().mean()
        if missing > 0.5:
            insights.append(f"⚠️ Column '{col}' has {missing*100:.1f}% missing values.")
    # Check for fraud
    if 'FraudFlag' in df.columns:
        fraud_rate = (df['FraudFlag'].astype(str) == "True").mean()
        if fraud_rate > 0.01:
            insights.append(f"⚠️ Fraud rate is {fraud_rate*100:.2f}% of all rows.")
            fixes.append("Investigate top fraud payment methods, customers, and states.")
    # Check for large losses
    if 'Profit' in df.columns:
        loss_cnt = (df['Profit'] < 0).sum()
        if loss_cnt > 0:
            insights.append(f"🚩 There are {loss_cnt} transactions with negative profit.")
            fixes.append("Review reasons for negative profit: refund, discount, chargeback, etc.")
    # Add more checks as needed
    return insights, fixes
