# frontend_streamlit/pages/13_Alerts.py
import os, json, time, requests, smtplib
from email.mime.text import MIMEText
import streamlit as st
import pandas as pd

ALERTS_PATH = "data/alerts.json"
os.makedirs("data", exist_ok=True)

def load_alerts():
    if os.path.exists(ALERTS_PATH):
        return json.load(open(ALERTS_PATH, "r", encoding="utf-8"))
    return []

def save_alerts(items):
    json.dump(items, open(ALERTS_PATH, "w", encoding="utf-8"), indent=2)

def eval_alert(df: pd.DataFrame, rule: dict) -> dict:
    """
    rule = { "column":"Revenue", "op":">", "value":1000, "window":"all" }
    """
    if df is None or df.empty or rule["column"] not in df.columns:
        return {"ok": False, "reason": "No data/column."}
    col = pd.to_numeric(df[rule["column"]], errors="coerce")
    if rule["op"] == ">":  triggered = (col > rule["value"]).any()
    elif rule["op"] == "<": triggered = (col < rule["value"]).any()
    elif rule["op"] == ">=": triggered = (col >= rule["value"]).any()
    elif rule["op"] == "<=": triggered = (col <= rule["value"]).any()
    elif rule["op"] == "==": triggered = (col == rule["value"]).any()
    else: triggered = False
    return {"ok": True, "triggered": bool(triggered)}

def send_email(to_addr, subject, body, smtp_server="localhost", smtp_port=25):
    try:
        msg = MIMEText(body, "plain")
        msg["Subject"] = subject; msg["From"] = "alerts@inai"; msg["To"] = to_addr
        with smtplib.SMTP(smtp_server, smtp_port, timeout=10) as s:
            s.send_message(msg)
        return True, "sent"
    except Exception as e:
        return False, str(e)

def send_slack(webhook_url, text):
    try:
        r = requests.post(webhook_url, json={"text": text}, timeout=10)
        return r.status_code == 200, r.text
    except Exception as e:
        return False, str(e)

st.title("🚨 Alerts")

df = st.session_state.get("validated_df") or st.session_state.get("uploaded_df")
alerts = load_alerts()

st.subheader("Create Alert")
col = st.selectbox("Column", df.columns if isinstance(df, pd.DataFrame) else [])
op  = st.selectbox("Operator", [">", "<", ">=", "<=", "=="])
val = st.number_input("Threshold", value=0.0)
channel = st.selectbox("Notify via", ["UI", "Email", "Slack"])
email_to = st.text_input("Email To", value="", placeholder="user@example.com") if channel=="Email" else ""
slack_url = st.text_input("Slack webhook", value="", placeholder="https://hooks.slack.com/...") if channel=="Slack" else ""
if st.button("Add Alert"):
    rule = {"column": col, "op": op, "value": val, "channel": channel, "email": email_to, "slack": slack_url}
    alerts.append(rule); save_alerts(alerts); st.success("Alert saved.")

st.subheader("Evaluate Alerts Now")
for i, rule in enumerate(alerts):
    res = eval_alert(df, rule)
    if not res.get("ok"):
        st.warning(f"{rule} — {res['reason']}")
        continue
    if res["triggered"]:
        msg = f"Alert triggered: {rule['column']} {rule['op']} {rule['value']}"
        if rule["channel"] == "UI":
            st.error(msg)
        elif rule["channel"] == "Email" and rule["email"]:
            ok, info = send_email(rule["email"], "INAI Alert", msg)
            st.write("Email:", "✅" if ok else f"❌ {info}")
        elif rule["channel"] == "Slack" and rule["slack"]:
            ok, info = send_slack(rule["slack"], msg)
            st.write("Slack:", "✅" if ok else f"❌ {info}")
    else:
        st.success(f"OK: {rule['column']} {rule['op']} {rule['value']}")
