# frontend_streamlit/pages/15_Settings.py
import os, json
import streamlit as st

SETTINGS_PATH = "data/settings.json"
os.makedirs("data", exist_ok=True)

def load_settings():
    if os.path.exists(SETTINGS_PATH):
        return json.load(open(SETTINGS_PATH, "r", encoding="utf-8"))
    return {"branding": {"title": "INAI", "logo": ""}, "privacy": {"share_links": True}, "email": {"smtp": "localhost", "port": 25}}

def save_settings(s):
    json.dump(s, open(SETTINGS_PATH, "w", encoding="utf-8"), indent=2)

st.title("⚙️ Settings")

s = load_settings()

st.subheader("Branding")
s["branding"]["title"] = st.text_input("App title", value=s["branding"].get("title","INAI"))
s["branding"]["logo"]  = st.text_input("Logo URL (optional)", value=s["branding"].get("logo",""))

st.subheader("Privacy & Sharing")
s["privacy"]["share_links"] = st.checkbox("Allow shareable links", value=s["privacy"].get("share_links", True))

st.subheader("Email (for Alerts)")
s["email"]["smtp"] = st.text_input("SMTP server", value=s["email"].get("smtp","localhost"))
s["email"]["port"] = st.number_input("SMTP port", value=int(s["email"].get("port",25)))

if st.button("Save Settings"):
    save_settings(s); st.success("Saved.")
