# frontend_streamlit/pages/14_UsersAndRoles.py
import os, json
import streamlit as st

USERS_PATH = "data/users_roles.json"
os.makedirs("data", exist_ok=True)

DEFAULT = {"workspaces": {"default": {"members": {}}}, "roles": ["viewer","editor","admin"]}

def load_users():
    if os.path.exists(USERS_PATH):
        return json.load(open(USERS_PATH, "r", encoding="utf-8"))
    return DEFAULT

def save_users(data):
    json.dump(data, open(USERS_PATH, "w", encoding="utf-8"), indent=2)

st.title("👥 Users & Roles")

data = load_users()
ws_names = list(data["workspaces"].keys())
ws = st.selectbox("Workspace", ws_names, index=ws_names.index("default") if "default" in ws_names else 0)

st.subheader("Members")
members = data["workspaces"][ws]["members"]
if not members:
    st.info("No members yet.")
else:
    st.table([{"user": u, "role": r} for u,r in members.items()])

st.subheader("Add / Update Member")
user = st.text_input("User email/ID")
role = st.selectbox("Role", data["roles"])
if st.button("Add/Update"):
    data["workspaces"][ws]["members"][user] = role
    save_users(data); st.success("Saved.")

st.subheader("Create Workspace")
new_ws = st.text_input("New workspace name")
if st.button("Create Workspace"):
    if new_ws and new_ws not in data["workspaces"]:
        data["workspaces"][new_ws] = {"members": {}}
        save_users(data); st.success("Created.")
    else:
        st.warning("Invalid or duplicate name.")
