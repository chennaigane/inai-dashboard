import streamlit as st, requests

st.title("Sign In")
email = st.text_input("Email"); pwd = st.text_input("Password", type="password")
if st.button("Sign In"):
    r = requests.post(st.secrets["api_url"]+"/auth/signin", json={"email":email,"password":pwd})
    if r.status_code==200:
        st.session_state["token"]=r.json()["token"]
        st.success("Signed in")
    else:
        st.error("Invalid credentials")
