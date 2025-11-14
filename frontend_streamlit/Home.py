import streamlit as st
import plotly.express as px
import pandas as pd
from PIL import Image
from streamlit_option_menu import option_menu

import streamlit as st
import duckdb

# frontend_streamlit/Home.py
import streamlit as st


st.title("INAI Home")
st.caption("Welcome to INAI — upload data, run SQL, build dashboards.")


# --- Custom CSS for Enterprise Glassy Sidebar and Navbar ---

st.markdown("""
<style>
body { background: #f5f7fb !important; font-family: 'Segoe UI','Roboto',Arial,sans-serif;}
/* Top navigation bar */
.top-bar {
    background: linear-gradient(90deg, #193AA3 10%, #233753 66%, #d32f2f 100%);
    color: #fff; padding: 10px 44px 16px 44px; border-radius: 26px 26px 0 0;
    display: flex; align-items: center; justify-content: space-between;
    box-shadow: 0 4px 32px #193aa355;
    font-size: 1.22rem;
    margin-bottom: 24px;
}
.nav-btns a {
    color: #fff; font-weight:500; text-decoration:none; margin-left: 40px;
    padding: 10px 24px; font-size:1.04rem; border-radius:8px;
    transition: background 0.25s, color 0.25s;
    background: linear-gradient(92deg, #f7f7fb22 0, #d32f2f22 100%);
}
.nav-btns a:hover { background:#285175c6; color:#ffe8e8;}
/* Glassmorphic Sidebar */
.css-1d391kg, .stSidebar, section[data-testid="stSidebar"] {
    background: linear-gradient(146deg, #1c2445aa 60%, #243753bf 100%);
    box-shadow: 0 10px 50px #26306690, 0 0 0 #fff0;
    border-top-right-radius:28px;
    border-bottom-right-radius:28px;
    color: #e0e3f1 !important;
    border: none;
}

.sidebar-tier.pro { border-left-color:#d32f2f;}
.sidebar-tier.int { border-left-color:#43d28c;}
.tier-title { font-size:1.09rem; font-weight:700; margin-bottom:4px; color: #fff;}
.tier-desc{ font-size:1rem;}
.sidebar-logo { margin-bottom:32px; }
</style>
""", unsafe_allow_html=True)

# --- Render Nav Bar (Optional Logo in Bar) ---
st.markdown("""
    <div class="top-bar">
        <span class="nav-btns">
            <a href="#">Platforms</a>
            <a href="#">Features</a>
            <a href="#">Solutions</a>
            <a href="#">Pricing</a>
            <a href="#">Resources</a>
        </span>
    </div>
""", unsafe_allow_html=True)


# ✅ Remove all page padding and margins
st.markdown("""
    <style>
        /* Remove default Streamlit white padding around content */
        .block-container {
            padding: 0rem 0rem 0rem 0rem;
            margin: 0;
            max-width: 100%;
        }

        /* Make top section (header/banner) align perfectly left */
        .main {
            padding-left: 0rem !important;
            padding-right: 0rem !important;
        }

        /* Optional: remove faint white shadow background from sides */
        section[data-testid="stSidebar"] + section {
            background-color: #ffffff !important;
            box-shadow: none !important;
        }

        /* Optional: make background match */
        .stApp {
            background-color: #ffffff;
        }
    </style>
""", unsafe_allow_html=True)


# --- Welcome Card ---

from PIL import Image
import streamlit as st

# Load and encode the image as base64 so it works everywhere
import base64

with open("Datasci.jpg", "rb") as f:
    img_bytes = f.read()
    img_base64 = base64.b64encode(img_bytes).decode()

st.markdown(f"""
<style>
.hero-wrapper {{
    position: relative;
    width: 100%;
    min-height: 320px;
    height: 330px;
    border-radius: 28px;
    overflow: hidden;
    margin-bottom: 36px;
    margin-top: 10px;
    box-shadow: 0 7px 30px #23375330;
    background: #10162b;
}}
.hero-bg-image {{
    position: absolute;
    z-index: 0;
    top: 0; left: 0; right: 0; bottom: 0;
    width: 100%; height: 100%;
    object-fit: cover;
    opacity: 0.54;
    filter: blur(0.8px) brightness(0.92);
    border-radius: 28px;
}}
.hero-content {{
    position: relative;
    z-index: 2;
    padding: 44px 38px 38px 58px;
    color: #fff;
    width: 80%;
    max-width: 900px;
    min-width: 340px;
}}
.hero-title {{
    font-size: 2.65rem;
    font-weight: 800;
    letter-spacing: 2px;
    margin-bottom: 2px;
    text-shadow:0 1px 16px #09144180;
    line-height: 3.1rem;
}}
.hero-title .brand {{
    background: linear-gradient(90deg,#206afc,#a94768 80%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}}
.hero-subhead {{
    font-size: 1.39rem;
    font-weight: 600;
    letter-spacing:0.5px;
    margin-bottom: 16px;
    color: #f6f6ff;
    text-shadow:0 1px 8px #253d6055;
}}
.hero-desc {{
    font-size: 1.11rem;
    color: #f5dcc4;
    text-shadow:0 1px 16px #2225;
}}
</style>
<div class='hero-wrapper'>
  <img src="data:image/jpg;base64,{img_base64}" class="hero-bg-image" />
  <div class='hero-content'>
    <div class='hero-title'>
      Welcome to <span class="brand">InaI </span>
    </div>
    <div class='hero-subhead'>
      AI Data Assistant — <span style="opacity:0.84;"></span>
    </div>
    <div class='hero-desc'>
     From Data to Decisions - An Agentic AI Self-Service Analytics Platform
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# --- Demo Feature Chart ---
st.subheader("Demo AI Data Visualization")
df = pd.DataFrame({
    "Month": ["Jan", "Feb", "Mar", "Apr", "May","June"],
    "Sales": [1200,1800,1750,2450,2600,3000],
    "Profit": [510,690,825,900,1350,1600]
})
fig = px.bar(df, x="Month", y=["Sales","Profit"], barmode="group", height=360,
    color_discrete_map={"Sales":"#193AA3","Profit":"#d32f2f"})
fig.update_layout(title="Monthly Performance", plot_bgcolor="#f5f7fb", paper_bgcolor="#f5f7fb", font=dict(family='Segoe UI,Roboto,Arial'))
st.plotly_chart(fig, use_container_width=True)


# --- Remove or disable sidebar ---
hide_streamlit_style = """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .css-1y0tads {visibility: hidden;} /* Hide sidebar */
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


# --- Custom CSS for glossy boxes and rich gradient backgrounds ---
rich_css = """
<style>
.plan-section {
    display: flex;
    justify-content: center;
    margin-top: 40px;
    gap: 2rem;
}
.plan-box {
    min-width: 270px;
    max-width: 320px;
    padding: 1.8rem 1.3rem 1.9rem 1.3rem;
    border-radius: 18px;
    background: linear-gradient(135deg, #153659, #26467a 70%, #d82c38 100%);
    box-shadow: 0 4px 20px 0 rgba(40,60,100,0.16), 0 1.5px 4px rgba(216,44,56,0.07);
    color: #fff;
    position: relative;
    overflow: hidden;
    margin-bottom: 12px;
}
.plan-title {
    font-size: 1.35rem;
    font-weight: 600;
    letter-spacing: 0.03rem;
    margin-bottom: 8px;
}
.plan-users {
    font-size: 1rem;
    margin-bottom: 11px;
    color: #F7BD51;
    font-weight: 500;
}
.plan-desc {
    font-size: 1.05rem;
    margin-bottom: 0;
    color: #f0f2f7;
    line-height: 1.5;
}
.plan-box.basic {
    background: linear-gradient(135deg, #195194 60%, #d82c38 100%);
}
.plan-box.intermediate {
    background: linear-gradient(135deg, #18243c 60%, #6EC1E4 100%);
}
.plan-box.pro {
    background: linear-gradient(135deg, #203040 50%, #d82c38 90%);
    box-shadow: 0 8px 32px rgba(216,44,56,0.23), 0 2px 8px rgba(21,54,89,0.18);
    border: 2.5px solid #F7BD51;
}
</style>
"""

# --- Sign In Button below About InaI ---
if st.button("Sign Up"):
    st.switch_page("pages/SignUp.py")

if st.button("Go to 3_Data Upload"):
    st.switch_page("pages/3_DataUpload.py")
