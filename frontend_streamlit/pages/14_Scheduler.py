# frontend_streamlit/pages/12_Scheduler.py
import os, json, time, uuid
import streamlit as st
from apscheduler.schedulers.background import BackgroundScheduler
from components.database import get_duckdb

JOB_PATH = "data/scheduled_jobs.json"
os.makedirs("data", exist_ok=True)

def load_jobs():
    if os.path.exists(JOB_PATH):
        return json.load(open(JOB_PATH, "r", encoding="utf-8"))
    return []

def save_jobs(items):
    json.dump(items, open(JOB_PATH, "w", encoding="utf-8"), indent=2, default=str)

st.title("⏰ Scheduler")

# Singleton scheduler (lives while app session is running)
if "scheduler" not in st.session_state:
    st.session_state["scheduler"] = BackgroundScheduler(daemon=True)
    st.session_state["scheduler"].start()
sched = st.session_state["scheduler"]

jobs = load_jobs()

# List jobs
st.subheader("Existing Jobs")
for j in jobs:
    st.write(f"- **{j['name']}** | cron: `{j['cron']}` | query: `{j['sql'][:60]}...` | last_run: {j.get('last_run')}")
    if st.button(f"Run now: {j['name']}", key=f"run_{j['id']}"):
        try:
            con = get_duckdb()
            con.execute(j["sql"]).fetchdf()  # you may store result if needed
            j["last_run"] = int(time.time())
            save_jobs(jobs)
            st.success("Executed.")
        except Exception as e:
            st.error(str(e))
    if st.button(f"Delete: {j['name']}", key=f"del_{j['id']}"):
        jobs = [x for x in jobs if x["id"] != j["id"]]
        save_jobs(jobs)
        st.experimental_rerun()

st.subheader("Create Job")
name = st.text_input("Name")
sql = st.text_area("SQL to run", height=150, placeholder="SELECT COUNT(*) FROM dataset;")
cron = st.text_input("Cron (APScheduler style)", value="*/5 * * * *")  # every 5 minutes via CronTrigger

if st.button("Add Job"):
    if name and sql:
        job = {"id": str(uuid.uuid4()), "name": name, "sql": sql, "cron": cron}
        jobs.append(job); save_jobs(jobs)
        st.success("Scheduled (note: for real cron, wire to backend worker).")
    else:
        st.warning("Name and SQL required.")
