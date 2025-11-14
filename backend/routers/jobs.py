from fastapi import APIRouter
from pydantic import BaseModel
from apscheduler.schedulers.background import BackgroundScheduler
from db import duck, session_scope
from models import Job
import httpx, os

router = APIRouter()
sched = BackgroundScheduler()
sched.start()

SLACK_WEBHOOK = os.getenv("SLACK_WEBHOOK")
EMAIL_WEBHOOK = os.getenv("EMAIL_WEBHOOK")  # send via your email service/webhook

class SaveJob(BaseModel): type:str; cron:str; payload:dict; org_id:int=1

def run_alert(payload:dict):
    sql = payload["sql"]; comparator = payload.get("comparator", ">="); threshold = float(payload["threshold"])
    val = duck.execute(sql).fetchone()[0]
    ok = eval(f"{val} {comparator} {threshold}")
    if ok and SLACK_WEBHOOK:
        httpx.post(SLACK_WEBHOOK, json={"text": f"Alert triggered: {sql} => {val} {comparator} {threshold}"})

@router.post("/save")
def save_job(body:SaveJob):
    with session_scope() as s:
        j = Job(org_id=body.org_id, type=body.type, cron=body.cron, payload=body.payload, enabled=True)
        s.add(j); s.flush()
        sched.add_job(lambda: run_alert(body.payload), trigger="cron", **cron_to_kwargs(body.cron), id=f"job-{j.id}", replace_existing=True)
        return {"ok": True, "id": j.id}

def cron_to_kwargs(expr:str):
    # "*/5 * * * *" -> APScheduler args
    m,h,dom,mon,dow = expr.split()
    kw={}
    if m!="*": kw["minute"]=m
    if h!="*": kw["hour"]=h
    if dom!="*": kw["day"]=dom
    if mon!="*": kw["month"]=mon
    if dow!="*": kw["day_of_week"]=dow
    return kw
