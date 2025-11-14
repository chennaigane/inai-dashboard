# backend/app/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import os

app = FastAPI(title="AI Dashboard Backend (staging)")

class Dashboard(BaseModel):
    id: str
    name: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/dashboards")
def list_dashboards():
    # placeholder: in production read from Snowflake via connector
    return [
        {"id": "d1", "name": "Sales Overview"},
        {"id": "d2", "name": "Traffic Summary"}
    ]

@app.post("/dashboards")
def create_dashboard(d: Dashboard):
    # placeholder: persist to DB in prod
    return {"created": d}

# backend/app/main.py
from fastapi import FastAPI

app = FastAPI(title="AI Dashboard Backend")

@app.get("/health")
def health():
    return {"status": "ok"}

# include routers here (after app is created)
# from . import routers
# app.include_router(routers.some_router)
