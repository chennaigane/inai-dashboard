from fastapi import FastAPI
from routers import auth, datasets, queries, dashboards, jobs, admin
from loguru import logger

app = FastAPI(title="INAI Backend", version="0.2")

app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(datasets.router, prefix="/datasets", tags=["datasets"])
app.include_router(queries.router, prefix="/queries", tags=["queries"])
app.include_router(dashboards.router, prefix="/dashboards", tags=["dashboards"])
app.include_router(jobs.router, prefix="/jobs", tags=["scheduler-alerts"])
app.include_router(admin.router, prefix="/admin", tags=["admin"])

@app.get("/health")
def health():
    return {"ok": True}
