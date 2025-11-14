from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from db import duck, apply_rls, session_scope
from models import SavedQuery, AuditLog
from sqlalchemy import select
import pandas as pd

router = APIRouter()

class RunSQL(BaseModel):
    sql:str
    params:dict|None=None
    user_id:int=1
    org_id:int=1
    limit:int=5000

@router.post("/run")
def run(body:RunSQL):
    sql = body.sql
    if body.params:
        for k,v in body.params.items():
            sql = sql.replace(f":{k}", repr(v))
    sql = apply_rls(sql, {"user_id":body.user_id, "org_id":body.org_id})
    df = duck.execute(sql).fetch_df()
    with session_scope() as s:
        s.add(AuditLog(org_id=body.org_id, user_id=body.user_id, action="RUN_SQL", details={"sql":sql, "rows":len(df)}))
    return {"columns": df.columns.tolist(), "rows": df.head(body.limit).to_dict("records")}

class Save(BaseModel):
    name:str; sql:str; params:dict|None=None; dataset_id:int; org_id:int=1; owner_id:int=1; is_public:bool=False

@router.post("/save")
def save(body:Save):
    with session_scope() as s:
        sq = SavedQuery(org_id=body.org_id, dataset_id=body.dataset_id, name=body.name, sql=body.sql, params=body.params or {}, owner_id=body.owner_id, is_public=body.is_public)
        s.add(sq); s.flush()
        return {"ok": True, "id": sq.id}

@router.get("/list")
def list_queries(org_id:int=1):
    with session_scope() as s:
        return [ {"id":q.id,"name":q.name,"sql":q.sql,"params":q.params} for q in s.execute(select(SavedQuery).where(SavedQuery.org_id==org_id)).scalars().all() ]
