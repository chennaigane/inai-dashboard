from fastapi import APIRouter
from pydantic import BaseModel
from sqlalchemy import select
from db import session_scope
from models import Dashboard

router = APIRouter()

class SaveDash(BaseModel):
    name:str
    layout_json:dict
    org_id:int=1
    owner_id:int=1
    is_public:bool=False

@router.post("/save")
def save(body:SaveDash):
    with session_scope() as s:
        d = Dashboard(org_id=body.org_id, name=body.name, layout_json=body.layout_json, owner_id=body.owner_id, is_public=body.is_public)
        s.add(d); s.flush()
        return {"ok": True, "id": d.id}

@router.get("/list")
def list_(org_id:int=1):
    with session_scope() as s:
        return [{"id":d.id,"name":d.name,"layout":d.layout_json,"is_public":d.is_public} for d in s.execute(select(Dashboard).where(Dashboard.org_id==org_id)).scalars().all()]
