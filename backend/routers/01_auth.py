from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, EmailStr
from passlib.hash import bcrypt
from jose import jwt
from datetime import datetime, timedelta
from sqlalchemy import select
from models import User, Org, Base
from db import engine, session_scope

SECRET="supersecret"  # put in env

router = APIRouter()

class Signup(BaseModel):
    org_name:str
    email:EmailStr
    password:str

class Signin(BaseModel):
    email:EmailStr
    password:str

@router.on_event("startup")
def init():
    Base.metadata.create_all(bind=engine)

@router.post("/signup")
def signup(body:Signup):
    with session_scope() as s:
        org = Org(name=body.org_name)
        s.add(org); s.flush()
        user = User(org_id=org.id, email=body.email, password_hash=bcrypt.hash(body.password), role="admin")
        s.add(user)
        return {"ok": True}

@router.post("/signin")
def signin(body:Signin):
    with session_scope() as s:
        user = s.execute(select(User).where(User.email==body.email)).scalar_one_or_none()
        if not user or not bcrypt.verify(body.password, user.password_hash):
            raise HTTPException(401, "Invalid credentials")
        token = jwt.encode({"sub": user.email, "uid": user.id, "org": user.org_id, "exp": datetime.utcnow()+timedelta(hours=12)}, SECRET)
        return {"token": token, "role": user.role}
