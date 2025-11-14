import duckdb, os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager

PG_DSN = os.getenv("PG_DSN", "postgresql+psycopg2://postgres:postgres@pg:5432/inai")
engine = create_engine(PG_DSN, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)

DUCK_PATH = os.getenv("DUCK_PATH", "/app/data/duck/analytics.duckdb")
os.makedirs(os.path.dirname(DUCK_PATH), exist_ok=True)
duck = duckdb.connect(DUCK_PATH)

# Example RLS hook: apply user_id or org_id filters into SQL if needed.
def apply_rls(sql:str, user_ctx:dict) -> str:
    # simplistic example—extend to map per-dataset rules
    return sql  # replace with injected WHERE if required

@contextmanager
def session_scope():
    s = SessionLocal()
    try:
        yield s
        s.commit()
    except:
        s.rollback()
        raise
    finally:
        s.close()
