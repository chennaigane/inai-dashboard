from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from pydantic import BaseModel
import pandas as pd, os, io, duckdb, pyarrow as pa, pyarrow.parquet as pq
from sqlalchemy import select
from models import Dataset, User, AuditLog
from db import duck, session_scope

router = APIRouter()
DATA_ROOT = "/app/data/parquet"
os.makedirs(DATA_ROOT, exist_ok=True)

class CatalogEntry(BaseModel):
    id:int; name:str; rows:int; bytes:int; storage_path:str; schema_json:dict; lineage:dict

@router.post("/upload")
async def upload(name:str = Form(...), file: UploadFile = File(...), owner_id:int = Form(...)):
    content = await file.read()
    # CSV/Excel autodetect
    if file.filename.lower().endswith(".xlsx"):
        df = pd.read_excel(io.BytesIO(content))
    else:
        df = pd.read_csv(io.BytesIO(content))
    table = pa.Table.from_pandas(df)
    dst = os.path.join(DATA_ROOT, f"{name}.parquet")
    pq.write_table(table, dst)
    duck.execute(f"CREATE VIEW IF NOT EXISTS {duckdb_valid(name)} AS SELECT * FROM '{dst}'")

    with session_scope() as s:
        ds = Dataset(name=name, owner_id=owner_id, org_id=1, storage_path=dst,
                     rows=len(df), bytes=os.path.getsize(dst),
                     schema_json=df.dtypes.astype(str).to_dict(),
                     lineage={"source":"upload","filename":file.filename})
        s.add(ds); s.flush()
        s.add(AuditLog(org_id=1, user_id=owner_id, action="CREATE_DATASET", details={"dataset":name}))
        return {"ok": True, "dataset_id": ds.id}

def duckdb_valid(name:str)->str:
    return ''.join(ch if ch.isalnum() or ch=='_' else '_' for ch in name)

@router.post("/register_sheet")
def register_sheet(name:str, sheet_url:str, owner_id:int):
    # DuckDB reads Google Sheets via CSV export (public or with service account—keep simple)
    duck.execute(f"CREATE OR REPLACE VIEW {duckdb_valid(name)} AS SELECT * FROM read_csv_auto('{sheet_url}')")
    with session_scope() as s:
        ds = Dataset(name=name, owner_id=owner_id, org_id=1, storage_path=sheet_url, engine="duckdb",
                     rows=0, bytes=0, schema_json={}, lineage={"source":"google_sheets"})
        s.add(ds); s.flush()
        return {"ok": True, "dataset_id": ds.id}

@router.post("/ingest_from_db")
def ingest_from_db(name:str, sql:str, conn_url:str, owner_id:int):
    # Any SQLAlchemy DB URL (postgresql+psycopg2://..., mysql+pymysql://..., mssql+pyodbc://...)
    con = duckdb.connect()
    con.execute(f"ATTACH '{conn_url}' AS ext (TYPE SQLALCHEMY);")
    df = con.execute(f"SELECT * FROM ext.query('{sql.replace(\"'\",\"''\")}')").fetch_df()
    dst = os.path.join(DATA_ROOT, f"{name}.parquet")
    pq.write_table(pa.Table.from_pandas(df), dst)
    duck.execute(f"CREATE OR REPLACE VIEW {duckdb_valid(name)} AS SELECT * FROM '{dst}'")
    with session_scope() as s:
        ds = Dataset(name=name, owner_id=owner_id, org_id=1, storage_path=dst,
                     rows=len(df), bytes=os.path.getsize(dst),
                     schema_json=df.dtypes.astype(str).to_dict(),
                     lineage={"source":"external_db","query":sql})
        s.add(ds); s.flush()
        return {"ok": True, "dataset_id": ds.id}

@router.get("/catalog", response_model=list[CatalogEntry])
def catalog():
    with session_scope() as s:
        rows = s.execute(select(Dataset)).scalars().all()
        return [CatalogEntry(id=r.id, name=r.name, rows=r.rows, bytes=r.bytes, storage_path=r.storage_path, schema_json=r.schema_json, lineage=r.lineage) for r in rows]

@router.post("/snapshot")
def snapshot(dataset_name:str, partition_by:str="ingest_date"):
    # Materialize to partitioned Parquet (simple snapshot)
    res = duck.execute(f"SELECT *, current_date() as {partition_by} FROM {duckdb_valid(dataset_name)}").arrow()
    dst_dir = os.path.join(DATA_ROOT, dataset_name+"_snap")
    os.makedirs(dst_dir, exist_ok=True)
    pq.write_to_dataset(res, root_path=dst_dir, partition_cols=[partition_by])
    return {"ok": True, "path": dst_dir}
