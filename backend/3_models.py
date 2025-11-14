from sqlalchemy.orm import declarative_base, relationship, Mapped, mapped_column
from sqlalchemy import String, Integer, DateTime, ForeignKey, JSON, Boolean, Text, UniqueConstraint, func

Base = declarative_base()

class Org(Base):
    __tablename__ = "orgs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(200), unique=True)

class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    org_id: Mapped[int] = mapped_column(ForeignKey("orgs.id"))
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(50), default="viewer")  # org-level role
    created_at: Mapped[DateTime] = mapped_column(DateTime, server_default=func.now())

class Dataset(Base):
    __tablename__ = "datasets"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    org_id: Mapped[int] = mapped_column(ForeignKey("orgs.id"))
    name: Mapped[str] = mapped_column(String(255))
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    storage_path: Mapped[str] = mapped_column(Text)  # Parquet folder or file
    engine: Mapped[str] = mapped_column(String(50), default="duckdb")
    rows: Mapped[int] = mapped_column(Integer, default=0)
    bytes: Mapped[int] = mapped_column(Integer, default=0)
    schema_json: Mapped[dict] = mapped_column(JSON, default={})
    lineage: Mapped[dict] = mapped_column(JSON, default={})
    created_at: Mapped[DateTime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[DateTime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())
    __table_args__ = (UniqueConstraint('org_id', 'name', name='_org_dataset_uc'),)

class SavedQuery(Base):
    __tablename__ = "saved_queries"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    org_id: Mapped[int] = mapped_column(ForeignKey("orgs.id"))
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"))
    name: Mapped[str] = mapped_column(String(255))
    sql: Mapped[str] = mapped_column(Text)
    params: Mapped[dict] = mapped_column(JSON, default={})
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    is_public: Mapped[bool] = mapped_column(Boolean, default=False)

class Dashboard(Base):
    __tablename__ = "dashboards"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    org_id: Mapped[int] = mapped_column(ForeignKey("orgs.id"))
    name: Mapped[str] = mapped_column(String(255))
    layout_json: Mapped[dict] = mapped_column(JSON, default={})  # list of cards with query_id, chart_type, options
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    is_public: Mapped[bool] = mapped_column(Boolean, default=False)

class Job(Base):
    __tablename__ = "jobs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    org_id: Mapped[int] = mapped_column(ForeignKey("orgs.id"))
    type: Mapped[str] = mapped_column(String(50))  # import|refresh|alert
    cron: Mapped[str] = mapped_column(String(90))
    payload: Mapped[dict] = mapped_column(JSON, default={})
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)

class AuditLog(Base):
    __tablename__ = "audit_log"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    org_id: Mapped[int] = mapped_column(ForeignKey("orgs.id"))
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    action: Mapped[str] = mapped_column(String(100))  # RUN_SQL, CREATE_DATASET, etc
    details: Mapped[dict] = mapped_column(JSON, default={})
    at: Mapped[DateTime] = mapped_column(DateTime, server_default=func.now())
