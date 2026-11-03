import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

load_dotenv()

# Default SQLite fallback database for zero-friction local execution
DEFAULT_DB_URL = "sqlite:///./data/traffic_system.db"
DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_DB_URL)

# Ensure data directory exists for SQLite
if DATABASE_URL.startswith("sqlite"):
    os.makedirs("./data", exist_ok=True)
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False}
    )
else:
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Columns added after the first release. ensure_schema() adds any that are missing,
# so an existing data/traffic_system.db keeps working without deleting it.
_NEW_COLUMNS = {
    "violations": {
        "reviewed_by": "VARCHAR(100)",
        "reviewed_at": "TIMESTAMP",
        "reject_reason": "VARCHAR(200)",
        "plate_edited": "BOOLEAN DEFAULT 0",
        "source": "VARCHAR(20) DEFAULT 'CCTV'",
        "merged_count": "INTEGER DEFAULT 0",
        "light_condition": "VARCHAR(10)",
        "reporter_name": "VARCHAR(100)",
        "reporter_phone": "VARCHAR(20)",
        "report_note": "TEXT",
    },
}

# Old status names -> review workflow names
_STATUS_MIGRATION = {
    "CONFIRMED": "APPROVED",
    "CHALLAN_ISSUED": "APPROVED",
    "DISMISSED": "REJECTED",
}


def ensure_schema():
    """Create tables, add missing columns and map old status values. Safe to call repeatedly."""
    from sqlalchemy import inspect, text
    import backend.db_models.models  # noqa: F401  (registers tables)

    Base.metadata.create_all(bind=engine)
    insp = inspect(engine)
    with engine.begin() as conn:
        for table, cols in _NEW_COLUMNS.items():
            if not insp.has_table(table):
                continue
            existing = {c["name"] for c in insp.get_columns(table)}
            for name, ddl in cols.items():
                if name not in existing:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}"))
        for old, new in _STATUS_MIGRATION.items():
            conn.execute(text("UPDATE violations SET status = :new WHERE status = :old"), {"new": new, "old": old})
