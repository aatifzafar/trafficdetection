"""
PostgreSQL Setup & Migration Script for TrafficAI

Usage:
    python scripts/setup_postgres.py --user postgres --password YOUR_PASSWORD --host localhost --port 5432 --dbname traffic_system
    python scripts/setup_postgres.py --migrate-from-sqlite  # copies existing records from SQLite to PostgreSQL

This script:
1. Connects to PostgreSQL server.
2. Creates the target database (default: traffic_system) if it doesn't exist.
3. Automatically creates all tables and schemas via SQLAlchemy.
4. Optionally migrates existing cameras, violations, evidence, and vehicles from SQLite.
5. Writes/updates DATABASE_URL in .env so both FastAPI and Streamlit automatically use PostgreSQL.
"""

import argparse
import os
import sys
import urllib.parse
from pathlib import Path
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.database.database import Base
import backend.db_models.models as models


def create_database_if_not_exists(user, password, host, port, dbname):
    print(f"Connecting to PostgreSQL server at {host}:{port} as user '{user}'...")
    conn = psycopg2.connect(
        dbname="postgres",
        user=user,
        password=password,
        host=host,
        port=port,
        connect_timeout=5,
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = conn.cursor()

    cursor.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s", (dbname,))
    exists = cursor.fetchone()
    if not exists:
        print(f"Creating database '{dbname}'...")
        cursor.execute(f'CREATE DATABASE "{dbname}"')
        print(f"Database '{dbname}' created successfully.")
    else:
        print(f"Database '{dbname}' already exists.")

    cursor.close()
    conn.close()


def initialize_schema(target_db_url):
    print("Creating all database tables via SQLAlchemy metadata...")
    engine = create_engine(target_db_url, pool_pre_ping=True)
    Base.metadata.create_all(bind=engine)
    print("Tables created: " + ", ".join(Base.metadata.tables.keys()))
    return engine


def row_to_dict(obj):
    return {c.name: getattr(obj, c.name) for c in obj.__table__.columns}


def migrate_from_sqlite(sqlite_path, pg_engine):
    abs_sqlite = os.path.abspath(sqlite_path)
    if not os.path.exists(abs_sqlite):
        print(f"SQLite database not found at {abs_sqlite}. Skipping migration.")
        return

    print(f"Migrating existing data from SQLite ({abs_sqlite}) to PostgreSQL...")
    sqlite_engine = create_engine(f"sqlite:///{abs_sqlite}", connect_args={"check_same_thread": False})
    SqliteSession = sessionmaker(bind=sqlite_engine)
    PgSession = sessionmaker(bind=pg_engine)

    sqlite_db = SqliteSession()
    pg_db = PgSession()

    try:
        # 1. CameraSource
        cams = sqlite_db.query(models.CameraSource).all()
        cam_count = 0
        for c in cams:
            if not pg_db.query(models.CameraSource).filter_by(id=c.id).first():
                pg_db.add(models.CameraSource(**row_to_dict(c)))
                cam_count += 1
        pg_db.commit()
        print(f"Migrated {cam_count} camera records.")

        # 2. Vehicle
        vehs = sqlite_db.query(models.Vehicle).all()
        veh_count = 0
        for v in vehs:
            if not pg_db.query(models.Vehicle).filter_by(id=v.id).first():
                pg_db.add(models.Vehicle(**row_to_dict(v)))
                veh_count += 1
        pg_db.commit()
        print(f"Migrated {veh_count} vehicle records.")

        # 3. Violation
        viols = sqlite_db.query(models.Violation).all()
        viol_count = 0
        for v in viols:
            if not pg_db.query(models.Violation).filter_by(id=v.id).first():
                pg_db.add(models.Violation(**row_to_dict(v)))
                viol_count += 1
        pg_db.commit()
        print(f"Migrated {viol_count} violation records.")

        # 4. Evidence
        evs = sqlite_db.query(models.Evidence).all()
        ev_count = 0
        for e in evs:
            if not pg_db.query(models.Evidence).filter_by(id=e.id).first():
                pg_db.add(models.Evidence(**row_to_dict(e)))
                ev_count += 1
        pg_db.commit()
        print(f"Migrated {ev_count} evidence records.")

        # 5. Alert
        alerts = sqlite_db.query(models.Alert).all()
        alert_count = 0
        for a in alerts:
            if not pg_db.query(models.Alert).filter_by(id=a.id).first():
                pg_db.add(models.Alert(**row_to_dict(a)))
                alert_count += 1
        pg_db.commit()
        print(f"Migrated {alert_count} alert records.")

        # Reset sequences in PostgreSQL so next auto-increments start after max(id)
        with pg_engine.begin() as conn:
            from sqlalchemy import text
            for table in ["camera_sources", "vehicles", "violations", "evidence", "alerts"]:
                conn.execute(text(f"SELECT setval(pg_get_serial_sequence('{table}', 'id'), COALESCE(MAX(id), 1)) FROM {table};"))

        print("Migration from SQLite to PostgreSQL completed successfully!")
    except Exception as e:
        pg_db.rollback()
        print(f"Error during migration: {e}")
    finally:
        sqlite_db.close()
        pg_db.close()


def update_env_file(pg_url):
    env_path = Path(".env")
    lines = []
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            lines = [l for l in f.readlines() if not l.startswith("DATABASE_URL=")]
    lines.append(f'DATABASE_URL="{pg_url}"\n')
    with open(env_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"Updated .env with DATABASE_URL={pg_url}")


def main():
    parser = argparse.ArgumentParser(description="PostgreSQL Database Creator & Migrator for TrafficAI")
    parser.add_argument("--user", default="postgres", help="PostgreSQL username (default: postgres)")
    parser.add_argument("--password", required=True, help="PostgreSQL password")
    parser.add_argument("--host", default="127.0.0.1", help="PostgreSQL host (default: 127.0.0.1)")
    parser.add_argument("--port", default=5432, type=int, help="PostgreSQL port (default: 5432)")
    parser.add_argument("--dbname", default="traffic_system", help="Database name to create/use (default: traffic_system)")
    parser.add_argument("--migrate-from-sqlite", action="store_true", help="Migrate data from existing SQLite database")
    parser.add_argument("--sqlite-path", default="data/traffic_system.db", help="Path to SQLite database")

    args = parser.parse_args()

    try:
        create_database_if_not_exists(
            user=args.user,
            password=args.password,
            host=args.host,
            port=args.port,
            dbname=args.dbname,
        )

        safe_password = urllib.parse.quote_plus(args.password)
        pg_url = f"postgresql://{args.user}:{safe_password}@{args.host}:{args.port}/{args.dbname}"
        pg_engine = initialize_schema(pg_url)

        if args.migrate_from_sqlite:
            migrate_from_sqlite(args.sqlite_path, pg_engine)

        update_env_file(pg_url)
        print("\nAll done! PostgreSQL is now initialized and configured as the primary database.")
    except Exception as e:
        print(f"\nSetup failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
