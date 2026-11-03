"""
Remove rows that earlier test runs wrote into the real database.

    python scripts/clean_test_data.py          # shows what would be deleted
    python scripts/clean_test_data.py --apply  # deletes it
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sqlalchemy import or_
from backend.database.database import SessionLocal, ensure_schema
from backend.db_models.models import Violation, Evidence, Alert, CameraSource


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="actually delete (default: dry run)")
    args = ap.parse_args()

    ensure_schema()
    db = SessionLocal()
    try:
        viols = db.query(Violation).filter(or_(
            Violation.plate_number.like("%TEST"),
            Violation.event_id.like("EVT-TEST-%"),
            Violation.event_id.like("EVT-REVIEW-%"),
        )).all()
        cams = db.query(CameraSource).filter(CameraSource.name == "Test Test Junction Camera").all()
        print(f"Test violations found: {len(viols)}")
        print(f"Test cameras found:    {len(cams)}")
        if not args.apply:
            print("Dry run. Re-run with --apply to delete them.")
            return
        ids = [v.id for v in viols]
        if ids:
            db.query(Evidence).filter(Evidence.violation_id.in_(ids)).delete(synchronize_session=False)
            db.query(Alert).filter(Alert.violation_id.in_(ids)).delete(synchronize_session=False)
            db.query(Violation).filter(Violation.id.in_(ids)).delete(synchronize_session=False)
        for c in cams:
            db.delete(c)
        db.commit()
        print("Deleted.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
