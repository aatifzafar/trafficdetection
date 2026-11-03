import datetime

import cv2
import numpy as np

from backend.database.database import SessionLocal
from backend.db_models.models import Evidence, Violation
from backend.services import citizen_service as cs


def test_photo_upload_becomes_pending_citizen_report(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(cs.settings, "EVIDENCE_DIR", str(tmp_path / "evidence"))
    img = np.full((120, 160, 3), 90, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    path = cs.save_upload("proof.jpg", buf.tobytes())
    when = datetime.datetime(2026, 11, 1, 10, 30)
    res = cs.process_upload(path, {"violation_type": "NO_HELMET", "plate": "ka 05 mx 4821", "location": "Silk Board",
                                   "when_utc": when, "name": "Asha", "phone": "9876543210", "note": "pillion"})
    assert res["frames"] == 0 and len(res["records"]) == 1 and not res["records"][0]["auto"]
    db = SessionLocal()
    try:
        v = db.query(Violation).filter(Violation.id == res["records"][0]["id"]).one()
        assert v.source == "CITIZEN" and v.status == "PENDING_REVIEW"
        assert v.plate_number == "KA05MX4821" and v.timestamp == when and v.reporter_name == "Asha"
        ev = db.query(Evidence).filter(Evidence.violation_id == v.id).one()
        assert ev.image_path and cv2.imread(ev.image_path) is not None
    finally:
        db.close()


def test_helpers():
    assert cs.mask_phone("98765 43210") == "••••••3210"
    assert cs.clean_plate(" ka-01 ab 1234 ") == "KA01AB1234"
    try:
        cs.save_upload("notes.txt", b"x")
        assert False, "text files must be refused"
    except ValueError:
        pass
