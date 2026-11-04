from pathlib import Path
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


def test_photo_upload_ai_auto_classification(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(cs.settings, "EVIDENCE_DIR", str(tmp_path / "evidence"))
    
    sample_path = Path("datasets/raw/05_triple_riding/valid/images/100_jpg.rf.1adbda0e75921ea1e7bd67534fee043d.jpg")
    if not sample_path.exists():
        return
        
    img_bytes = sample_path.read_bytes()
    upload_path = cs.save_upload("proof_triple.jpg", img_bytes)
    
    info = {
        "violation_type": "NO_HELMET",
        "plate": "KA01AB1234",
        "location": "MG Road",
        "name": "Test User",
        "phone": "9999988888",
    }
    
    res = cs.process_upload(upload_path, info)
    assert res["ai_detected"] is True
    assert len(res["records"]) == 2
    rec_types = {r["type"] for r in res["records"]}
    assert "TRIPLE_RIDING" in rec_types
    assert "NO_HELMET" in rec_types
    for r in res["records"]:
        assert r["confidence"] > 0.40
        assert r["auto"] is True


def test_four_wheeler_with_seatbelt_compliant(tmp_path, monkeypatch):
    """Test 1: Four-wheeler image with seatbelt is compliant and never flagged as NO_HELMET."""
    monkeypatch.setattr(cs, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(cs.settings, "EVIDENCE_DIR", str(tmp_path / "evidence"))

    sb_path = Path("data/seatbelt/test_samples/ds1_seatbelttraining_1-MaleGlasses_mp4-1_jpg.rf.4ccea420d7d0e07b00d0dfbf1e02cb17_00005.jpg")
    if not sb_path.exists():
        return
    upload_path = cs.save_upload("car_seatbelt.jpg", sb_path.read_bytes())
    res = cs.process_upload(upload_path, {"violation_type": "NO_SEATBELT", "location": "Brigade Road"})

    assert "NO_HELMET" not in res["all_violations"], "Four-wheeler must NEVER receive NO_HELMET"
    assert "NO_SEATBELT" not in res["all_violations"], "Compliant seatbelt must NOT be flagged as violation"
    assert len(res["records"]) == 1
    assert not res["records"][0]["auto"]  # Forwarded for manual review as compliant


def test_four_wheeler_without_seatbelt_violation(tmp_path, monkeypatch):
    """Test 2: Four-wheeler image without seatbelt detects NO_SEATBELT violation."""
    monkeypatch.setattr(cs, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(cs.settings, "EVIDENCE_DIR", str(tmp_path / "evidence"))

    ns_path = Path("data/seatbelt/test_samples/ds2_yashinta_Screenshot-2025-12-25-141743_png.rf.3647621d5a237bbbf64dd0a40829542c_00150.jpg")
    if not ns_path.exists():
        return
    upload_path = cs.save_upload("car_no_seatbelt.jpg", ns_path.read_bytes())
    res = cs.process_upload(upload_path, {"violation_type": "AUTO_DETECT", "location": "Outer Ring Road"})

    assert "NO_SEATBELT" in res["all_violations"], "Must detect NO_SEATBELT"
    assert "NO_HELMET" not in res["all_violations"], "Must NOT detect NO_HELMET on four-wheeler"
    assert len(res["records"]) >= 1
    assert any(r["type"] == "NO_SEATBELT" and r["auto"] for r in res["records"])


def test_two_wheeler_without_helmet_violation(tmp_path, monkeypatch):
    """Test 3: Two-wheeler without helmet detects NO_HELMET and never NO_SEATBELT."""
    monkeypatch.setattr(cs, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(cs.settings, "EVIDENCE_DIR", str(tmp_path / "evidence"))

    video_path = Path("data/videos/22.mp4")
    if not video_path.exists():
        return
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
    ok, frame = cap.read()
    cap.release()
    assert ok

    ok, buf = cv2.imencode(".jpg", frame)
    upload_path = cs.save_upload("bike_no_helmet.jpg", buf.tobytes())
    res = cs.process_upload(upload_path, {"violation_type": "AUTO_DETECT", "location": "Hosur Road"})

    assert "NO_HELMET" in res["all_violations"], "Must detect NO_HELMET on two-wheeler"
    assert "NO_SEATBELT" not in res["all_violations"], "Two-wheeler must NEVER receive NO_SEATBELT"


def test_two_wheeler_with_helmet_compliance(tmp_path, monkeypatch):
    """Test 4: Two-wheeler rider with helmet does not generate false violation."""
    monkeypatch.setattr(cs, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(cs.settings, "EVIDENCE_DIR", str(tmp_path / "evidence"))

    # Test synthetic two-wheeler with green helmet box
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    upload_path = cs.save_upload("bike_helmet.jpg", buf.tobytes())
    res = cs.process_upload(upload_path, {"violation_type": "NO_HELMET", "location": "Koramangala"})

    assert "NO_HELMET" not in res["all_violations"]
    assert "NO_SEATBELT" not in res["all_violations"]


def test_unclear_vehicle_type_inconclusive(tmp_path, monkeypatch):
    """Test 5: Image with unclear vehicle returns inconclusive without inventing violations."""
    monkeypatch.setattr(cs, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(cs.settings, "EVIDENCE_DIR", str(tmp_path / "evidence"))

    img = np.full((300, 400, 3), 128, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    upload_path = cs.save_upload("unclear.jpg", buf.tobytes())
    res = cs.process_upload(upload_path, {"violation_type": "AUTO_DETECT", "location": "Silk Board"})

    assert len(res["all_violations"]) == 0
    assert not res["ai_detected"]
    assert len(res["records"]) == 1
    assert not res["records"][0]["auto"]


def test_unclear_seatbelt_visibility_low_sharpness(tmp_path, monkeypatch):
    """Test 6: Blurred four-wheeler image suppresses automated penalty due to low optical sharpness."""
    monkeypatch.setattr(cs, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(cs.settings, "EVIDENCE_DIR", str(tmp_path / "evidence"))

    sample_car = Path("data/citizen_uploads/CIT-20261104100520-2301/upload.png")
    if not sample_car.exists():
        return
    img = cv2.imread(str(sample_car))
    blurred = cv2.GaussianBlur(img, (45, 45), 0)
    ok, buf = cv2.imencode(".jpg", blurred)
    upload_path = cs.save_upload("blurred_car.jpg", buf.tobytes())
    res = cs.process_upload(upload_path, {"violation_type": "NO_SEATBELT", "location": "Airport Road"})

    assert len(res["all_violations"]) == 0, "Low sharpness must NOT invent a violation"
    assert not res["records"][0]["auto"]


def test_correct_violation_to_vehicle_routing(tmp_path, monkeypatch):
    """Test 7: Direct routing test ensuring four-wheelers never get helmet rules and motorcycles never get seatbelt."""
    user_car = Path("data/citizen_uploads/CIT-20261104100520-2301/upload.png")
    if user_car.exists():
        img = cv2.imread(str(user_car))
        classification = cs.classify_citizen_photo(img, claimed_viol="AUTO_DETECT")
        assert classification["is_four_wheeler"] is True
        assert "NO_HELMET" not in classification["unique_violations"]

    video_path = Path("data/videos/22.mp4")
    if video_path.exists():
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
        ok, frame = cap.read()
        cap.release()
        if ok:
            classification = cs.classify_citizen_photo(frame, claimed_viol="AUTO_DETECT")
            assert classification["is_two_wheeler"] is True
            assert "NO_SEATBELT" not in classification["unique_violations"]


def test_user_car_upload_regression_prevents_false_positive(tmp_path, monkeypatch):
    """Test 8 (Mandatory Regression): The user's exact test image (upload.png) has a fastened seatbelt.
    The system MUST NOT create a false NO_SEATBELT violation or NO_HELMET violation.
    It returns INCONCLUSIVE and forwards for manual officer review rather than inventing a penalty."""
    import json
    from dashboard.views.review import _why

    monkeypatch.setattr(cs, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(cs.settings, "EVIDENCE_DIR", str(tmp_path / "evidence"))

    car_img = Path("data/citizen_uploads/CIT-20261104100520-2301/upload.png")
    if not car_img.exists():
        return

    upload_path = cs.save_upload("test_user_car.png", car_img.read_bytes())
    res = cs.process_upload(upload_path, {
        "violation_type": "AUTO_DETECT",
        "plate": "KA01AB1234",
        "location": "Koramangala 100ft Rd",
        "name": "Pavan Test",
        "phone": "9876543210",
        "note": "Car driver verification",
    })

    assert "NO_HELMET" not in res["all_violations"], "Must NEVER classify car as NO_HELMET"
    assert "NO_SEATBELT" not in res["all_violations"], "Must NOT create a false NO_SEATBELT violation on belted driver"
    assert res["compliance_status"] == "INCONCLUSIVE", "Must be INCONCLUSIVE when seatbelt cannot be reliably confirmed"
    assert len(res["records"]) == 1
    rec = res["records"][0]
    assert rec["auto"] is False, "Must NOT be auto-classified as violation"
    assert rec["type"] == "CITIZEN_REPORT", "Must default to safe CITIZEN_REPORT for manual officer review"

    # Check database persistence
    db = SessionLocal()
    try:
        v = db.query(Violation).filter(Violation.id == rec["id"]).first()
        assert v is not None
        assert v.violation_type == "CITIZEN_REPORT"
        assert v.status == "PENDING_REVIEW"
        assert v.source == "CITIZEN"

        ev = db.query(Evidence).filter(Evidence.violation_id == v.id).first()
        assert ev is not None
        assert ev.image_path and Path(ev.image_path).exists()
        assert ev.metadata_json and Path(ev.metadata_json).exists()

        with open(ev.metadata_json, "r") as mf:
            meta = json.load(mf)
        assert meta["actual_type"] == "CITIZEN_REPORT"
        assert meta["extra"]["status_category"] == "INCONCLUSIVE"
        assert "officer review" in meta["extra"]["detection_note"].lower()
    finally:
        db.close()
