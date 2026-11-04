import os
import cv2
import pytest
import numpy as np
from pathlib import Path

from backend.services import citizen_service as cs
from pipeline.traffic_pipeline import TrafficVisionPipeline
from algorithms.violation_rules.triple_riding_rule import TripleRidingRule
from algorithms.association.rider_association import MotorcycleRiderGroup
from algorithms.tracking.base_tracker import TrackedObject


def test_scenario_1_motorcycle_with_helmet_no_violation(tmp_path, monkeypatch):
    """Scenario 1: Motorcycle with helmet generates NO violation."""
    monkeypatch.setattr(cs, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(cs.settings, "EVIDENCE_DIR", str(tmp_path / "evidence"))

    # Test with synthetic clear motorcycle + helmet
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    upload_path = cs.save_upload("bike_helmet.jpg", buf.tobytes())

    res = cs.process_upload(upload_path, {"violation_type": "NO_HELMET", "location": "MG Road"})
    assert "NO_HELMET" not in res["all_violations"], "Compliant helmet must not trigger violation"
    assert "NO_SEATBELT" not in res["all_violations"], "Two-wheeler must not trigger seatbelt"


def test_scenario_2_motorcycle_without_helmet_detected(tmp_path, monkeypatch):
    """Scenario 2: Motorcycle without helmet detects NO_HELMET violation."""
    monkeypatch.setattr(cs, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(cs.settings, "EVIDENCE_DIR", str(tmp_path / "evidence"))

    video_path = Path("data/videos/22.mp4")
    assert video_path.exists(), "Surveillance video 22.mp4 must exist"
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
    ok, frame = cap.read()
    cap.release()
    assert ok, "Frame 50 must be readable"

    ok, buf = cv2.imencode(".jpg", frame)
    upload_path = cs.save_upload("bike_no_helmet.jpg", buf.tobytes())
    res = cs.process_upload(upload_path, {"violation_type": "AUTO_DETECT", "location": "Hosur Road"})

    assert "NO_HELMET" in res["all_violations"], "Must detect NO_HELMET on two-wheeler"
    assert "NO_SEATBELT" not in res["all_violations"], "Must strictly suppress seatbelt on two-wheeler"


def test_scenario_3_motorcycle_with_two_riders_no_triple_riding():
    """Scenario 3: Motorcycle with 2 riders does NOT trigger TRIPLE_RIDING violation."""
    rule = TripleRidingRule(temporal_window=5, min_positive_frames=3, cooldown_seconds=5.0)

    bike = TrackedObject(track_id=101, class_name="motorcycle", class_id=3, bbox=[100, 100, 200, 200], confidence=0.9)
    group = MotorcycleRiderGroup(bike)

    r1 = TrackedObject(track_id=201, class_name="person", class_id=0, bbox=[120, 80, 160, 160], confidence=0.85)
    r2 = TrackedObject(track_id=202, class_name="person", class_id=0, bbox=[140, 90, 180, 170], confidence=0.85)

    group.add_rider(r1)
    group.add_rider(r2)

    context = {
        "motorcycle_groups": [group],
        "frame_idx": 1,
        "fps": 25.0,
    }

    # Evaluate across 5 consecutive frames
    for f in range(1, 6):
        context["frame_idx"] = f
        viols = rule.evaluate(context)
        assert len(viols) == 0, f"Frame {f}: Two riders must NEVER trigger TRIPLE_RIDING"


def test_scenario_4_motorcycle_with_three_riders_triple_riding_detected():
    """Scenario 4: Motorcycle with 3 riders triggers TRIPLE_RIDING violation."""
    rule = TripleRidingRule(temporal_window=5, min_positive_frames=3, cooldown_seconds=5.0)

    bike = TrackedObject(track_id=101, class_name="motorcycle", class_id=3, bbox=[100, 100, 200, 200], confidence=0.9)
    group = MotorcycleRiderGroup(bike)

    r1 = TrackedObject(track_id=201, class_name="person", class_id=0, bbox=[120, 80, 150, 160], confidence=0.85)
    r2 = TrackedObject(track_id=202, class_name="person", class_id=0, bbox=[140, 85, 170, 165], confidence=0.85)
    r3 = TrackedObject(track_id=203, class_name="person", class_id=0, bbox=[160, 90, 190, 170], confidence=0.85)

    group.add_rider(r1)
    group.add_rider(r2)
    group.add_rider(r3)

    context = {
        "motorcycle_groups": [group],
        "frame_idx": 1,
        "fps": 25.0,
    }

    # Temporal confirmation: 3 positive frames
    v1 = rule.evaluate(context)
    assert len(v1) == 0
    context["frame_idx"] = 2
    v2 = rule.evaluate(context)
    assert len(v2) == 0
    context["frame_idx"] = 3
    v3 = rule.evaluate(context)
    assert len(v3) == 1
    assert v3[0]["violation_type"] == "TRIPLE_RIDING"
    assert v3[0]["track_id"] == 101


def test_scenario_5_car_with_seatbelt_no_violation(tmp_path, monkeypatch):
    """Scenario 5: Car with fastened seatbelt does NOT trigger NO_SEATBELT or NO_HELMET violation."""
    monkeypatch.setattr(cs, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(cs.settings, "EVIDENCE_DIR", str(tmp_path / "evidence"))

    sb_path = Path("data/seatbelt/test_samples/ds1_seatbelttraining_1-MaleGlasses_mp4-1_jpg.rf.4ccea420d7d0e07b00d0dfbf1e02cb17_00005.jpg")
    if not sb_path.exists():
        pytest.skip("Seatbelt test sample not found")

    upload_path = cs.save_upload("car_seatbelt.jpg", sb_path.read_bytes())
    res = cs.process_upload(upload_path, {"violation_type": "NO_SEATBELT", "location": "Brigade Road"})

    assert "NO_HELMET" not in res["all_violations"], "Four-wheeler must never get NO_HELMET"
    assert "NO_SEATBELT" not in res["all_violations"], "Fastened seatbelt must not be penalized"


def test_scenario_6_car_without_seatbelt_violation_detected(tmp_path, monkeypatch):
    """Scenario 6: Car without seatbelt detects genuine NO_SEATBELT violation."""
    monkeypatch.setattr(cs, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(cs.settings, "EVIDENCE_DIR", str(tmp_path / "evidence"))

    ns_path = Path("data/seatbelt/test_samples/ds2_yashinta_Screenshot-2025-12-25-141743_png.rf.3647621d5a237bbbf64dd0a40829542c_00150.jpg")
    if not ns_path.exists():
        pytest.skip("No-seatbelt sample not found")

    upload_path = cs.save_upload("car_no_seatbelt.jpg", ns_path.read_bytes())
    res = cs.process_upload(upload_path, {"violation_type": "AUTO_DETECT", "location": "Outer Ring Road"})

    assert "NO_SEATBELT" in res["all_violations"], "Genuine unbelted driver must trigger NO_SEATBELT"
    assert "NO_HELMET" not in res["all_violations"], "Must never flag car as NO_HELMET"


def test_scenario_7_unclear_blurred_image_inconclusive(tmp_path, monkeypatch):
    """Scenario 7: Unclear or heavily blurred image returns INCONCLUSIVE without inventing false violations."""
    monkeypatch.setattr(cs, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(cs.settings, "EVIDENCE_DIR", str(tmp_path / "evidence"))

    # Create heavily blurred image
    gray = np.full((300, 400, 3), 128, dtype=np.uint8)
    blurred = cv2.GaussianBlur(gray, (51, 51), 0)
    ok, buf = cv2.imencode(".jpg", blurred)
    upload_path = cs.save_upload("blurred.jpg", buf.tobytes())

    res = cs.process_upload(upload_path, {"violation_type": "AUTO_DETECT", "location": "Silk Board Junction"})
    assert len(res["all_violations"]) == 0, "Unclear image must not invent violations"
    assert not res["ai_detected"]
    assert len(res["records"]) == 1
    assert res["records"][0]["auto"] is False, "Must require manual officer review"

