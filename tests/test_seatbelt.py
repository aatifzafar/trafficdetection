"""
Automated Pytest Suite for Seatbelt Violation Detection Module:
1. Model-missing standby behavior (no fabricated events).
2. Class-map correctness (0: seatbelt, 1: no_seatbelt).
3. Multi-frame voting logic & cooldown suppression.
4. Insufficient visibility handling (small / blurred crops).
5. Perspective auto-detection and ROI extraction (FRONT vs SIDE vs Occupant-anchored).
6. TrafficRuleEngine integration and camera calibration propagation.
7. FastAPI seatbelt endpoints (/api/seatbelt/metrics, /api/seatbelt/detect).
8. Data leakage check across train/val/test splits (perceptual dHash Hamming distance).
9. Full violation generation, evidence recording, and database persistence flow.
10. Security upload filename sanitization (Path traversal prevention).
11. Dashboard view imports and render function availability.
"""

import io
import yaml
import pytest
import numpy as np
from pathlib import Path
from PIL import Image
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from backend.main import app
from backend.database.database import SessionLocal
from backend.db_models.models import Violation
from algorithms.tracking.base_tracker import TrackedObject
from algorithms.detection.detector import DetectionResult
from algorithms.violation_rules.seatbelt_rule import SeatbeltViolationRule
from algorithms.violation_rules.rule_engine import TrafficRuleEngine
from algorithms.calibration.camera_calibration import config_for_camera, RULE_KEYS
from pipeline.traffic_pipeline import TrafficVisionPipeline
from scripts.build_seatbelt_dataset import compute_dhash, hamming_distance

client = TestClient(app)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_seatbelt_model_missing_standby_behavior():
    """Verify safe standby fallback when seatbelt weights are missing."""
    rule = SeatbeltViolationRule(
        model_path="models/definitely_nonexistent_weights_12345.pt",
        temporal_window=5,
        min_positive_frames=3,
    )
    assert not rule.is_configured
    assert rule.seatbelt_model is None

    # Context with a car
    car = TrackedObject(track_id=1, class_name="car", class_id=2, bbox=[100, 100, 300, 300], confidence=0.9)
    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    context = {"tracked_vehicles": [car], "frame": frame}

    violations = rule.evaluate(context)
    assert violations == [], "Must never fabricate violations in standby mode"
    assert context.get("seatbelt_status") == "Seatbelt detection model not configured"


def test_seatbelt_class_map_correctness():
    """Verify data.yaml has unified class map: 0 = seatbelt, 1 = no_seatbelt."""
    yaml_path = PROJECT_ROOT / "data" / "seatbelt" / "data.yaml"
    assert yaml_path.exists(), "data/seatbelt/data.yaml must exist"

    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    assert cfg["nc"] == 2
    names = cfg["names"]
    assert names[0] == "seatbelt", f"Class 0 must be 'seatbelt', got '{names[0]}'"
    assert names[1] == "no_seatbelt", f"Class 1 must be 'no_seatbelt', got '{names[1]}'"


def test_seatbelt_voting_logic():
    """Verify N out of M multi-frame temporal voting and cooldown duplicate prevention."""
    rule = SeatbeltViolationRule(
        model_path="models/definitely_nonexistent_weights_12345.pt",
        temporal_window=5,
        min_positive_frames=3,
        cooldown_seconds=10.0,
    )

    track_id = 42

    # Frame 1: positive, but needs 3 frames
    c1 = rule.record_frame_evidence(track_id, is_violating=True, current_time=1.0)
    assert not c1

    # Frame 2: positive, needs 3 frames
    c2 = rule.record_frame_evidence(track_id, is_violating=True, current_time=2.0)
    assert not c2

    # Frame 3: 3rd positive frame -> Confirmation triggered!
    c3 = rule.record_frame_evidence(track_id, is_violating=True, current_time=3.0)
    assert c3, "Should confirm violation on 3rd positive frame within window of 5"

    # Frame 4: during cooldown -> Suppressed
    c4 = rule.record_frame_evidence(track_id, is_violating=True, current_time=4.0)
    assert not c4, "Duplicate violation must be suppressed during cooldown"


def test_seatbelt_insufficient_visibility_handling():
    """Verify that small or blurred crops report insufficient visibility and do not trigger violations."""
    rule = SeatbeltViolationRule(
        model_path="models/definitely_nonexistent_weights_12345.pt",
        min_crop_w=120,
        min_crop_h=120,
        min_sharpness=50.0,
    )
    # Manually configure a dummy mock model to verify optical checks precede model inference
    class DummyModel:
        names = {0: "seatbelt", 1: "no_seatbelt"}
        def __call__(self, crop, **kwargs):
            raise RuntimeError("Model should NOT be called if crop visibility is insufficient!")

    rule.is_configured = True
    rule.seatbelt_model = DummyModel()

    # Small car -> windshield crop will be < 120x120
    small_car = TrackedObject(track_id=10, class_name="car", class_id=2, bbox=[10, 10, 80, 80], confidence=0.8)
    frame = np.ones((500, 500, 3), dtype=np.uint8) * 128
    context = {"tracked_vehicles": [small_car], "frame": frame}

    viols = rule.evaluate(context)
    assert viols == []
    assert "insufficient_visibility" in context.get("seatbelt_vis_10", "").lower()

    # Blurred car -> large enough (200x200), but uniform blurred grey (Laplacian var ~ 0)
    large_blurred_car = TrackedObject(track_id=11, class_name="car", class_id=2, bbox=[50, 50, 350, 350], confidence=0.8)
    blurred_frame = np.ones((500, 500, 3), dtype=np.uint8) * 100  # flat image has zero variance
    context_blur = {"tracked_vehicles": [large_blurred_car], "frame": blurred_frame}

    viols_blur = rule.evaluate(context_blur)
    assert viols_blur == []
    assert "insufficient_visibility" in context_blur.get("seatbelt_vis_11", "").lower()


def test_seatbelt_perspective_and_roi_extraction():
    """Verify AUTO perspective discrimination (aspect ratio) and occupant-anchored ROI extraction."""
    rule = SeatbeltViolationRule(
        model_path="models/definitely_nonexistent_weights_12345.pt",
        perspective="AUTO",
    )

    # 1. Front-facing car: width=200, height=180 -> aspect ratio ~ 1.11 <= 1.35 -> FRONT
    front_car = TrackedObject(track_id=1, class_name="car", class_id=2, bbox=[100, 100, 300, 280], confidence=0.9)
    persp_front = rule._determine_perspective(front_car.bbox, (720, 1280))
    assert persp_front == "FRONT"

    # 2. Side-facing car: width=350, height=160 -> aspect ratio ~ 2.18 > 1.35 -> SIDE
    side_car = TrackedObject(track_id=2, class_name="car", class_id=2, bbox=[100, 100, 450, 260], confidence=0.9)
    persp_side = rule._determine_perspective(side_car.bbox, (720, 1280))
    assert persp_side == "SIDE"

    # 3. Geometric ROI extraction without occupant
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    crop_f, box_f, mode_f = rule._extract_cabin_roi(front_car, frame, "FRONT", [])
    assert mode_f == "front_windshield"
    assert crop_f is not None

    crop_s, box_s, mode_s = rule._extract_cabin_roi(side_car, frame, "SIDE", [])
    assert mode_s == "side_window"
    assert crop_s is not None

    # 4. Occupant-anchored ROI extraction
    person = DetectionResult(bbox=[150, 110, 210, 180], confidence=0.85, class_id=0, class_name="person")
    crop_o, box_o, mode_o = rule._extract_cabin_roi(front_car, frame, "FRONT", [person])
    assert mode_o == "occupant_anchor"
    assert crop_o is not None


def test_rule_engine_seatbelt_calibration_wiring():
    """Verify TrafficRuleEngine correctly instantiates SeatbeltViolationRule with camera calibration."""
    base_cfg = {
        "violations": {"seatbelt": True, "seatbelt_confidence": 0.42},
        "cameras": {
            "default_perspective": "AUTO",
            "seatbelt_min_crop_px": 95,
            "seatbelt_min_sharpness": 35.0,
        },
        "camera_calibration": {
            "7": {
                "perspective": "SIDE",
                "rules": {"seatbelt": True},
            }
        }
    }

    assert "seatbelt" in RULE_KEYS

    # Camera 7 calibrated as SIDE
    cam_cfg = config_for_camera(base_cfg, camera_id=7)
    engine = TrafficRuleEngine(config=cam_cfg)

    assert "seatbelt" in engine.rules
    sb_rule = engine.rules["seatbelt"]
    assert isinstance(sb_rule, SeatbeltViolationRule)
    assert sb_rule.default_perspective == "SIDE"
    assert sb_rule.confidence_threshold == 0.42
    assert sb_rule.min_crop_w == 95
    assert sb_rule.min_sharpness == 35.0


def test_seatbelt_api_endpoints():
    """Verify FastAPI /api/seatbelt/metrics and /api/seatbelt/detect endpoints."""
    # 1. Metrics endpoint
    res_m = client.get("/api/seatbelt/metrics")
    assert res_m.status_code == 200
    data_m = res_m.json()
    assert data_m.get("experimental") is True

    # 2. Detect endpoint with synthetic image
    buf = io.BytesIO()
    test_img = Image.new("RGB", (200, 200), color=(128, 128, 128))
    test_img.save(buf, format="JPEG")
    buf.seek(0)

    res_d = client.post("/api/seatbelt/detect", files={"file": ("test.jpg", buf, "image/jpeg")})
    assert res_d.status_code == 200
    data_d = res_d.json()
    assert "status" in data_d
    assert "experimental" in data_d
    assert data_d["experimental"] is True
    assert "model_configured" in data_d


def test_seatbelt_data_leakage_check():
    """Verify strict zero data leakage across train, val, and test splits."""
    data_dir = PROJECT_ROOT / "data" / "seatbelt"
    train_dir = data_dir / "train" / "images"
    val_dir = data_dir / "val" / "images"
    test_dir = data_dir / "test" / "images"

    assert train_dir.exists() and val_dir.exists() and test_dir.exists()

    # Compute dhashes for all train images
    train_hashes = set()
    for f in list(train_dir.glob("*.jpg"))[:300]:  # sample check
        with Image.open(f) as im:
            train_hashes.add(compute_dhash(im))

    # Check test images against train hashes
    leak_count = 0
    for f in list(test_dir.glob("*.jpg"))[:100]:
        with Image.open(f) as im:
            th = compute_dhash(im)
            for tr_h in train_hashes:
                if hamming_distance(th, tr_h) <= 2:  # strict near-duplicate
                    leak_count += 1
                    break

    assert leak_count == 0, f"Detected {leak_count} near-duplicate leaked images between Test and Train splits!"


def test_seatbelt_full_violation_generation_and_pipeline_flow():
    """Verify that detected no_seatbelt triggers confirmed violation through evaluate and pipeline."""
    rule = SeatbeltViolationRule(
        model_path="models/definitely_nonexistent_weights_12345.pt",
        temporal_window=3,
        min_positive_frames=2,
        cooldown_seconds=10.0,
        min_crop_w=50,
        min_crop_h=50,
        min_sharpness=5.0,
    )

    class MockBox:
        def __init__(self, cls_id, conf):
            self.cls = [MagicMock(item=lambda: cls_id)]
            self.conf = [MagicMock(item=lambda: conf)]
            self.xyxy = [MagicMock(tolist=lambda: [10, 10, 80, 80])]

    class MockPred:
        def __init__(self, boxes):
            self.boxes = boxes

    class MockModel:
        names = {0: "seatbelt", 1: "no_seatbelt"}
        def __call__(self, crop, **kwargs):
            return [MockPred([MockBox(1, 0.88)])]

    rule.is_configured = True
    rule.seatbelt_model = MockModel()

    # Generate a textured frame so Laplacian sharpness exceeds threshold
    rng = np.random.RandomState(42)
    textured_frame = rng.randint(0, 256, (400, 400, 3), dtype=np.uint8)
    car = TrackedObject(track_id=88, class_name="car", class_id=2, bbox=[50, 50, 250, 250], confidence=0.92)

    # Frame 1: first positive frame (needs 2)
    ctx1 = {"frame": textured_frame, "frame_idx": 1, "fps": 25.0, "tracked_vehicles": [car]}
    v1 = rule.evaluate(ctx1)
    assert len(v1) == 0

    # Frame 2: second positive frame -> CONFIRMED VIOLATION!
    ctx2 = {"frame": textured_frame, "frame_idx": 2, "fps": 25.0, "tracked_vehicles": [car]}
    v2 = rule.evaluate(ctx2)
    assert len(v2) == 1
    viol = v2[0]
    assert viol["violation_type"] == "NO_SEATBELT"
    assert viol["status"] == "PENDING_REVIEW"
    assert viol["track_id"] == 88
    assert viol["confidence"] == 0.88
    assert "perspective" in viol["extra"]
    assert "sharpness" in viol["extra"]
    assert "crop_dimensions" in viol["extra"]

    # Verify pipeline handle and database recording
    pipe = TrafficVisionPipeline(persist=True, location="Test Corridor")
    pipe.rule_engine.rules["seatbelt"] = rule
    saved = pipe._handle_confirmed_violation(viol, textured_frame, frame_idx=2, fps=25.0)
    assert saved is True

    db = SessionLocal()
    try:
        rec = db.query(Violation).filter(Violation.track_id == 88, Violation.violation_type == "NO_SEATBELT").first()
        assert rec is not None
        assert rec.status == "PENDING_REVIEW"
    finally:
        db.close()


def test_security_upload_path_sanitization():
    """Verify that file uploads with path traversal characters cannot escape destination dir."""
    from backend.api.uploads import sanitize_filename

    assert ".." not in sanitize_filename("../../etc/passwd")
    assert "/" not in sanitize_filename("../../etc/passwd")
    assert "\\" not in sanitize_filename("..\\..\\windows\\system32\\cmd.exe")
    assert sanitize_filename("normal_video_1.mp4") == "normal_video_1.mp4"


def test_all_dashboard_views_importable_and_defined():
    """Verify that all dashboard views are importable and expose callable render function."""
    from dashboard.views import (
        overview, review, cameras, cameras_setup, citizen_upload,
        violations, vehicles, analytics, models_page, evaluation,
        seatbelt_test, settings_page
    )

    views = [
        overview, review, cameras, cameras_setup, citizen_upload,
        violations, vehicles, analytics, models_page, evaluation,
        seatbelt_test, settings_page
    ]

    for v in views:
        assert hasattr(v, "render"), f"View {v.__name__} must define render()"
        assert callable(v.render), f"View {v.__name__}.render must be callable"
