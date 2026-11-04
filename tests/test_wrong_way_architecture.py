"""
tests/test_wrong_way_architecture.py
Verification suite for Intelligent Transportation Systems (ITS) Wrong-Way Driving Architecture.
Tests cover:
  Test 1: Vehicle moving in permitted direction -> NO VIOLATION.
  Test 2: Vehicle moving against permitted direction -> WRONG WAY CONFIRMED.
  Test 3: Stationary vehicle -> NO VIOLATION.
  Test 4: Vehicle making a normal turn -> NO WRONG-WAY VIOLATION.
  Test 5: Static photograph of a vehicle -> VEHICLE DETECTED, DIRECTION NOT VERIFIED.
  Test 6: Noisy or incomplete tracking (< 8 observations) -> PENDING VERIFICATION.
  Test 7: Multiple vehicles travelling in different directions -> Independent evaluation.
  Test 8: Vehicle briefly moving against direction due to tracking noise -> NO CONFIRMED VIOLATION.
"""

import time
import pytest
import numpy as np
from pathlib import Path
from algorithms.tracking.base_tracker import TrackedObject
from algorithms.violation_rules.wrong_way_rule import WrongWayViolationRule


def create_vehicle(track_id: int, trajectory_points: list, class_name: str = "car") -> TrackedObject:
    """Helper to instantiate TrackedObject with a populated centroid trajectory."""
    first_pt = trajectory_points[0]
    last_pt = trajectory_points[-1]
    w, h = 60.0, 80.0
    v = TrackedObject(
        track_id=track_id,
        class_name=class_name,
        class_id=2,
        bbox=[last_pt[0] - w/2, last_pt[1] - h/2, last_pt[0] + w/2, last_pt[1] + h/2],
        confidence=0.90,
    )
    v.trajectory = [list(pt) for pt in trajectory_points]
    return v


# ==============================================================================
# TEST 1: Vehicle moving in permitted direction
# Expected: NO VIOLATION
# ==============================================================================
def test_1_vehicle_moving_in_permitted_direction():
    rule = WrongWayViolationRule(
        lane_vector=[0.0, 1.0],  # Downward permitted flow
        max_angle_deviation_deg=120.0,
        min_displacement_px=15.0,
        min_trajectory_obs=8,
        temporal_window=10,
        min_positive_frames=6,
    )

    # Vehicle travelling downward from Y=100 to Y=280 (dy = +180px, theta = 0 deg)
    points = [[200.0, 100.0 + i * 20.0] for i in range(10)]
    v_compliant = create_vehicle(track_id=1, trajectory_points=points)

    context = {"tracked_vehicles": [v_compliant], "fps": 25.0, "frame_idx": 10}
    violations = rule.evaluate(context)

    assert len(violations) == 0, "Compliant vehicle moving with lane flow must not be flagged"


# ==============================================================================
# TEST 2: Vehicle moving against permitted direction
# Expected: WRONG WAY CONFIRMED
# ==============================================================================
def test_2_vehicle_moving_against_permitted_direction():
    rule = WrongWayViolationRule(
        lane_vector=[0.0, 1.0],  # Downward flow
        max_angle_deviation_deg=120.0,
        min_displacement_px=15.0,
        min_trajectory_obs=8,
        temporal_window=10,
        min_positive_frames=6,
        cooldown_seconds=15.0,
    )

    # Vehicle driving backwards (upwards: dy = -180px, theta = 180 deg)
    violations = []
    for frame in range(10):
        # Rolling trajectory moving upwards
        points = [[200.0, 400.0 - (frame + i) * 15.0] for i in range(9)]
        v_offender = create_vehicle(track_id=2, trajectory_points=points)
        context = {"tracked_vehicles": [v_offender], "fps": 25.0, "frame_idx": frame}
        viols = rule.evaluate(context)
        if viols:
            violations.extend(viols)

    assert len(violations) >= 1, "Vehicle driving against lane direction must be confirmed"
    top_viol = violations[0]
    assert top_viol["violation_type"] == "WRONG_WAY"
    assert top_viol["track_id"] == 2
    assert top_viol["confidence"] >= 0.75
    assert top_viol["extra"]["motion_verified"] is True
    assert top_viol["extra"]["angle_deviation_deg"] >= 120.0


# ==============================================================================
# TEST 3: Stationary vehicle
# Expected: NO VIOLATION
# ==============================================================================
def test_3_stationary_vehicle():
    rule = WrongWayViolationRule(
        lane_vector=[0.0, 1.0],
        max_angle_deviation_deg=120.0,
        min_displacement_px=15.0,
        min_trajectory_obs=8,
        temporal_window=10,
        min_positive_frames=6,
    )

    # Vehicle stationary at [200, 200] with small sensor jitter (< 3 pixels total)
    jitter_points = [
        [200.0, 200.0],
        [200.3, 200.1],
        [199.8, 200.4],
        [200.1, 199.9],
        [200.4, 200.2],
        [199.9, 200.0],
        [200.2, 200.3],
        [200.0, 200.1],
        [200.1, 200.2],
    ]
    v_parked = create_vehicle(track_id=3, trajectory_points=jitter_points)

    context = {"tracked_vehicles": [v_parked], "fps": 25.0, "frame_idx": 15}
    violations = rule.evaluate(context)

    assert len(violations) == 0, "Stationary vehicle under 15px displacement threshold must not trigger violation"


# ==============================================================================
# TEST 4: Vehicle making a normal turn
# Expected: NO WRONG-WAY VIOLATION
# ==============================================================================
def test_4_vehicle_making_normal_turn():
    rule = WrongWayViolationRule(
        lane_vector=[0.0, 1.0],  # Permitted flow is downward
        max_angle_deviation_deg=120.0,
        min_displacement_px=15.0,
        min_trajectory_obs=8,
        temporal_window=10,
        min_positive_frames=6,
    )

    # Vehicle moves downward and makes a right turn (dx > 0, dy > 0: angle ~45 deg)
    turn_points = [
        [100.0, 100.0],
        [105.0, 120.0],
        [112.0, 140.0],
        [122.0, 160.0],
        [135.0, 180.0],
        [150.0, 200.0],
        [170.0, 220.0],
        [195.0, 240.0],
        [225.0, 260.0],
    ]
    v_turning = create_vehicle(track_id=4, trajectory_points=turn_points)

    context = {"tracked_vehicles": [v_turning], "fps": 25.0, "frame_idx": 20}
    violations = rule.evaluate(context)

    assert len(violations) == 0, "Normal intersection turn (angle < 90 deg) must not be flagged as wrong way"


# ==============================================================================
# TEST 5: Static photograph of a vehicle
# Expected: VEHICLE DETECTED, DIRECTION NOT VERIFIED (No fake 95% confidence)
# ==============================================================================
def test_5_static_photograph_direction_not_verified(tmp_path):
    import cv2
    from backend.services import citizen_service as cs

    # Create synthetic test image of a vehicle on road
    dummy_img_path = tmp_path / "traffic_photo.jpg"
    blank_frame = np.full((480, 640, 3), 120, dtype=np.uint8)
    # Draw simple car shape
    cv2.rectangle(blank_frame, (200, 200), (350, 350), (40, 40, 200), -1)
    cv2.imwrite(str(dummy_img_path), blank_frame)

    info = {
        "violation_type": "WRONG_WAY",
        "plate": "KA01AB1234",
        "location": "MG Road",
        "note": "Car appeared opposite in photo",
        "original_name": "traffic_photo.jpg",
    }

    result = cs.process_upload(dummy_img_path, info)
    records = result.get("records", [])

    assert len(records) >= 1, "Report should be logged for officer review"
    rec = records[0]
    assert rec["type"] == "WRONG_WAY"
    # Ground truth integrity: Single static photo MUST NOT claim AI verified motion
    assert rec["confidence"] == 0.0, "Static photo must have 0.0 AI confidence for motion"
    assert rec["auto"] is False, "Static photo must not be marked as auto AI-verified"
    assert rec["verification_status"] == "NOT_VERIFIED", "Status must explicitly reflect NOT_VERIFIED"


# ==============================================================================
# TEST 6: Noisy or incomplete tracking (< 8 observations)
# Expected: PENDING VERIFICATION (NO premature violation)
# ==============================================================================
def test_6_noisy_or_incomplete_tracking():
    rule = WrongWayViolationRule(
        lane_vector=[0.0, 1.0],
        max_angle_deviation_deg=120.0,
        min_displacement_px=15.0,
        min_trajectory_obs=8,  # Minimum 8 observations required
        temporal_window=10,
        min_positive_frames=6,
    )

    # Incomplete track: only 4 frames observed so far
    short_points = [
        [200.0, 300.0],
        [200.0, 280.0],
        [200.0, 260.0],
        [200.0, 240.0],
    ]
    v_short = create_vehicle(track_id=6, trajectory_points=short_points)

    context = {"tracked_vehicles": [v_short], "fps": 25.0, "frame_idx": 5}
    violations = rule.evaluate(context)

    assert len(violations) == 0, "Incomplete track (< 8 points) must be pending, not flagged"


# ==============================================================================
# TEST 7: Multiple vehicles travelling in different directions
# Expected: Independent track evaluation
# ==============================================================================
def test_7_multiple_vehicles_evaluated_independently():
    rule = WrongWayViolationRule(
        lane_vector=[0.0, 1.0],
        max_angle_deviation_deg=120.0,
        min_displacement_px=15.0,
        min_trajectory_obs=8,
        temporal_window=10,
        min_positive_frames=6,
        cooldown_seconds=15.0,
    )

    # Vehicle A: Legal flow (downward)
    pts_legal = [[100.0, 100.0 + i * 20.0] for i in range(10)]
    # Vehicle B: Wrong-way flow (upward)
    pts_illegal = [[300.0, 400.0 - i * 20.0] for i in range(10)]
    # Vehicle C: Stationary
    pts_parked = [[500.0, 200.0] for _ in range(10)]

    vA = create_vehicle(track_id=10, trajectory_points=pts_legal)
    vB = create_vehicle(track_id=20, trajectory_points=pts_illegal)
    vC = create_vehicle(track_id=30, trajectory_points=pts_parked)

    # Evaluate across 6 consecutive frames
    all_viols = []
    for frame in range(6):
        context = {"tracked_vehicles": [vA, vB, vC], "fps": 25.0, "frame_idx": frame}
        viols = rule.evaluate(context)
        if viols:
            all_viols.extend(viols)

    flagged_ids = [v["track_id"] for v in all_viols]
    assert 20 in flagged_ids, "Illegal vehicle (track 20) must be confirmed"
    assert 10 not in flagged_ids, "Legal vehicle (track 10) must not be flagged"
    assert 30 not in flagged_ids, "Parked vehicle (track 30) must not be flagged"


# ==============================================================================
# TEST 8: Vehicle briefly moving against direction due to tracking noise
# Expected: NO CONFIRMED VIOLATION (Temporal confirmation window filtering)
# ==============================================================================
def test_8_brief_tracking_noise_suppressed():
    rule = WrongWayViolationRule(
        lane_vector=[0.0, 1.0],
        max_angle_deviation_deg=120.0,
        min_displacement_px=15.0,
        min_trajectory_obs=8,
        temporal_window=10,
        min_positive_frames=6,  # Requires at least 6 confirming frames in window
    )

    # Vehicle is driving legally forward for 8 frames, experiences 2 frames of tracking glitch, then resumes
    all_viols = []
    for frame in range(10):
        if frame in (4, 5):
            # Glitch frame: anomalous reverse jump
            pts = [[200.0, 300.0 - i * 15.0] for i in range(9)]
        else:
            # Normal forward driving
            pts = [[200.0, 100.0 + (frame + i) * 15.0] for i in range(9)]

        v_noisy = create_vehicle(track_id=8, trajectory_points=pts)
        context = {"tracked_vehicles": [v_noisy], "fps": 25.0, "frame_idx": frame}
        viols = rule.evaluate(context)
        if viols:
            all_viols.extend(viols)

    assert len(all_viols) == 0, "Tracking noise lasting only 2 frames must be filtered out by temporal confirmation window (>=6/10 required)"


# ==============================================================================
# TEST 9: Photo carriageway orientation consensus analysis
# Expected: Opposing vehicle flagged as outlier candidate; compliant vehicles marked normal
# ==============================================================================
def test_9_photo_orientation_consensus_analyzer():
    from algorithms.violation_rules.photo_orientation_analyzer import PhotoOrientationAnalyzer
    from algorithms.detection.detector import DetectionResult

    analyzer = PhotoOrientationAnalyzer(min_consensus_ratio=0.65)

    # Synthetic test image
    img = np.zeros((480, 640, 3), dtype=np.uint8)

    # Create synthetic detections: 3 oncoming vehicles (bright front) and 1 receding (red taillights)
    # Draw simulated front features (bright white headlights in lower section) on oncoming vehicles
    for x, y in [(50, 200), (200, 200), (350, 200)]:
        img[y+30:y+70, x+10:x+80] = 230  # bright headlight/grille reflection

    # Draw simulated rear features (horizontal bumper and trunk lid seams across rear fascia)
    rx, ry = 500, 200
    img[ry+45:ry+50, rx+10:rx+80] = 180
    img[ry+60:ry+65, rx+5:rx+85] = 220
    img[ry+40:ry+70, rx+5:rx+25, 2] = 240
    img[ry+40:ry+70, rx+65:rx+85, 2] = 240

    detections = [
        DetectionResult(bbox=[50, 200, 140, 280], confidence=0.88, class_name="car", class_id=2),
        DetectionResult(bbox=[200, 200, 290, 280], confidence=0.85, class_name="car", class_id=2),
        DetectionResult(bbox=[350, 200, 440, 280], confidence=0.82, class_name="car", class_id=2),
        DetectionResult(bbox=[500, 200, 590, 280], confidence=0.79, class_name="car", class_id=2),
    ]

    res = analyzer.analyze_photo(img, detections)
    assert res["dominant_orientation"] == "FACING_CAMERA"
    assert res["has_outlier"] is True
    assert len(res["outlier_candidates"]) == 1
    assert res["outlier_candidates"][0]["index"] == 3
    assert res["outlier_candidates"][0]["orientation"] == "FACING_AWAY"

    # Verify annotation rendering
    ann = analyzer.render_annotated_photo(img, res)
    assert ann is not None
    assert ann.shape == img.shape


# ==============================================================================
# TEST 10: Real CCTV photo proof verification with citizen service
# Expected: Red car detected as unverified wrong-way candidate with 77% confidence
# ==============================================================================
def test_10_cctv_highway_citizen_photo_upload():
    from backend.services.citizen_service import process_upload

    img_path = Path("data/citizen_uploads/CIT-20261104104358-D1FA/upload.png")
    if not img_path.exists():
        pytest.skip("Test image upload.png not found")

    info = {
        "violation_type": "AUTO_DETECT",
        "location": "Airport Highway Tollway",
        "name": "Audit Officer",
        "note": "Vehicles on highway carriageway",
    }
    result = process_upload(img_path, info)

    assert result["detected"] >= 1
    records = result["records"]
    ww_records = [r for r in records if r["type"] == "WRONG_WAY"]
    assert len(ww_records) >= 1

    rec = ww_records[0]
    assert rec["motion_verified"] is False
    assert rec["confidence"] > 0.70
    assert "UNVERIFIED" in rec["verification_status"]


# ==============================================================================
# TEST 11: Deep Learning Bi-LSTM Checkpoint Output Verification
# Expected: wrong_way_lstm.pt loads, executes forward pass, and outputs correct class probabilities
# ==============================================================================
def test_11_deep_learning_bi_lstm_model_inference():
    import torch
    from algorithms.violation_rules.wrong_way_net import WrongWayLSTM, extract_trajectory_features

    model_path = Path("models/wrong_way_lstm.pt")
    assert model_path.exists(), "models/wrong_way_lstm.pt checkpoint must exist"

    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    assert "model_state_dict" in ckpt
    assert ckpt["input_dim"] == 7
    assert ckpt["num_classes"] == 2

    model = WrongWayLSTM(
        input_dim=ckpt["input_dim"],
        hidden_dim=ckpt["hidden_dim"],
        num_layers=ckpt["num_layers"],
        num_classes=ckpt["num_classes"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 1. Test retrograde / counterflow motion (driving upward against downward [0, 1] lane)
    wrong_traj = [
        [500.0, 500.0], [500.0, 470.0], [502.0, 440.0], [501.0, 410.0],
        [500.0, 380.0], [503.0, 350.0], [500.0, 320.0], [502.0, 290.0]
    ]
    feat_wrong = extract_trajectory_features(wrong_traj, [0.0, 1.0])
    assert feat_wrong is not None
    assert feat_wrong.shape == (8, 7)

    t_wrong = torch.tensor(feat_wrong, dtype=torch.float32).unsqueeze(0)
    probs_wrong = model.predict_proba(t_wrong).detach().cpu().numpy()[0]
    print(f"\n[Test 11 Output] Retrograde Trajectory Probabilities: Compliant={probs_wrong[0]:.4f}, Wrong-Way={probs_wrong[1]:.4f}")
    assert probs_wrong[1] > 0.85, f"Expected high wrong-way probability, got {probs_wrong[1]}"
    assert np.argmax(probs_wrong) == 1, "Argmax class output must be 1 (WRONG_WAY)"

    # 2. Test compliant lane-following motion (driving downward with downward [0, 1] lane)
    legal_traj = [
        [500.0, 200.0], [502.0, 230.0], [505.0, 260.0], [503.0, 290.0],
        [501.0, 320.0], [504.0, 350.0], [506.0, 380.0], [508.0, 410.0]
    ]
    feat_legal = extract_trajectory_features(legal_traj, [0.0, 1.0])
    assert feat_legal is not None
    assert feat_legal.shape == (8, 7)

    t_legal = torch.tensor(feat_legal, dtype=torch.float32).unsqueeze(0)
    probs_legal = model.predict_proba(t_legal).detach().cpu().numpy()[0]
    print(f"[Test 11 Output] Compliant Trajectory Probabilities: Compliant={probs_legal[0]:.4f}, Wrong-Way={probs_legal[1]:.4f}")
    assert probs_legal[0] > 0.85, f"Expected high compliant probability, got {probs_legal[0]}"
    assert np.argmax(probs_legal) == 0, "Argmax class output must be 0 (COMPLIANT_FLOW)"



