import time
import pytest
import numpy as np
from vision.tracker.base_tracker import TrackedObject
from vision.association.rider_association import MotorcycleRiderGroup
from vision.rules.helmet_rule import HelmetViolationRule
from vision.rules.triple_riding_rule import TripleRidingRule
from vision.rules.red_light_rule import RedLightViolationRule, line_segments_intersect
from vision.rules.wrong_way_rule import WrongWayViolationRule


def test_triple_riding_rule_temporal_confirmation():
    rule = TripleRidingRule(temporal_window=5, min_positive_frames=3, cooldown_seconds=5.0)

    bike = TrackedObject(track_id=101, class_name="motorcycle", class_id=3, bbox=[100, 100, 200, 200], confidence=0.9)
    group = MotorcycleRiderGroup(bike)

    # 3 riders associated with motorcycle -> Triple riding!
    r1 = TrackedObject(track_id=201, class_name="person", class_id=0, bbox=[120, 80, 160, 160], confidence=0.85)
    r2 = TrackedObject(track_id=202, class_name="person", class_id=0, bbox=[140, 90, 180, 170], confidence=0.85)
    r3 = TrackedObject(track_id=203, class_name="person", class_id=0, bbox=[150, 95, 190, 175], confidence=0.85)

    group.add_rider(r1)
    group.add_rider(r2)
    group.add_rider(r3)

    context = {"motorcycle_groups": [group]}

    # Frame 1: positive, but needs 3 frames
    v1 = rule.evaluate(context)
    assert len(v1) == 0

    # Frame 2: positive, needs 3 frames
    v2 = rule.evaluate(context)
    assert len(v2) == 0

    # Frame 3: 3rd positive frame -> Confirmation triggered!
    v3 = rule.evaluate(context)
    assert len(v3) == 1
    assert v3[0]["violation_type"] == "TRIPLE_RIDING"
    assert v3[0]["track_id"] == 101

    # Frame 4: cooldown in effect -> Duplicate prevention!
    v4 = rule.evaluate(context)
    assert len(v4) == 0, "Duplicate violation should be suppressed during cooldown"


def test_red_light_stop_line_intersection():
    # Stop line at Y = 500 from X=100 to X=900
    stop_line = {"x1": 100.0, "y1": 500.0, "x2": 900.0, "y2": 500.0}
    rule = RedLightViolationRule(stop_line=stop_line, temporal_window=3, min_positive_frames=1, cooldown_seconds=5.0)

    # Car moving downward across the line from Y=480 to Y=520
    car = TrackedObject(track_id=55, class_name="car", class_id=2, bbox=[400, 460, 500, 500], confidence=0.92)
    car.trajectory = [[450.0, 480.0], [450.0, 520.0]]  # Crossed line

    context = {
        "tracked_vehicles": [car],
        "traffic_signal_state": "RED",
        "frame": np.zeros((720, 1280, 3), dtype=np.uint8),
    }

    viols = rule.evaluate(context)
    assert len(viols) == 1
    assert viols[0]["violation_type"] == "RED_LIGHT_JUMP"
    assert viols[0]["track_id"] == 55

    # Should not trigger if signal is GREEN
    car2 = TrackedObject(track_id=56, class_name="car", class_id=2, bbox=[400, 460, 500, 500], confidence=0.92)
    car2.trajectory = [[450.0, 480.0], [450.0, 520.0]]
    context_green = {
        "tracked_vehicles": [car2],
        "traffic_signal_state": "GREEN",
        "frame": np.zeros((720, 1280, 3), dtype=np.uint8),
    }
    viols_green = rule.evaluate(context_green)
    assert len(viols_green) == 0


def test_wrong_way_direction_detection():
    # Permitted lane vector: [0, 1] (Downwards)
    rule = WrongWayViolationRule(lane_vector=[0.0, 1.0], max_angle_deviation_deg=120.0, temporal_window=3, min_positive_frames=1, cooldown_seconds=5.0)

    # Vehicle driving backwards (Upwards): dy is negative
    v_wrong = TrackedObject(track_id=88, class_name="car", class_id=2, bbox=[300, 300, 400, 400], confidence=0.88)
    v_wrong.trajectory = [
        [350.0, 400.0],
        [350.0, 380.0],
        [350.0, 360.0],
        [350.0, 340.0],
        [350.0, 320.0],
        [350.0, 300.0],
        [350.0, 280.0],
    ]

    context = {"tracked_vehicles": [v_wrong]}
    viols = rule.evaluate(context)
    assert len(viols) == 1
    assert viols[0]["violation_type"] == "WRONG_WAY"
    assert viols[0]["track_id"] == 88


def test_helmet_rule_detection():
    from unittest.mock import MagicMock
    from vision.detector.detector import DetectionResult

    rule = HelmetViolationRule(temporal_window=3, min_positive_frames=1, cooldown_seconds=5.0)

    bike = TrackedObject(track_id=12, class_name="motorcycle", class_id=3, bbox=[200, 200, 350, 350], confidence=0.9)
    group = MotorcycleRiderGroup(bike)

    # Rider bbox: [220, 150, 300, 300] -> top 35% is Y in [150, 202.5], X in [220, 300]
    rider = TrackedObject(track_id=45, class_name="person", class_id=0, bbox=[220, 150, 300, 300], confidence=0.88)
    group.add_rider(rider)

    # Mock custom detector
    mock_detector = MagicMock()
    mock_detector.has_custom_model = True

    # 1. Test no_helmet detection whose centre (260, 175) falls in top 35% of rider
    no_helmet_det = DetectionResult(
        bbox=[240, 160, 280, 190],
        confidence=0.92,
        class_id=1,
        class_name="no_helmet",
    )

    context = {
        "motorcycle_groups": [group],
        "frame": np.zeros((400, 600, 3), dtype=np.uint8),
        "detector": mock_detector,
        "detections": [no_helmet_det],
        "frame_idx": 0,
        "fps": 25.0,
    }

    viols = rule.evaluate(context)
    assert len(viols) == 1
    assert viols[0]["violation_type"] == "NO_HELMET"
    assert viols[0]["track_id"] == 12
    assert viols[0]["extra"]["mode"] == "CUSTOM_MODEL"

    # 2. Test helmet detection: a helmet box in top 35% should NOT trigger a violation
    bike2 = TrackedObject(track_id=13, class_name="motorcycle", class_id=3, bbox=[200, 200, 350, 350], confidence=0.9)
    group2 = MotorcycleRiderGroup(bike2)
    rider2 = TrackedObject(track_id=46, class_name="person", class_id=0, bbox=[220, 150, 300, 300], confidence=0.88)
    group2.add_rider(rider2)

    helmet_det = DetectionResult(
        bbox=[240, 160, 280, 190],
        confidence=0.95,
        class_id=0,
        class_name="helmet",
    )
    context_helmet = {
        "motorcycle_groups": [group2],
        "frame": np.zeros((400, 600, 3), dtype=np.uint8),
        "detector": mock_detector,
        "detections": [helmet_det],
        "frame_idx": 0,
        "fps": 25.0,
    }
    viols_helmet = rule.evaluate(context_helmet)
    assert len(viols_helmet) == 0

