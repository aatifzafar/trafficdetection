"""Illegal parking and unsafe movement rules, on synthetic tracks (no model needed)."""

import math

from algorithms.tracking.base_tracker import TrackedObject
from algorithms.violation_rules.illegal_parking_rule import IllegalParkingRule
from algorithms.violation_rules.unsafe_movement_rule import UnsafeMovementRule, count_reversals
from algorithms.violation_rules.rule_engine import TrafficRuleEngine
from algorithms.calibration.camera_calibration import config_for_camera

ZONE = [[100, 300], [500, 300], [500, 600], [100, 600]]
FPS = 10.0


def _car(tid, box, frame_idx=0):
    return TrackedObject(track_id=tid, class_name="car", class_id=2, bbox=box, confidence=0.9, frame_idx=frame_idx)


def _run_parking(rule, boxes):
    """boxes: list of bbox per frame for one car (track 1). Returns frames where a violation fired."""
    car = _car(1, boxes[0])
    fired = []
    for i, b in enumerate(boxes, start=1):
        car.update(b, 0.9, i)
        if rule.evaluate({"tracked_vehicles": [car], "frame_idx": i, "fps": FPS}):
            fired.append(i)
    return fired


def test_parked_car_in_zone_is_flagged_once():
    rule = IllegalParkingRule(zones=[ZONE], max_stationary_seconds=5, cooldown_seconds=300)
    fired = _run_parking(rule, [[200, 400, 320, 480]] * 120)  # 12 s still, wheels inside the zone
    assert len(fired) == 1
    assert 50 <= fired[0] <= 52  # after 5 s at 10 fps


def test_car_outside_zone_is_not_flagged():
    rule = IllegalParkingRule(zones=[ZONE], max_stationary_seconds=5)
    assert _run_parking(rule, [[600, 400, 720, 480]] * 120) == []


def test_moving_car_in_zone_is_not_flagged():
    rule = IllegalParkingRule(zones=[ZONE], max_stationary_seconds=5)
    boxes = [[110 + 3 * i, 400, 230 + 3 * i, 480] for i in range(120)]  # crawls through the zone
    assert _run_parking(rule, boxes) == []


def test_jitter_does_not_reset_the_timer():
    rule = IllegalParkingRule(zones=[ZONE], max_stationary_seconds=5)
    boxes = [[200 + (i % 3) * 4, 400 + (i % 2) * 3, 320 + (i % 3) * 4, 480 + (i % 2) * 3] for i in range(80)]
    assert len(_run_parking(rule, boxes)) == 1


def test_no_zone_means_rule_is_inactive():
    rule = IllegalParkingRule(zones=[], max_stationary_seconds=1)
    assert rule.status == "no_zone"
    assert _run_parking(rule, [[200, 400, 320, 480]] * 50) == []


def _run_unsafe(rule, centres, w=60, h=40):
    car = _car(7, [centres[0][0] - w / 2, centres[0][1] - h / 2, centres[0][0] + w / 2, centres[0][1] + h / 2])
    fired = []
    for i, (cx, cy) in enumerate(centres[1:], start=1):
        car.update([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 0.9, i)
        out = rule.evaluate({"tracked_vehicles": [car], "frame_idx": i, "fps": 25.0})
        fired.extend(out)
    return fired


def test_count_reversals():
    assert count_reversals([0, 1, 2, 3, 4, 5], 1.0) == 0
    assert count_reversals([0, 5, 0, 5, 0], 2.0) == 3
    assert count_reversals([0, 0.5, 0, 0.5, 0], 2.0) == 0  # small wiggles ignored


def test_weaving_vehicle_is_flagged():
    rule = UnsafeMovementRule()
    # moves down the image 8 px per frame while swinging +-40 px sideways (vehicle width 60)
    centres = [(400 + 40 * math.sin(i / 4.0), 100 + 8 * i) for i in range(70)]
    v = _run_unsafe(rule, centres)
    assert v, "weaving should be flagged"
    assert "WEAVING" in v[0]["extra"]["reasons"]


def test_straight_vehicle_with_tracker_jitter_is_not_flagged():
    rule = UnsafeMovementRule()
    centres = [(400 + (3 if i % 2 else -3), 100 + 8 * i) for i in range(70)]
    assert _run_unsafe(rule, centres) == []


def test_lane_change_is_not_weaving():
    rule = UnsafeMovementRule()
    centres = [(400 + min(60, max(0, (i - 20) * 4)), 100 + 8 * i) for i in range(70)]  # one smooth shift
    assert _run_unsafe(rule, centres) == []


def test_over_speed_needs_calibration():
    centres = [(400, 100 + 20 * i) for i in range(40)]  # 20 px per frame at 25 fps = 500 px/s
    assert _run_unsafe(UnsafeMovementRule(speed_limit_kmph=40), centres) == []  # no pixels_per_meter -> off
    v = _run_unsafe(UnsafeMovementRule(pixels_per_meter=10, speed_limit_kmph=40), centres)  # 50 m/s = 180 km/h
    assert v and "OVER_SPEED" in v[0]["extra"]["reasons"]
    assert 170 < v[0]["extra"]["speed_kmph"] < 190


def test_engine_and_camera_calibration_wiring():
    cfg = {"violations": {"illegal_parking": True, "unsafe_movement": True, "helmet": False, "triple_riding": False,
                          "red_light": False, "wrong_way": False, "seatbelt": False, "mobile_phone": False},
           "camera_calibration": {"3": {"no_parking_zones": [ZONE], "pixels_per_meter": 12, "speed_limit_kmph": 30}}}
    eng = TrafficRuleEngine(config=config_for_camera(cfg, 3))
    assert set(eng.rules) == {"illegal_parking", "unsafe_movement"}
    assert len(eng.rules["illegal_parking"].zones) == 1
    assert eng.rules["unsafe_movement"].pixels_per_meter == 12
    assert eng.rules["unsafe_movement"].speed_limit_kmph == 30
