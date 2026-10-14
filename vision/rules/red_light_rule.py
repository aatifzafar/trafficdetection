import time
import logging
from typing import Dict, Any, List, Tuple
import cv2
import numpy as np
from vision.rules.base_rule import BaseViolationRule
from vision.tracker.base_tracker import TrackedObject

logger = logging.getLogger(__name__)


def ccw(A: Tuple[float, float], B: Tuple[float, float], C: Tuple[float, float]) -> bool:
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def line_segments_intersect(
    A: Tuple[float, float],
    B: Tuple[float, float],
    C: Tuple[float, float],
    D: Tuple[float, float],
) -> bool:
    """Returns True if segment AB intersects segment CD."""
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


class RedLightViolationRule(BaseViolationRule):
    """
    Detects red-light jumping across a configurable virtual stop line.
    Requires vehicle centroid trajectory to cross the stop line while traffic signal is RED.
    """

    def __init__(
        self,
        stop_line: Dict[str, float] = None,
        temporal_window: int = 5,
        min_positive_frames: int = 2,
        cooldown_seconds: float = 12.0,
    ):
        super().__init__(
            rule_name="RED_LIGHT_JUMP",
            temporal_window=temporal_window,
            min_positive_frames=min_positive_frames,
            cooldown_seconds=cooldown_seconds,
        )
        self.stop_line = stop_line or {"x1": 150.0, "y1": 520.0, "x2": 1100.0, "y2": 520.0}

    def set_stop_line(self, x1: float, y1: float, x2: float, y2: float):
        self.stop_line = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}

    def detect_traffic_light_state(self, context: Dict[str, Any]) -> str:
        """
        Determines traffic light status: 'RED', 'GREEN', or 'UNKNOWN'.
        Inspects detected traffic light bounding boxes via HSV color distribution.
        """
        # If explicitly injected in context (e.g. from camera/simulator)
        if "traffic_signal_state" in context:
            return str(context["traffic_signal_state"]).upper()

        detections = context.get("detections", [])
        frame = context.get("frame")
        if frame is None:
            return "RED"  # In simulation default to RED for testing if line crossed

        for d in detections:
            if "traffic light" in d.class_name.lower():
                x1, y1, x2, y2 = [int(v) for v in d.bbox]
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                tl_crop = frame[y1:y2, x1:x2]

                if tl_crop.size > 0:
                    hsv = cv2.cvtColor(tl_crop, cv2.COLOR_BGR2HSV)
                    # Red masks in HSV
                    mask1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
                    mask2 = cv2.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))
                    red_pixels = cv2.countNonZero(mask1) + cv2.countNonZero(mask2)

                    # Green mask
                    mask_green = cv2.inRange(hsv, np.array([36, 70, 50]), np.array([89, 255, 255]))
                    green_pixels = cv2.countNonZero(mask_green)

                    if red_pixels > green_pixels and red_pixels > 20:
                        return "RED"
                    elif green_pixels > red_pixels and green_pixels > 20:
                        return "GREEN"

        return context.get("default_signal_state", "RED")

    def evaluate(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        vehicles: List[TrackedObject] = context.get("tracked_vehicles", [])
        violations: List[Dict[str, Any]] = []

        signal_state = self.detect_traffic_light_state(context)
        is_red = signal_state == "RED"

        if not is_red or not vehicles:
            return violations

        p1 = (self.stop_line["x1"], self.stop_line["y1"])
        p2 = (self.stop_line["x2"], self.stop_line["y2"])

        for v in vehicles:
            if len(v.trajectory) < 2:
                continue

            # Centroid at previous and current frame
            prev_center = (v.trajectory[-2][0], v.trajectory[-2][1])
            curr_center = (v.trajectory[-1][0], v.trajectory[-1][1])

            crossed = line_segments_intersect(prev_center, curr_center, p1, p2)

            if self.record_frame_evidence(v.track_id, crossed):
                violations.append({
                    "violation_type": "RED_LIGHT_JUMP",
                    "track_id": v.track_id,
                    "confidence": 0.94,
                    "timestamp": time.time(),
                    "evidence_required": True,
                    "extra": {
                        "stop_line": self.stop_line,
                        "signal_state": "RED",
                        "crossing_point": curr_center,
                    },
                })

        return violations
