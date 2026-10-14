import time
import math
import logging
from typing import Dict, Any, List
import numpy as np
from vision.rules.base_rule import BaseViolationRule
from vision.tracker.base_tracker import TrackedObject

logger = logging.getLogger(__name__)


class WrongWayViolationRule(BaseViolationRule):
    """
    Detects wrong-way driving against configured lane flow direction vectors.
    Uses multi-frame velocity centroid vector smoothing and angular deviation thresholding.
    """

    def __init__(
        self,
        lane_vector: List[float] = None,
        max_angle_deviation_deg: float = 120.0,
        min_displacement_px: float = 15.0,
        temporal_window: int = 10,
        min_positive_frames: int = 6,
        cooldown_seconds: float = 15.0,
    ):
        super().__init__(
            rule_name="WRONG_WAY",
            temporal_window=temporal_window,
            min_positive_frames=min_positive_frames,
            cooldown_seconds=cooldown_seconds,
        )
        self.lane_vector = lane_vector or [0.0, 1.0]  # Default downward traffic flow
        self.max_angle_deviation_deg = max_angle_deviation_deg
        self.min_displacement_px = min_displacement_px

    def set_lane_vector(self, dx: float, dy: float):
        norm = math.sqrt(dx * dx + dy * dy)
        if norm > 1e-4:
            self.lane_vector = [dx / norm, dy / norm]

    def evaluate(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        vehicles: List[TrackedObject] = context.get("tracked_vehicles", [])
        violations: List[Dict[str, Any]] = []

        # Reference vector normalized
        ref_dx, ref_dy = self.lane_vector
        ref_norm = math.sqrt(ref_dx * ref_dx + ref_dy * ref_dy)
        if ref_norm < 1e-4:
            return violations
        ref_dx /= ref_norm
        ref_dy /= ref_norm

        for v in vehicles:
            if len(v.trajectory) < 6:
                continue

            # Velocity vector across last 8 frames
            vx, vy = v.get_velocity_vector(window=8)
            displacement = math.sqrt(vx * vx + vy * vy)

            # Must exceed minimum movement threshold to avoid stationary jitter
            if displacement < self.min_displacement_px:
                self.record_frame_evidence(v.track_id, False)
                continue

            # Dot product to find angle
            dot = (vx * ref_dx + vy * ref_dy) / displacement
            dot = max(-1.0, min(1.0, dot))
            angle_rad = math.acos(dot)
            angle_deg = math.degrees(angle_rad)

            is_wrong_way = angle_deg >= self.max_angle_deviation_deg

            if self.record_frame_evidence(v.track_id, is_wrong_way):
                conf = min(0.96, 0.70 + (angle_deg / 180.0) * 0.25)
                violations.append({
                    "violation_type": "WRONG_WAY",
                    "track_id": v.track_id,
                    "confidence": round(conf, 3),
                    "timestamp": time.time(),
                    "evidence_required": True,
                    "extra": {
                        "lane_vector": self.lane_vector,
                        "velocity_vector": [round(vx, 2), round(vy, 2)],
                        "angle_deviation_deg": round(angle_deg, 1),
                        "displacement_px": round(displacement, 1),
                    },
                })

        return violations
