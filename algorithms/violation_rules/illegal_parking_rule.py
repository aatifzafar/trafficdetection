"""
Illegal parking: a vehicle that stays still inside a calibrated no-parking zone for longer than the
allowed time.

Zones are drawn per camera on the dashboard's Camera setup page (polygons in image pixels, saved to
data/camera_calibration.yaml as `no_parking_zones`). Without a zone the rule does nothing, because
"parked" only means something where parking is not allowed.

A vehicle counts as inside a zone when the middle of the bottom edge of its box (where the wheels
touch the road) is inside the polygon. It counts as still while its centre stays within a small
distance of where it stopped; moving further restarts the timer.
"""

import math
import logging
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from algorithms.violation_rules.base_rule import BaseViolationRule

logger = logging.getLogger(__name__)

PARKABLE = {"car", "motorcycle", "bus", "truck", "auto", "autorickshaw", "bicycle"}


class IllegalParkingRule(BaseViolationRule):
    def __init__(
        self,
        zones: Optional[List[List[List[float]]]] = None,
        max_stationary_seconds: float = 60.0,
        move_tolerance_px: float = 25.0,
        grace_seconds: float = 3.0,
        cooldown_seconds: float = 300.0,
    ):
        super().__init__(
            rule_name="ILLEGAL_PARKING",
            temporal_window=1,
            min_positive_frames=1,
            cooldown_seconds=cooldown_seconds,
        )
        self.zones = [np.array(z, dtype=np.float32) for z in (zones or []) if z and len(z) >= 3]
        self.max_stationary_seconds = float(max_stationary_seconds)
        self.move_tolerance_px = float(move_tolerance_px)
        self.grace_seconds = float(grace_seconds)  # a vehicle hidden for a moment keeps its timer
        # track_id -> {"zone", "anchor", "since", "last_t"}
        self._state: Dict[int, Dict[str, Any]] = {}
        self.status = "ready" if self.zones else "no_zone"

    def _zone_of(self, x: float, y: float) -> Optional[int]:
        for i, z in enumerate(self.zones):
            if cv2.pointPolygonTest(z, (float(x), float(y)), False) >= 0:
                return i
        return None

    def evaluate(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        violations: List[Dict[str, Any]] = []
        if not self.zones:
            return violations
        fps = context.get("fps", 25.0) or 25.0
        now = context.get("frame_idx", 0) / fps

        for v in context.get("tracked_vehicles", []):
            if v.class_name.lower() not in PARKABLE or v.track_id is None:
                continue
            x1, y1, x2, y2 = v.bbox
            zone = self._zone_of((x1 + x2) / 2.0, y2)
            if zone is None:
                self._state.pop(v.track_id, None)
                continue
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            tol = max(self.move_tolerance_px, 0.15 * math.hypot(x2 - x1, y2 - y1))
            st = self._state.get(v.track_id)
            if st is None or st["zone"] != zone or math.hypot(cx - st["anchor"][0], cy - st["anchor"][1]) > tol:
                st = {"zone": zone, "anchor": (cx, cy), "since": now, "last_t": now}
                self._state[v.track_id] = st
            st["last_t"] = now
            dwell = now - st["since"]
            if dwell >= self.max_stationary_seconds and not self.is_cooling_down(v.track_id, current_time=now):
                self.last_triggered[v.track_id] = now
                violations.append({
                    "violation_type": "ILLEGAL_PARKING",
                    "track_id": v.track_id,
                    "confidence": round(min(0.95, 0.75 + 0.05 * dwell / self.max_stationary_seconds), 3),
                    "timestamp": now,
                    "evidence_required": True,
                    "extra": {
                        "vehicle_class": v.class_name,
                        "zone_index": zone,
                        "stationary_seconds": round(dwell, 1),
                        "allowed_seconds": self.max_stationary_seconds,
                    },
                })

        # forget vehicles that left the picture
        for tid in [t for t, s in self._state.items() if now - s["last_t"] > self.grace_seconds]:
            del self._state[tid]
        return violations
