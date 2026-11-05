"""
Unsafe vehicle movement, from each vehicle's tracked path:

1. Weaving (zig-zag): the vehicle swings left and right of its own line of travel repeatedly.
   Measured in units of the vehicle's own width, so it works without any camera calibration.
2. Over-speeding: only when the camera has `pixels_per_meter` calibrated (Camera setup page);
   speed = distance moved in pixels / pixels_per_meter / time. Without that calibration the speed in
   km/h cannot be known, so this check stays off.

Both are confirmed over several frames (6 of 10 by default) like the other rules.
"""

import math
import logging
from typing import Any, Dict, List, Optional

from algorithms.violation_rules.base_rule import BaseViolationRule

logger = logging.getLogger(__name__)

MOVING = {"car", "motorcycle", "bus", "truck", "auto", "autorickshaw", "bicycle"}


def count_reversals(values: List[float], threshold: float) -> int:
    """Number of direction changes in a 1-D signal, ignoring wiggles smaller than threshold."""
    if len(values) < 3 or threshold <= 0:
        return 0
    trend, hi, lo, rev = 0, values[0], values[0], 0
    for x in values[1:]:
        if trend == 0:
            hi, lo = max(hi, x), min(lo, x)
            if hi - x >= threshold:
                trend, lo = -1, x
            elif x - lo >= threshold:
                trend, hi = 1, x
        elif trend == 1:
            if x > hi:
                hi = x
            elif hi - x >= threshold:
                rev, trend, lo = rev + 1, -1, x
        else:
            if x < lo:
                lo = x
            elif x - lo >= threshold:
                rev, trend, hi = rev + 1, 1, x
    return rev


def _smooth(points: List[List[float]], k: int = 5) -> List[List[float]]:
    if len(points) < k:
        return points
    out = []
    for i in range(len(points)):
        a, b = max(0, i - k // 2), min(len(points), i + k // 2 + 1)
        seg = points[a:b]
        out.append([sum(p[0] for p in seg) / len(seg), sum(p[1] for p in seg) / len(seg)])
    return out


class UnsafeMovementRule(BaseViolationRule):
    def __init__(
        self,
        pixels_per_meter: Optional[float] = None,
        speed_limit_kmph: Optional[float] = None,
        weave_min_reversals: int = 2,
        weave_amplitude_ratio: float = 0.35,
        min_track_frames: int = 20,
        temporal_window: int = 10,
        min_positive_frames: int = 6,
        cooldown_seconds: float = 15.0,
    ):
        super().__init__(
            rule_name="UNSAFE_MOVEMENT",
            temporal_window=temporal_window,
            min_positive_frames=min_positive_frames,
            cooldown_seconds=cooldown_seconds,
        )
        self.pixels_per_meter = float(pixels_per_meter) if pixels_per_meter else None
        self.speed_limit_kmph = float(speed_limit_kmph) if speed_limit_kmph else None
        self.weave_min_reversals = int(weave_min_reversals)
        self.weave_amplitude_ratio = float(weave_amplitude_ratio)
        self.min_track_frames = int(min_track_frames)

    def _speed_kmph(self, traj: List[List[float]], fps: float, n: int = 6) -> Optional[float]:
        if not self.pixels_per_meter or len(traj) < n:
            return None
        p0, p1 = traj[-n], traj[-1]
        metres = math.hypot(p1[0] - p0[0], p1[1] - p0[1]) / self.pixels_per_meter
        return metres / ((n - 1) / fps) * 3.6

    def _weaving(self, traj: List[List[float]], width: float):
        pts = _smooth(traj)
        (x0, y0), (x1, y1) = pts[0], pts[-1]
        length = math.hypot(x1 - x0, y1 - y0)
        if length < 2.0 * width:  # not travelling far enough to judge
            return 0, 0.0
        ux, uy = (x1 - x0) / length, (y1 - y0) / length
        lateral = [(p[0] - x0) * -uy + (p[1] - y0) * ux for p in pts]
        thr = self.weave_amplitude_ratio * width
        swing = (max(lateral) - min(lateral)) / max(width, 1.0)
        return count_reversals(lateral, thr), swing

    def evaluate(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        violations: List[Dict[str, Any]] = []
        fps = context.get("fps", 25.0) or 25.0
        now = context.get("frame_idx", 0) / fps

        for v in context.get("tracked_vehicles", []):
            if v.class_name.lower() not in MOVING or v.track_id is None:
                continue
            traj = v.trajectory
            if len(traj) < self.min_track_frames:
                continue
            width = max(1.0, v.bbox[2] - v.bbox[0])
            reversals, swing = self._weaving(traj, width)
            speed = self._speed_kmph(traj, fps)
            weaving = reversals >= self.weave_min_reversals
            overspeed = speed is not None and self.speed_limit_kmph is not None and speed > self.speed_limit_kmph
            if self.record_frame_evidence(v.track_id, weaving or overspeed, current_time=now):
                reasons = []
                if weaving:
                    reasons.append("WEAVING")
                if overspeed:
                    reasons.append("OVER_SPEED")
                violations.append({
                    "violation_type": "UNSAFE_MOVEMENT",
                    "track_id": v.track_id,
                    "confidence": round(0.80 if overspeed else min(0.92, 0.70 + 0.05 * reversals), 3),
                    "timestamp": now,
                    "evidence_required": True,
                    "extra": {
                        "reasons": reasons,
                        "vehicle_class": v.class_name,
                        "direction_changes": reversals,
                        "lateral_swing_widths": round(swing, 2),
                        "speed_kmph": round(speed, 1) if speed is not None else None,
                        "speed_limit_kmph": self.speed_limit_kmph,
                    },
                })
        return violations
