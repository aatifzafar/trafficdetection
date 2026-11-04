import time
import math
import logging
from typing import Dict, Any, List
from algorithms.violation_rules.base_rule import BaseViolationRule
from algorithms.association.rider_association import MotorcycleRiderGroup

logger = logging.getLogger(__name__)


class TripleRidingRule(BaseViolationRule):
    """
    Detects triple (or greater) riding on two-wheelers.
    Supports both:
    1. Direct visual object detection (Method 2 - trained custom YOLO class 'MORE_THAN_TWO_PERSONS')
    2. Geometric spatial association (Method 1 - heuristic clustering of >2 riders)
    """

    def __init__(
        self,
        temporal_window: int = 10,
        min_positive_frames: int = 6,
        cooldown_seconds: float = 12.0,
    ):
        super().__init__(
            rule_name="TRIPLE_RIDING",
            temporal_window=temporal_window,
            min_positive_frames=min_positive_frames,
            cooldown_seconds=cooldown_seconds,
        )
        self._direct_tracks: Dict[int, Dict[str, Any]] = {}
        self._next_direct_id: int = 9001

    def _get_or_create_direct_track(self, cx: float, cy: float, frame_idx: int) -> int:
        """Associates unmatched direct triple-riding detections across frames by spatial proximity."""
        best_id = None
        min_dist = 160.0  # Max pixel displacement between consecutive frames
        for tid, tinfo in list(self._direct_tracks.items()):
            # Expire tracks not seen for 25 frames (~1 second)
            if frame_idx - tinfo["last_frame"] > 25:
                del self._direct_tracks[tid]
                continue
            dist = math.hypot(cx - tinfo["cx"], cy - tinfo["cy"])
            if dist < min_dist:
                min_dist = dist
                best_id = tid

        if best_id is not None:
            self._direct_tracks[best_id]["cx"] = cx
            self._direct_tracks[best_id]["cy"] = cy
            self._direct_tracks[best_id]["last_frame"] = frame_idx
            return best_id

        new_id = self._next_direct_id
        self._next_direct_id += 1
        self._direct_tracks[new_id] = {"cx": cx, "cy": cy, "last_frame": frame_idx}
        return new_id

    def evaluate(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        groups: List[MotorcycleRiderGroup] = context.get("motorcycle_groups", [])
        violations: List[Dict[str, Any]] = []

        frame_idx = context.get("frame_idx", 0)
        fps = context.get("fps", 25.0) or 25.0
        current_time = (frame_idx / fps) if frame_idx > 0 else time.time()

        # Check for direct object detection predictions (Method 2)
        raw_dets = context.get("detections", [])
        direct_triple_boxes = [
            d for d in raw_dets
            if d.class_name.lower().replace(" ", "_") in {
                "more_than_two_persons", "triple_riding", "triple_rider", "tripleriding", "triple-riding-bikes"
            }
        ]

        matched_dt_indices = set()
        for group in groups:
            bike_id = group.motorcycle.track_id
            bx1, by1, bx2, by2 = group.motorcycle.bbox
            bw = bx2 - bx1
            bh = by2 - by1

            # Match any direct model detection with this tracked motorcycle
            direct_match = False
            direct_conf = 0.88
            for idx, dt in enumerate(direct_triple_boxes):
                dcx, dcy = dt.center
                if (bx1 - 0.35 * bw <= dcx <= bx2 + 0.35 * bw) and (by1 - 1.20 * bh <= dcy <= by2 + 0.35 * bh):
                    direct_match = True
                    direct_conf = max(direct_conf, dt.confidence)
                    matched_dt_indices.add(idx)
                    break

            # Either direct detection match (Method 2) or heuristic rider count (Method 1)
            is_triple = direct_match or (group.rider_count > 2)

            if self.record_frame_evidence(bike_id, is_triple, current_time=current_time):
                conf = direct_conf if direct_match else min(0.98, 0.75 + 0.08 * group.rider_count)
                violations.append({
                    "violation_type": "TRIPLE_RIDING",
                    "track_id": bike_id,
                    "confidence": round(conf, 3),
                    "timestamp": time.time(),
                    "evidence_required": True,
                    "extra": {
                        "detection_mode": "DIRECT_MODEL" if direct_match else "SPATIAL_HEURISTIC",
                        "rider_count": max(group.rider_count, 3) if direct_match else group.rider_count,
                        "rider_ids": [r.track_id for r in group.riders],
                        "class_name": "MORE_THAN_TWO_PERSONS" if direct_match else "MOTORCYCLE_CLUSTER",
                    },
                })

        # Process any unmatched direct triple riding detections (when base COCO missed the bike)
        tracked_vehicles = context.get("tracked_vehicles", [])
        for idx, dt in enumerate(direct_triple_boxes):
            if idx in matched_dt_indices:
                continue
            dcx, dcy = dt.center
            matched_vid = None
            min_dist = float("inf")
            for tv in tracked_vehicles:
                vx1, vy1, vx2, vy2 = tv.bbox
                if vx1 <= dcx <= vx2 and vy1 <= dcy <= vy2:
                    matched_vid = tv.track_id
                    break
                dist = math.hypot(dcx - (vx1 + vx2) / 2.0, dcy - (vy1 + vy2) / 2.0)
                if dist < min_dist and dist < 150.0:
                    min_dist = dist
                    matched_vid = tv.track_id

            if matched_vid is not None:
                assigned_id = matched_vid
            elif dt.track_id is not None:
                assigned_id = dt.track_id
            else:
                assigned_id = self._get_or_create_direct_track(dcx, dcy, frame_idx)

            if self.record_frame_evidence(assigned_id, True, current_time=current_time):
                violations.append({
                    "violation_type": "TRIPLE_RIDING",
                    "track_id": assigned_id,
                    "confidence": round(dt.confidence, 3),
                    "timestamp": time.time(),
                    "evidence_required": True,
                    "extra": {
                        "detection_mode": "DIRECT_MODEL",
                        "rider_count": 3,
                        "rider_ids": [],
                        "class_name": "MORE_THAN_TWO_PERSONS",
                        "bbox": dt.bbox,
                    },
                })

        return violations
