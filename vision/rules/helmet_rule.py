import time
import logging
from typing import Dict, Any, List
import cv2
import numpy as np
from vision.rules.base_rule import BaseViolationRule
from vision.association.rider_association import MotorcycleRiderGroup

logger = logging.getLogger(__name__)


class HelmetViolationRule(BaseViolationRule):
    """
    Evaluates helmet compliance for motorcycle riders and pillion passengers.
    Uses rider head-crop analysis and custom model (or baseline heuristic in demo mode).
    """

    def __init__(
        self,
        temporal_window: int = 10,
        min_positive_frames: int = 6,
        cooldown_seconds: float = 10.0,
    ):
        super().__init__(
            rule_name="NO_HELMET",
            temporal_window=temporal_window,
            min_positive_frames=min_positive_frames,
            cooldown_seconds=cooldown_seconds,
        )

    def evaluate(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        groups: List[MotorcycleRiderGroup] = context.get("motorcycle_groups", [])
        detector = context.get("detector")
        frame: np.ndarray = context.get("frame")
        violations: List[Dict[str, Any]] = []

        if frame is None or not groups:
            return violations

        has_custom = bool(
            detector
            and (
                getattr(detector, "has_custom_model", False)
                or getattr(detector, "has_custom_helmet_model", False)
            )
        )
        detections = context.get("detections", [])
        frame_idx = context.get("frame_idx", 0)
        fps = context.get("fps", 25.0) or 25.0
        current_time = frame_idx / fps

        # Separate custom model detections if available
        helmet_centres = []
        no_helmet_centres = []
        if has_custom and detections:
            for d in detections:
                cname = d.class_name.lower().replace(" ", "_")
                if cname in ["no_helmet", "without_helmet", "nohelmet"]:
                    no_helmet_centres.append((d.center[0], d.center[1], d.confidence))
                elif cname in ["helmet", "with_helmet", "withhelmet"]:
                    helmet_centres.append((d.center[0], d.center[1], d.confidence))

        for group in groups:
            bike_id = group.motorcycle.track_id
            for rider in group.riders:
                rx1, ry1, rx2, ry2 = [float(v) for v in rider.bbox]
                rider_h = ry2 - ry1
                if rider_h < 30:
                    continue

                is_no_helmet = False
                conf = 0.85
                mode = "CUSTOM_MODEL" if has_custom else "BASELINE_HEURISTIC"

                if has_custom:
                    # Top 35% of rider's bounding box
                    head_y_max = ry1 + 0.35 * rider_h

                    # Check for no_helmet box centre inside top 35%
                    matching_no_helmet = [
                        c for c in no_helmet_centres
                        if rx1 <= c[0] <= rx2 and ry1 <= c[1] <= head_y_max
                    ]
                    # Check for helmet box centre inside top 35%
                    matching_helmet = [
                        c for c in helmet_centres
                        if rx1 <= c[0] <= rx2 and ry1 <= c[1] <= head_y_max
                    ]

                    if matching_no_helmet:
                        is_no_helmet = True
                        conf = max(c[2] for c in matching_no_helmet)
                    elif matching_helmet:
                        is_no_helmet = False  # helmet box there -> not a violation
                    else:
                        is_no_helmet = False  # no box -> unknown (not a violation)
                else:
                    # Baseline heuristic when NO custom model exists
                    head_y2 = min(int(ry2), int(ry1 + 0.28 * rider_h))
                    h_f, w_f = frame.shape[:2]
                    head_crop = frame[max(0, int(ry1)):min(h_f, head_y2), max(0, int(rx1)):min(w_f, int(rx2))]
                    if head_crop.size > 0:
                        gray_head = cv2.cvtColor(head_crop, cv2.COLOR_BGR2GRAY)
                        edges = cv2.Canny(gray_head, 80, 180)
                        edge_ratio = np.count_nonzero(edges) / max(1, edges.size)
                        if edge_ratio > 0.08:
                            is_no_helmet = True
                            conf = min(0.92, 0.70 + float(edge_ratio * 2.0))

                # Temporal confirmation check using video time
                if self.record_frame_evidence(rider.track_id, is_no_helmet, current_time=current_time):
                    violations.append({
                        "violation_type": "NO_HELMET",
                        "track_id": bike_id,
                        "rider_track_id": rider.track_id,
                        "confidence": round(conf, 3),
                        "timestamp": current_time,
                        "evidence_required": True,
                        "extra": {
                            "bike_track_id": bike_id,
                            "rider_count": group.rider_count,
                            "mode": mode,
                        },
                    })

        return violations

