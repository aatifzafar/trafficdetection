import time
import logging
from typing import Dict, Any, List
import cv2
import numpy as np
from algorithms.violation_rules.base_rule import BaseViolationRule
from algorithms.association.rider_association import MotorcycleRiderGroup

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
        allow_baseline: bool = False,
    ):
        super().__init__(
            rule_name="NO_HELMET",
            temporal_window=temporal_window,
            min_positive_frames=min_positive_frames,
            cooldown_seconds=cooldown_seconds,
        )
        # Without the trained model the edge heuristic cannot tell a helmet from hair
        # (tested: it flagged 23 of 23 helmet-wearing riders). It only runs if explicitly enabled.
        self.allow_baseline = allow_baseline
        self.status = "ready"

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
                or getattr(detector, "model_triple", None) is not None
            )
        )
        if not has_custom and not self.allow_baseline:
            self.status = "model_missing"
            context["helmet_status"] = "Helmet model not trained yet (models/traffic_custom.pt missing)"
            return violations
        self.status = "custom_model" if has_custom else "baseline_heuristic"

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
            for rider_idx, rider in enumerate(group.riders):
                rx1, ry1, rx2, ry2 = [float(v) for v in rider.bbox]
                rider_h = ry2 - ry1
                if rider_h < 30:
                    continue

                is_no_helmet = False
                conf = 0.85
                mode = "CUSTOM_MODEL" if has_custom else "BASELINE_HEURISTIC"

                if has_custom:
                    # Top region of rider's bounding box (generous margin for top of head)
                    rw = rx2 - rx1
                    head_y_min = ry1 - 0.20 * rider_h
                    head_y_max = ry1 + 0.40 * rider_h
                    head_x_min = rx1 - 0.15 * rw
                    head_x_max = rx2 + 0.15 * rw

                    matching_no_helmet = [
                        c for c in no_helmet_centres
                        if head_x_min <= c[0] <= head_x_max and head_y_min <= c[1] <= head_y_max
                    ]
                    matching_helmet = [
                        c for c in helmet_centres
                        if head_x_min <= c[0] <= head_x_max and head_y_min <= c[1] <= head_y_max
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
                    rw = rx2 - rx1
                    cx = (rx1 + rx2) / 2
                    h_f, w_f = frame.shape[:2]
                    head_x1 = max(0, int(cx - 0.25 * rw))
                    head_x2 = min(w_f, int(cx + 0.25 * rw))
                    head_y2 = min(int(ry2), int(ry1 + 0.28 * rider_h))
                    head_crop = frame[max(0, int(ry1)):min(h_f, head_y2), head_x1:head_x2]
                    if head_crop.size > 0:
                        gray_head = cv2.cvtColor(head_crop, cv2.COLOR_BGR2GRAY)
                        edges = cv2.Canny(gray_head, 60, 150)
                        edge_ratio = np.count_nonzero(edges) / max(1, edges.size)
                        # Natural edge density threshold in focused head region
                        if edge_ratio > 0.045:
                            is_no_helmet = True
                            conf = min(0.85, 0.60 + float(edge_ratio * 3.0))

                # Track cooldown per seating position on this motorcycle (driver vs pillion)
                # to prevent duplicate penalties caused by tracker ID switches across momentary occlusions
                seat_role = "driver" if rider_idx == 0 else f"pillion_{rider_idx}"
                cooldown_key = f"{bike_id}_{seat_role}"

                # Temporal confirmation check using video time
                if self.record_frame_evidence(cooldown_key, is_no_helmet, current_time=current_time):
                    violations.append({
                        "violation_type": "NO_HELMET",
                        "track_id": bike_id,
                        "rider_track_id": rider.track_id,
                        "confidence": round(conf, 3),
                        "timestamp": current_time,
                        "evidence_required": True,
                        "extra": {
                            "bike_track_id": bike_id,
                            "rider_track_id": rider.track_id,
                            "seat_role": seat_role,
                            "rider_count": group.rider_count,
                            "mode": mode,
                        },
                    })

        return violations

