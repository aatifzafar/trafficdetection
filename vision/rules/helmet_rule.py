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

        for group in groups:
            bike_id = group.motorcycle.track_id
            for rider in group.riders:
                rx1, ry1, rx2, ry2 = [int(v) for v in rider.bbox]
                h, w = frame.shape[:2]
                rx1, ry1 = max(0, rx1), max(0, ry1)
                rx2, ry2 = min(w, rx2), min(h, ry2)

                rider_h = ry2 - ry1
                if rider_h < 30:
                    continue

                # Head crop: top 25% of rider bounding box
                head_y2 = min(ry2, int(ry1 + 0.28 * rider_h))
                head_crop = frame[ry1:head_y2, rx1:rx2]

                is_no_helmet = False
                conf = 0.85

                if detector and getattr(detector, "has_custom_helmet_model", False):
                    # Inference with custom helmet model
                    try:
                        res = detector.helmet_model(head_crop, verbose=False)
                        if res and len(res[0].boxes) > 0:
                            for b in res[0].boxes:
                                cls_name = detector.helmet_model.names.get(int(b.cls[0].item()), "")
                                if "no_helmet" in cls_name.lower():
                                    is_no_helmet = True
                                    conf = float(b.conf[0].item())
                                    break
                    except Exception as e:
                        logger.debug(f"Helmet model error: {e}")
                else:
                    # Prototype / Research Baseline Mode:
                    # In demo mode without custom weights, inspect head crop edge density & color saturation
                    # Human hair/skin generally exhibits high texture/gradient variance compared to smooth helmets
                    if head_crop.size > 0:
                        gray_head = cv2.cvtColor(head_crop, cv2.COLOR_BGR2GRAY)
                        edges = cv2.Canny(gray_head, 80, 180)
                        edge_ratio = np.count_nonzero(edges) / max(1, edges.size)
                        # Higher edge variance and dark texture correlates with unhelmeted hair in baseline scenes
                        if edge_ratio > 0.08:
                            is_no_helmet = True
                            conf = min(0.92, 0.70 + float(edge_ratio * 2.0))

                # Temporal confirmation check
                if self.record_frame_evidence(rider.track_id, is_no_helmet):
                    violations.append({
                        "violation_type": "NO_HELMET",
                        "track_id": bike_id,  # Associate violation with vehicle
                        "rider_track_id": rider.track_id,
                        "confidence": round(conf, 3),
                        "timestamp": time.time(),
                        "evidence_required": True,
                        "extra": {
                            "bike_track_id": bike_id,
                            "rider_count": group.rider_count,
                            "mode": "CUSTOM_MODEL" if (detector and getattr(detector, "has_custom_helmet_model", False)) else "RESEARCH_PROTOTYPE_BASELINE",
                        },
                    })

        return violations
