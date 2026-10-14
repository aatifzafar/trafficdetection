import time
import logging
from typing import Dict, Any, List
from vision.rules.base_rule import BaseViolationRule

logger = logging.getLogger(__name__)


class MobilePhoneViolationRule(BaseViolationRule):
    """
    Detects mobile-phone usage while operating a vehicle.
    Correlates detected 'cell phone' objects with driver/rider bounding boxes.
    """

    def __init__(
        self,
        temporal_window: int = 8,
        min_positive_frames: int = 5,
        cooldown_seconds: float = 15.0,
    ):
        super().__init__(
            rule_name="MOBILE_PHONE_WHILE_DRIVING",
            temporal_window=temporal_window,
            min_positive_frames=min_positive_frames,
            cooldown_seconds=cooldown_seconds,
        )

    def evaluate(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        detections = context.get("detections", [])
        tracked_vehicles = context.get("tracked_vehicles", [])
        motorcycle_groups = context.get("motorcycle_groups", [])
        violations = []

        # Find cell phone detections
        phone_dets = [
            d for d in detections if d.class_name.lower() in {"cell phone", "phone", "mobile phone"}
        ]

        if not phone_dets:
            return violations

        # Check association with motorcycle riders
        for group in motorcycle_groups:
            bike_id = group.motorcycle.track_id
            for rider in group.riders:
                rx1, ry1, rx2, ry2 = rider.bbox
                rider_phone_detected = False
                best_conf = 0.0

                for p in phone_dets:
                    px1, py1, px2, py2 = p.bbox
                    # Check if phone is inside or overlapping rider bbox
                    if not (px2 < rx1 or px1 > rx2 or py2 < ry1 or py1 > ry2):
                        rider_phone_detected = True
                        best_conf = max(best_conf, p.confidence)

                if self.record_frame_evidence(rider.track_id, rider_phone_detected):
                    violations.append({
                        "violation_type": "MOBILE_PHONE_WHILE_DRIVING",
                        "track_id": bike_id,
                        "confidence": round(best_conf or 0.85, 3),
                        "timestamp": time.time(),
                        "evidence_required": True,
                        "extra": {
                            "vehicle_type": "motorcycle",
                            "rider_track_id": rider.track_id,
                        },
                    })

        # Check association with car drivers
        for v in tracked_vehicles:
            if v.class_name.lower() in {"car", "truck", "bus"}:
                vx1, vy1, vx2, vy2 = v.bbox
                car_phone_detected = False
                best_conf = 0.0

                for p in phone_dets:
                    px1, py1, px2, py2 = p.bbox
                    if not (px2 < vx1 or px1 > vx2 or py2 < vy1 or py1 > vy2):
                        car_phone_detected = True
                        best_conf = max(best_conf, p.confidence)

                if self.record_frame_evidence(v.track_id, car_phone_detected):
                    violations.append({
                        "violation_type": "MOBILE_PHONE_WHILE_DRIVING",
                        "track_id": v.track_id,
                        "confidence": round(best_conf or 0.85, 3),
                        "timestamp": time.time(),
                        "evidence_required": True,
                        "extra": {"vehicle_type": v.class_name},
                    })

        return violations
