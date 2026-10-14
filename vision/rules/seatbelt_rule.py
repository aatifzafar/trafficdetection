import time
import logging
from typing import Dict, Any, List
from vision.rules.base_rule import BaseViolationRule

logger = logging.getLogger(__name__)


class SeatbeltViolationRule(BaseViolationRule):
    """
    Evaluates driver seatbelt compliance for four-wheelers.
    Gracefully disables itself and reports 'Seatbelt detection model not configured'
    when custom weights are not present, avoiding synthetic or fake events.
    """

    def __init__(
        self,
        model_path: str = "models/seatbelt_model.pt",
        temporal_window: int = 10,
        min_positive_frames: int = 6,
        cooldown_seconds: float = 15.0,
    ):
        super().__init__(
            rule_name="NO_SEATBELT",
            temporal_window=temporal_window,
            min_positive_frames=min_positive_frames,
            cooldown_seconds=cooldown_seconds,
        )
        self.model_path = model_path
        self.is_configured = False
        self.seatbelt_model = None

        self._load_model()

    def _load_model(self):
        import os
        if os.path.exists(self.model_path):
            try:
                from ultralytics import YOLO
                self.seatbelt_model = YOLO(self.model_path)
                self.is_configured = True
                logger.info(f"Loaded seatbelt model from {self.model_path}")
            except Exception as e:
                logger.warning(f"Failed to load seatbelt model: {e}")
        else:
            logger.info("Seatbelt detection model not configured. Rule in operational standby.")

    def evaluate(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        # As explicitly mandated: DO NOT generate fake detections if required model is unavailable.
        if not self.is_configured or self.seatbelt_model is None:
            # Operational status logged/reflected in context
            context["seatbelt_status"] = "Seatbelt detection model not configured"
            return []

        # If custom model is configured, run inference on car windshield crops
        cars = [v for v in context.get("tracked_vehicles", []) if v.class_name.lower() in {"car", "truck", "bus"}]
        frame = context.get("frame")
        violations = []

        if frame is None:
            return violations

        for car in cars:
            x1, y1, x2, y2 = [int(v) for v in car.bbox]
            h, w = frame.shape[:2]
            crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            if crop.size == 0:
                continue

            try:
                res = self.seatbelt_model(crop, verbose=False)
                is_no_seatbelt = False
                conf = 0.80
                if res and len(res[0].boxes) > 0:
                    for b in res[0].boxes:
                        cls_name = self.seatbelt_model.names.get(int(b.cls[0].item()), "")
                        if "no_seatbelt" in cls_name.lower():
                            is_no_seatbelt = True
                            conf = float(b.conf[0].item())
                            break

                if self.record_frame_evidence(car.track_id, is_no_seatbelt):
                    violations.append({
                        "violation_type": "NO_SEATBELT",
                        "track_id": car.track_id,
                        "confidence": round(conf, 3),
                        "timestamp": time.time(),
                        "evidence_required": True,
                        "extra": {"car_track_id": car.track_id},
                    })
            except Exception as e:
                logger.debug(f"Seatbelt evaluation error: {e}")

        return violations
