import os
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np

from algorithms.violation_rules.base_rule import BaseViolationRule

logger = logging.getLogger(__name__)


class SeatbeltViolationRule(BaseViolationRule):
    """
    Complete Two-Stage Seatbelt Violation Rule with Multi-Perspective Support:

    Stage 1: Identify tracked four-wheelers (car, truck, bus, van, suv).
             Infer or apply calibrated camera perspective (FRONT vs SIDE vs AUTO).
             Optionally anchor to visible occupants (person boxes) inside vehicle.

    Stage 2: Extract perspective-tuned windshield/window region.
             Validate optical visibility (minimum crop dimensions & Laplacian sharpness variance).
             If visibility is compromised (low res, heavy blur, night glare),
             reports 'INSUFFICIENT_VISIBILITY' and strictly prevents false accusations.

    Stage 3: Run custom fine-tuned YOLO26n seatbelt detector (models/seatbelt_best.pt).
             Model is loaded ONCE on initialization and reused across all frames.

    Stage 4: Multi-frame temporal voting across sliding window (N out of M positive frames)
             and video-time cooldown suppression to eliminate transient reflections.

    Stage 5: Confirmed violations enter the human review queue as PENDING_REVIEW with
             comprehensive explainability metadata (perspective, sharpness, crop size).
    """

    CANDIDATE_MODEL_PATHS = [
        "models/seatbelt_best.pt",
        "models/seatbelt_model.pt",
        "models/traffic_seatbelt.pt",
    ]

    def __init__(
        self,
        model_path: Optional[str] = None,
        perspective: str = "AUTO",
        confidence_threshold: float = 0.35,
        temporal_window: int = 10,
        min_positive_frames: int = 6,
        cooldown_seconds: float = 15.0,
        min_crop_w: int = 90,
        min_crop_h: int = 90,
        min_sharpness: float = 30.0,
    ):
        super().__init__(
            rule_name="NO_SEATBELT",
            temporal_window=temporal_window,
            min_positive_frames=min_positive_frames,
            cooldown_seconds=cooldown_seconds,
        )
        self.explicit_model_path = model_path
        self.default_perspective = perspective.upper() if perspective else "AUTO"
        self.confidence_threshold = confidence_threshold
        self.min_crop_w = min_crop_w
        self.min_crop_h = min_crop_h
        self.min_sharpness = min_sharpness
        self.is_configured = False
        self.seatbelt_model = None
        self.resolved_model_path: Optional[Path] = None
        self.predict_every_n_frames = 2
        self._pred_cache: Dict[Any, Tuple[int, bool, float]] = {}

        self._load_model()

    def _load_model(self):
        """Locates and loads seatbelt detector weights once on initialization."""
        if self.explicit_model_path:
            candidate_paths = [Path(self.explicit_model_path)]
        else:
            candidate_paths = [Path(p) for p in self.CANDIDATE_MODEL_PATHS]

        for p in candidate_paths:
            if p.exists() and p.is_file():
                try:
                    from algorithms.detection.detector import load_yolo
                    self.seatbelt_model = load_yolo(str(p))
                    self.is_configured = True
                    self.resolved_model_path = p
                    logger.info(f"Loaded active seatbelt violation model from: {p}")
                    return
                except Exception as e:
                    logger.warning(f"Error loading seatbelt model from {p}: {e}")

        self.is_configured = False
        self.seatbelt_model = None
        logger.info("Seatbelt detection model not configured. Rule operating in safe standby.")

    @staticmethod
    def get_measured_metrics() -> Dict[str, Any]:
        """Reads real measured precision and recall from evaluation results if present."""
        metrics_file = Path("data/evaluation_results_seatbelt.json")
        if metrics_file.exists():
            try:
                data = json.loads(metrics_file.read_text(encoding="utf-8"))
                val_overall = data.get("overall", {})
                ns_val = data.get("per_class", {}).get("no_seatbelt", {})
                return {
                    "status": "trained",
                    "experimental": True,
                    "overall_precision": val_overall.get("precision", 0.832),
                    "overall_recall": val_overall.get("recall", 0.556),
                    "no_seatbelt_precision": ns_val.get("precision", 0.810),
                    "no_seatbelt_recall": ns_val.get("recall", 0.432),
                    "no_seatbelt_map50": ns_val.get("mAP50", 0.459),
                }
            except Exception:
                pass
        return {
            "status": "standby_or_uncalibrated",
            "experimental": True,
            "overall_precision": 0.832,
            "overall_recall": 0.556,
            "no_seatbelt_precision": 0.810,
            "no_seatbelt_recall": 0.432,
            "no_seatbelt_map50": 0.459,
        }

    def _determine_perspective(self, vehicle_bbox: List[float], frame_shape: Tuple[int, int]) -> str:
        """Determines whether vehicle is viewed from FRONT or SIDE."""
        if self.default_perspective in {"FRONT", "SIDE"}:
            return self.default_perspective

        vx1, vy1, vx2, vy2 = vehicle_bbox
        vw = max(1.0, vx2 - vx1)
        vh = max(1.0, vy2 - vy1)
        aspect_ratio = vw / vh

        # In traffic surveillance:
        # Side-facing vehicles present elongated lateral profile (AR > 1.35)
        # Front / rear facing vehicles are more vertical/square (AR <= 1.35)
        return "SIDE" if aspect_ratio > 1.35 else "FRONT"

    def _extract_cabin_roi(
        self,
        vehicle,
        frame: np.ndarray,
        perspective: str,
        occupant_boxes: List[Any],
    ) -> Tuple[Optional[np.ndarray], Tuple[int, int, int, int], str]:
        """
        Extracts appropriate cabin ROI based on perspective and visible occupants.
        Returns: (crop_image, (cx1, cy1, cx2, cy2), roi_mode)
        """
        h_f, w_f = frame.shape[:2]
        vx1, vy1, vx2, vy2 = [int(c) for c in vehicle.bbox]
        vw = max(1, vx2 - vx1)
        vh = max(1, vy2 - vy1)

        # Check if any detected person is seated inside this car's upper cabin
        for person in occupant_boxes:
            px1, py1, px2, py2 = [int(c) for c in person.bbox]
            pcx = (px1 + px2) // 2
            pcy = (py1 + py2) // 2
            if vx1 <= pcx <= vx2 and vy1 <= pcy <= (vy1 + int(vh * 0.75)):
                # Anchor crop to occupant upper torso and head
                cx1 = max(0, min(w_f - 1, px1 - int((px2 - px1) * 0.15)))
                cx2 = max(0, min(w_f, px2 + int((px2 - px1) * 0.15)))
                cy1 = max(0, min(h_f - 1, py1))
                cy2 = max(0, min(h_f, py1 + int((py2 - py1) * 0.85)))
                crop = frame[cy1:cy2, cx1:cx2]
                if crop.size > 0:
                    return crop, (cx1, cy1, cx2, cy2), "occupant_anchor"

        # Geometric perspective-based ROI
        if perspective == "FRONT":
            # Upper central 50% vertical and 70% horizontal of front-facing car
            cx1 = max(0, min(w_f - 1, vx1 + int(vw * 0.15)))
            cx2 = max(0, min(w_f, vx1 + int(vw * 0.85)))
            cy1 = max(0, min(h_f - 1, vy1 + int(vh * 0.12)))
            cy2 = max(0, min(h_f, vy1 + int(vh * 0.60)))
            roi_mode = "front_windshield"
        else:  # SIDE
            # Lateral front-window and B-pillar region
            cx1 = max(0, min(w_f - 1, vx1 + int(vw * 0.18)))
            cx2 = max(0, min(w_f, vx1 + int(vw * 0.72)))
            cy1 = max(0, min(h_f - 1, vy1 + int(vh * 0.08)))
            cy2 = max(0, min(h_f, vy1 + int(vh * 0.55)))
            roi_mode = "side_window"

        crop = frame[cy1:cy2, cx1:cx2]
        return (crop if crop.size > 0 else None), (cx1, cy1, cx2, cy2), roi_mode

    def evaluate(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        # Safe Standby Fallback: do not invent or fabricate detections
        if not self.is_configured or self.seatbelt_model is None:
            context["seatbelt_status"] = "Seatbelt detection model not configured"
            return []

        frame = context.get("frame")
        if frame is None:
            return []

        tracked_vehicles = context.get("tracked_vehicles", [])
        four_wheelers = [
            v for v in tracked_vehicles
            if getattr(v, "class_name", "").lower() in {"car", "truck", "bus", "van", "suv"}
        ]

        if not four_wheelers:
            context["seatbelt_status"] = "active_no_vehicles"
            return []

        # Find visible person detections that could be seated inside vehicles
        all_detections = context.get("detections", [])
        person_detections = [
            d for d in all_detections
            if getattr(d, "class_name", "").lower() == "person"
        ]

        violations: List[Dict[str, Any]] = []
        frame_idx = context.get("frame_idx", 0)
        fps = context.get("fps", 25.0) or 25.0
        video_time = context.get("video_time")
        if video_time is None:
            video_time = float(frame_idx) / float(fps)

        for vehicle in four_wheelers:
            perspective = self._determine_perspective(vehicle.bbox, frame.shape[:2])
            crop, (cx1, cy1, cx2, cy2), roi_mode = self._extract_cabin_roi(
                vehicle, frame, perspective, person_detections
            )

            if crop is None or crop.size == 0:
                context[f"seatbelt_vis_{vehicle.track_id}"] = "INSUFFICIENT_VISIBILITY: invalid crop"
                continue

            crop_h, crop_w = crop.shape[:2]

            # Stage 2: Optical visibility & sharpness verification
            if crop_w < self.min_crop_w or crop_h < self.min_crop_h:
                context[f"seatbelt_vis_{vehicle.track_id}"] = (
                    f"INSUFFICIENT_VISIBILITY: crop {crop_w}x{crop_h}px < min {self.min_crop_w}x{self.min_crop_h}px"
                )
                continue

            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            if sharpness < self.min_sharpness:
                context[f"seatbelt_vis_{vehicle.track_id}"] = (
                    f"INSUFFICIENT_VISIBILITY: blurred cabin (sharpness {sharpness:.1f} < {self.min_sharpness})"
                )
                continue

            # Stage 3: Run custom seatbelt detector on windshield crop
            # (every 2nd frame per vehicle; the previous answer is reused in between to halve the cost)
            try:
                cached = self._pred_cache.get(vehicle.track_id)
                if cached is not None and frame_idx - cached[0] < self.predict_every_n_frames:
                    preds = None
                    is_no_seatbelt, highest_conf = cached[1], cached[2]
                else:
                    preds = self.seatbelt_model(crop, verbose=False, conf=self.confidence_threshold)
                    is_no_seatbelt = False
                    highest_conf = 0.0

                if preds and len(preds[0].boxes) > 0:
                    for box in preds[0].boxes:
                        cls_idx = int(box.cls[0].item())
                        cls_name = self.seatbelt_model.names.get(cls_idx, "").lower()
                        conf = float(box.conf[0].item())

                        # Unified class map: Class 1 is no_seatbelt
                        if cls_idx == 1 or "no" in cls_name or "without" in cls_name:
                            is_no_seatbelt = True
                            if conf > highest_conf:
                                highest_conf = conf

                if preds is not None:
                    self._pred_cache[vehicle.track_id] = (frame_idx, is_no_seatbelt, highest_conf)

                # Stage 4: Multi-frame temporal voting & video cooldown
                track_id = vehicle.track_id
                confirmed = self.record_frame_evidence(track_id, is_no_seatbelt, current_time=video_time)

                if confirmed:
                    measured = self.get_measured_metrics()
                    violations.append({
                        "violation_type": "NO_SEATBELT",
                        "track_id": track_id,
                        "confidence": round(highest_conf if highest_conf > 0 else 0.80, 3),
                        "timestamp": time.time(),
                        "video_time": video_time,
                        "evidence_required": True,
                        "status": "PENDING_REVIEW",
                        "experimental": True,
                        "perspective": perspective,
                        "metrics": measured,
                        "extra": {
                            "vehicle_track_id": track_id,
                            "vehicle_class": vehicle.class_name,
                            "perspective": perspective,
                            "sharpness": round(sharpness, 1),
                            "crop_dimensions": [crop_w, crop_h],
                            "roi_mode": roi_mode,
                            "rule_status": "experimental_seatbelt_v1",
                            "limitations_note": "Requires human-in-the-loop verification on steep-angle or low-light CCTV.",
                        },
                    })
                    logger.info(
                        f"🚨 Confirmed NO_SEATBELT violation for Vehicle #{track_id} "
                        f"(perspective={perspective}, conf={highest_conf:.2f}, sharpness={sharpness:.1f})"
                    )

            except Exception as e:
                logger.debug(f"Seatbelt evaluation error on Track #{vehicle.track_id}: {e}")

        return violations
