import time
import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import torch
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class DetectionResult:
    def __init__(
        self,
        bbox: List[float],  # [x1, y1, x2, y2]
        confidence: float,
        class_id: int,
        class_name: str,
        track_id: Optional[int] = None,
    ):
        self.bbox = [float(x) for x in bbox]
        self.confidence = float(confidence)
        self.class_id = int(class_id)
        self.class_name = class_name
        self.track_id = int(track_id) if track_id is not None else None

    @property
    def x1(self) -> float:
        return self.bbox[0]

    @property
    def y1(self) -> float:
        return self.bbox[1]

    @property
    def x2(self) -> float:
        return self.bbox[2]

    @property
    def y2(self) -> float:
        return self.bbox[3]

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bbox": self.bbox,
            "confidence": round(self.confidence, 3),
            "class_id": self.class_id,
            "class_name": self.class_name,
            "track_id": self.track_id,
        }


TRIPLE_MODEL_KEEP = {
    "MORE_THAN_TWO_PERSONS",
    "USING_MOBILE",
    "WITHOUT_HELMET",
    "WITH_HELMET",
    "NO_HELMET",
    "HELMET",
    "TRIPLE_RIDING",
}


class TrafficDetector:
    """
    Traffic object detector supporting PyTorch GPU/CPU inference,
    custom trained models, and graceful prototype fallback.
    YOLO strictly handles vehicle and object localization, helmet compliance,
    and rider count. Kinematic violations (Wrong-Way, Speed) are handled
    separately via tracking and trajectory analysis.
    """

    def __init__(
        self,
        model_traffic_path: str = "models/yolo26n.pt",
        model_custom_path: Optional[str] = "models/traffic_custom.pt",
        model_helmet_path: Optional[str] = "models/helmet_model.pt",
        model_plate_path: Optional[str] = "models/plate_model.pt",
        model_triple_path: Optional[str] = "models/triple_riding_best.pt",
        confidence: float = 0.35,
        iou: float = 0.45,
        device: str = "auto",
    ):
        self.confidence = confidence
        self.iou = iou
        # running statistics of the vehicle/person model, used to compare YOLO26n and YOLOv8n
        self.stats = {"calls": 0, "ms": 0.0, "boxes": 0, "conf_sum": 0.0}
        self.device = self._resolve_device(device)

        # Ensure models directory exists
        os.makedirs("models", exist_ok=True)

        logger.info(f"Initializing TrafficDetector on device: {self.device}")

        # Primary Traffic Model (Vehicles, Riders, Traffic Lights, Cell Phones)
        self.model_traffic = YOLO(model_traffic_path)

        # Unified Custom Model check (classes: 0 helmet, 1 no_helmet, 2 number_plate)
        self.model_custom = None
        self.has_custom_model = False
        self.custom_imgsz = None
        if model_custom_path and Path(model_custom_path).exists():
            try:
                self.model_custom = YOLO(model_custom_path)
                self.has_custom_model = True
                self.custom_imgsz = getattr(self.model_custom, "overrides", {}).get("imgsz", None)
                if not self.custom_imgsz or "832" in str(model_custom_path) or "custom" in str(model_custom_path):
                    self.custom_imgsz = 832
                logger.info(f"Loaded unified custom model from {model_custom_path} (trained imgsz: {self.custom_imgsz})")
            except Exception as e:
                logger.warning(f"Failed to load custom model from {model_custom_path}: {e}")
        else:
            logger.info("Custom traffic model not found (models/traffic_custom.pt). Operating in Baseline Heuristic Mode.")

        # Second custom model (team's triple-riding model). Only its triple-riding and phone classes are used;
        # helmet / no-helmet / number plate always come from traffic_custom.pt (better accuracy there).
        self.model_triple = None
        self.triple_imgsz = None
        if model_triple_path and Path(model_triple_path).exists():
            try:
                self.model_triple = YOLO(model_triple_path)
                self.triple_imgsz = getattr(self.model_triple, "overrides", {}).get("imgsz", None)
                logger.info(f"Loaded triple-riding model from {model_triple_path}")
            except Exception as e:
                logger.warning(f"Failed to load triple-riding model from {model_triple_path}: {e}")

        # Legacy custom models support / aliases for backwards compatibility
        self.helmet_model = self.model_custom
        self.has_custom_helmet_model = self.has_custom_model or (self.model_triple is not None)
        self.plate_model = self.model_custom
        self.has_custom_plate_model = self.has_custom_model

    def _resolve_device(self, req_device: str) -> str:
        if req_device == "cuda":
            return "cuda" if torch.cuda.is_available() else "cpu"
        elif req_device == "cpu":
            return "cpu"
        else:  # auto
            return "cuda" if torch.cuda.is_available() else "cpu"

    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        """
        Runs object detection on a single RGB/BGR frame.
        Executes COCO model for vehicles/persons and optional custom model once for helmet/no_helmet/plate.
        """
        t0 = time.perf_counter()
        results = self.model_traffic(
            frame,
            conf=self.confidence,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )
        self.stats["calls"] += 1
        self.stats["ms"] += (time.perf_counter() - t0) * 1000.0

        detections: List[DetectionResult] = []
        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            names = self.model_traffic.names

            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = names.get(cls_id, str(cls_id))
                self.stats["boxes"] += 1
                self.stats["conf_sum"] += conf

                detections.append(
                    DetectionResult(
                        bbox=xyxy,
                        confidence=conf,
                        class_id=cls_id,
                        class_name=cls_name,
                    )
                )

        # Run custom model ONCE per frame on full frame if loaded
        if self.has_custom_model and self.model_custom is not None:
            try:
                kwargs = {
                    "conf": self.confidence,
                    "iou": self.iou,
                    "device": self.device,
                    "verbose": False,
                }
                if self.custom_imgsz:
                    kwargs["imgsz"] = self.custom_imgsz
                custom_results = self.model_custom(frame, **kwargs)
                if custom_results and len(custom_results) > 0 and custom_results[0].boxes is not None:
                    c_boxes = custom_results[0].boxes
                    c_names = getattr(self.model_custom, "names", {0: "helmet", 1: "no_helmet", 2: "number_plate"})
                    for box in c_boxes:
                        xyxy = box.xyxy[0].cpu().numpy().tolist()
                        conf = float(box.conf[0].cpu().numpy())
                        cls_id = int(box.cls[0].cpu().numpy())
                        if isinstance(c_names, dict):
                            cls_name = c_names.get(cls_id, str(cls_id))
                        elif isinstance(c_names, list) and cls_id < len(c_names):
                            cls_name = c_names[cls_id]
                        else:
                            cls_name = {0: "helmet", 1: "no_helmet", 2: "number_plate"}.get(cls_id, str(cls_id))

                        detections.append(
                            DetectionResult(
                                bbox=xyxy,
                                confidence=conf,
                                class_id=cls_id,
                                class_name=cls_name,
                            )
                        )
            except Exception as e:
                logger.warning(f"Custom model inference error: {e}")

        # Triple-riding model: keep only the classes the main custom model does not have
        if self.model_triple is not None:
            try:
                kwargs = {"conf": self.confidence, "iou": self.iou, "device": self.device, "verbose": False}
                if self.triple_imgsz:
                    kwargs["imgsz"] = self.triple_imgsz
                t_res = self.model_triple(frame, **kwargs)
                if t_res and t_res[0].boxes is not None:
                    t_names = getattr(self.model_triple, "names", {}) or {}
                    for box in t_res[0].boxes:
                        cls_id = int(box.cls[0].cpu().numpy())
                        cls_name = t_names.get(cls_id, str(cls_id)) if isinstance(t_names, dict) else str(cls_id)
                        cname_upper = str(cls_name).upper()
                        if cname_upper not in TRIPLE_MODEL_KEEP:
                            continue
                        norm_name = ('no_helmet' if cname_upper in {'WITHOUT_HELMET', 'NO_HELMET'} else 'helmet' if cname_upper in {'WITH_HELMET', 'HELMET'} else 'MORE_THAN_TWO_PERSONS' if cname_upper in {'MORE_THAN_TWO_PERSONS', 'TRIPLE_RIDING', 'TRIPLE_RIDER'} else 'USING_MOBILE' if cname_upper in {'USING_MOBILE', 'MOBILE_PHONE'} else cls_name)
                        detections.append(
                            DetectionResult(
                                bbox=box.xyxy[0].cpu().numpy().tolist(),
                                confidence=float(box.conf[0].cpu().numpy()),
                                class_id=100 + cls_id,
                                class_name=norm_name,
                            )
                        )
            except Exception as e:
                logger.warning(f"Triple-riding model inference error: {e}")

        return detections

