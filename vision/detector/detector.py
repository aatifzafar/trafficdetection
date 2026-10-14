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


class TrafficDetector:
    """
    Traffic object detector supporting PyTorch GPU/CPU inference,
    custom trained models, and graceful prototype fallback.
    """

    def __init__(
        self,
        model_traffic_path: str = "models/yolov8n.pt",
        model_helmet_path: Optional[str] = "models/helmet_model.pt",
        model_plate_path: Optional[str] = "models/plate_model.pt",
        confidence: float = 0.35,
        iou: float = 0.45,
        device: str = "auto",
    ):
        self.confidence = confidence
        self.iou = iou
        self.device = self._resolve_device(device)

        # Ensure models directory exists
        os.makedirs("models", exist_ok=True)

        logger.info(f"Initializing TrafficDetector on device: {self.device}")

        # Primary Traffic Model (Vehicles, Riders, Traffic Lights, Cell Phones)
        # Ultralytics downloads yolov8n.pt if not present locally
        self.model_traffic = YOLO(model_traffic_path)

        # Custom Helmet Model check
        self.helmet_model = None
        self.has_custom_helmet_model = False
        if model_helmet_path and Path(model_helmet_path).exists():
            try:
                self.helmet_model = YOLO(model_helmet_path)
                self.has_custom_helmet_model = True
                logger.info(f"Loaded custom helmet model from {model_helmet_path}")
            except Exception as e:
                logger.warning(f"Failed to load custom helmet model: {e}")
        else:
            logger.info("Custom helmet model not found. Operating in Baseline/Prototype Mode.")

        # Custom License Plate Model check
        self.plate_model = None
        self.has_custom_plate_model = False
        if model_plate_path and Path(model_plate_path).exists():
            try:
                self.plate_model = YOLO(model_plate_path)
                self.has_custom_plate_model = True
                logger.info(f"Loaded custom plate model from {model_plate_path}")
            except Exception as e:
                logger.warning(f"Failed to load custom plate model: {e}")
        else:
            logger.info("Custom plate model not found. Plate crops will utilize vehicle candidate regions.")

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
        """
        results = self.model_traffic(
            frame,
            conf=self.confidence,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )

        detections: List[DetectionResult] = []
        if not results:
            return detections

        res = results[0]
        boxes = res.boxes

        if boxes is None or len(boxes) == 0:
            return detections

        names = self.model_traffic.names

        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            cls_name = names.get(cls_id, str(cls_id))

            detections.append(
                DetectionResult(
                    bbox=xyxy.tolist(),
                    confidence=conf,
                    class_id=cls_id,
                    class_name=cls_name,
                )
            )

        return detections
