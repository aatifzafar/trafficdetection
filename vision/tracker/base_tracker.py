from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np
from vision.detector.detector import DetectionResult


class TrackedObject:
    def __init__(
        self,
        track_id: int,
        class_name: str,
        class_id: int,
        bbox: List[float],
        confidence: float,
        frame_idx: int = 0,
    ):
        self.track_id = track_id
        self.class_name = class_name
        self.class_id = class_id
        self.bbox = [float(x) for x in bbox]
        self.confidence = float(confidence)
        self.frame_idx = frame_idx
        self.trajectory: List[List[float]] = [self.center]  # List of [cx, cy]
        self.total_frames = 1
        self.first_frame = frame_idx
        self.last_frame = frame_idx

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
    def center(self) -> List[float]:
        return [(self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0]

    def update(self, bbox: List[float], confidence: float, frame_idx: int):
        self.bbox = [float(x) for x in bbox]
        self.confidence = float(confidence)
        self.trajectory.append(self.center)
        if len(self.trajectory) > 60:
            self.trajectory.pop(0)
        self.total_frames += 1
        self.last_frame = frame_idx

    def get_velocity_vector(self, window: int = 5) -> List[float]:
        """Calculates displacement vector [dx, dy] over recent window frames."""
        if len(self.trajectory) < 2:
            return [0.0, 0.0]
        pts = self.trajectory[-min(window, len(self.trajectory)):]
        dx = pts[-1][0] - pts[0][0]
        dy = pts[-1][1] - pts[0][1]
        return [dx, dy]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "class_name": self.class_name,
            "class_id": self.class_id,
            "bbox": self.bbox,
            "confidence": round(self.confidence, 3),
            "frame_idx": self.frame_idx,
            "center": self.center,
            "total_frames": self.total_frames,
        }


class BaseTracker(ABC):
    """Abstract interface for Multi-Object Trackers."""

    @abstractmethod
    def update(
        self,
        frame: np.ndarray,
        detections: List[DetectionResult],
        frame_idx: int,
    ) -> List[TrackedObject]:
        """
        Takes raw frame and detection bounding boxes, updates tracking states,
        and returns active tracked objects with persistent IDs.
        """
        pass

    @abstractmethod
    def reset(self):
        """Clears all tracking histories."""
        pass
