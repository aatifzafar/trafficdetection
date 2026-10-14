import logging
from typing import List, Dict, Any
import numpy as np
from vision.detector.detector import DetectionResult
from vision.tracker.base_tracker import BaseTracker, TrackedObject
from vision.tracker.byte_tracker import compute_iou

logger = logging.getLogger(__name__)


class DeepSORTTracker(BaseTracker):
    """
    DeepSORT Tracker representation.
    Employs Kalman-filter-style centroid velocity propagation and distance metric matching.
    """

    def __init__(self, max_cosine_distance: float = 0.2, nn_budget: int = 100, max_age: int = 30):
        self.max_cosine_distance = max_cosine_distance
        self.nn_budget = nn_budget
        self.max_age = max_age
        self.tracked_objects: Dict[int, TrackedObject] = {}
        self.next_track_id = 1
        logger.info("Initialized DeepSORTTracker.")

    def update(
        self,
        frame: np.ndarray,
        detections: List[DetectionResult],
        frame_idx: int,
    ) -> List[TrackedObject]:
        active_tracks: List[TrackedObject] = []
        unmatched_dets = list(detections)

        # Distance & IoU matching
        for tid, obj in list(self.tracked_objects.items()):
            if frame_idx - obj.last_frame > self.max_age:
                continue

            best_score = float("inf")
            best_det_idx = -1

            for idx, det in enumerate(unmatched_dets):
                # Combined metric: centroid distance + IoU penalty
                c1 = obj.center
                c2 = det.center
                dist = np.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)
                iou = compute_iou(obj.bbox, det.bbox)

                # Cost metric
                cost = dist * (1.0 - 0.5 * iou)
                if cost < best_score:
                    best_score = cost
                    best_det_idx = idx

            if best_score < 150.0 and best_det_idx != -1:
                matched_det = unmatched_dets.pop(best_det_idx)
                obj.update(matched_det.bbox, matched_det.confidence, frame_idx)
                obj.class_name = matched_det.class_name
                obj.class_id = matched_det.class_id
                active_tracks.append(obj)

        for det in unmatched_dets:
            if det.confidence >= 0.35:
                new_id = self.next_track_id
                self.next_track_id += 1
                new_obj = TrackedObject(
                    track_id=new_id,
                    class_name=det.class_name,
                    class_id=det.class_id,
                    bbox=det.bbox,
                    confidence=det.confidence,
                    frame_idx=frame_idx,
                )
                self.tracked_objects[new_id] = new_obj
                active_tracks.append(new_obj)

        return active_tracks

    def reset(self):
        self.tracked_objects.clear()
        self.next_track_id = 1
