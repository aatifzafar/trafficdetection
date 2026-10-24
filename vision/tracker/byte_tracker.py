import logging
from typing import List, Dict, Any, Optional
import numpy as np
from types import SimpleNamespace
from vision.detector.detector import DetectionResult
from vision.tracker.base_tracker import BaseTracker, TrackedObject

logger = logging.getLogger(__name__)


def compute_iou(boxA: List[float], boxB: List[float]) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interWidth = max(0.0, xB - xA)
    interHeight = max(0.0, yB - yA)
    interArea = interWidth * interHeight

    boxAArea = max(1e-5, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = max(1e-5, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    return interArea / float(boxAArea + boxBArea - interArea)


class ByteTrackTracker(BaseTracker):
    """
    ByteTrack Tracker implementation.
    Uses Ultralytics BYTETracker engine with smooth fallback to robust IoU-Centroid matching.
    """

    def __init__(
        self,
        track_thresh: float = 0.35,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        frame_rate: int = 30,
    ):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.frame_rate = frame_rate

        self.tracked_objects: Dict[int, TrackedObject] = {}
        self.next_track_id = 1
        self.ultralytics_tracker = None

        try:
            from ultralytics.trackers.byte_tracker import BYTETracker

            args = SimpleNamespace(
                track_high_thresh=track_thresh,
                track_low_thresh=max(0.05, track_thresh * 0.4),
                new_track_thresh=min(0.9, track_thresh + 0.05),
                track_buffer=track_buffer,
                match_thresh=match_thresh,
                fuse_score=False,
                fps=frame_rate,
            )
            self.ultralytics_tracker = BYTETracker(args)
            logger.info("Initialized Ultralytics BYTETracker successfully.")
        except Exception as e:
            logger.warning(f"Could not initialize Ultralytics BYTETracker: {e}. Using high-precision IoU tracker.")

    def update(
        self,
        frame: np.ndarray,
        detections: List[DetectionResult],
        frame_idx: int,
    ) -> List[TrackedObject]:
        if not detections:
            return []

        # If Ultralytics tracker is available, convert detections
        if self.ultralytics_tracker is not None:
            try:
                import torch
                from ultralytics.engine.results import Boxes

                det_array = []
                for d in detections:
                    det_array.append([d.x1, d.y1, d.x2, d.y2, d.confidence, float(d.class_id)])

                h, w = frame.shape[:2] if frame is not None else (720, 1280)
                det_tensor = torch.tensor(det_array, dtype=torch.float32)
                boxes = Boxes(det_tensor, orig_shape=(h, w))

                tracks = self.ultralytics_tracker.update(boxes.cpu().numpy(), img=frame)

                active_tracks: List[TrackedObject] = []
                if tracks is not None and len(tracks) > 0:
                    for row in tracks:
                        x1, y1, x2, y2, tid, score, cls_id, det_idx = row[:8]
                        tid = int(tid)
                        score = float(score)
                        cls_id = int(cls_id)
                        det_idx = int(det_idx)

                        cls_name = "vehicle"
                        if 0 <= det_idx < len(detections):
                            cls_name = detections[det_idx].class_name
                        else:
                            for d in detections:
                                if compute_iou([x1, y1, x2, y2], d.bbox) > 0.4:
                                    cls_name = d.class_name
                                    break

                        if tid in self.tracked_objects:
                            obj = self.tracked_objects[tid]
                            obj.update([float(x1), float(y1), float(x2), float(y2)], score, frame_idx)
                        else:
                            obj = TrackedObject(
                                track_id=tid,
                                class_name=cls_name,
                                class_id=cls_id,
                                bbox=[float(x1), float(y1), float(x2), float(y2)],
                                confidence=score,
                                frame_idx=frame_idx,
                            )
                            self.tracked_objects[tid] = obj

                        active_tracks.append(obj)

                return active_tracks
            except Exception as e:
                logger.warning(f"ByteTracker engine step failed: {e}. Falling back to internal IoU matching.")

        # Fallback IoU tracker
        if self.ultralytics_tracker is None:
            logger.warning("Ultralytics tracker unavailable. Using internal IoU tracker.")
        return self._fallback_iou_update(detections, frame_idx)

    def _fallback_iou_update(
        self, detections: List[DetectionResult], frame_idx: int
    ) -> List[TrackedObject]:
        active_tracks: List[TrackedObject] = []
        unmatched_dets = list(detections)

        # Match existing tracks with best IoU
        for tid, obj in list(self.tracked_objects.items()):
            if frame_idx - obj.last_frame > self.track_buffer:
                continue

            best_iou = 0.0
            best_det_idx = -1

            for idx, det in enumerate(unmatched_dets):
                iou = compute_iou(obj.bbox, det.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = idx

            if best_iou >= 0.25 and best_det_idx != -1:
                matched_det = unmatched_dets.pop(best_det_idx)
                obj.update(matched_det.bbox, matched_det.confidence, frame_idx)
                obj.class_name = matched_det.class_name
                obj.class_id = matched_det.class_id
                active_tracks.append(obj)

        # Create new tracks for remaining detections
        for det in unmatched_dets:
            if det.confidence >= self.track_thresh:
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
