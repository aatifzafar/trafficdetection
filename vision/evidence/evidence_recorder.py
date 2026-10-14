import os
import json
import logging
import datetime
from collections import deque
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class PendingEvidenceClip:
    def __init__(
        self,
        event_id: str,
        event_folder: Path,
        trigger_frame_idx: int,
        target_end_frame_idx: int,
        fps: float,
        frame_shape: Tuple[int, int],
        pre_frames: List[np.ndarray],
    ):
        self.event_id = event_id
        self.event_folder = event_folder
        self.trigger_frame_idx = trigger_frame_idx
        self.target_end_frame_idx = target_end_frame_idx
        self.fps = fps
        self.frame_shape = frame_shape  # (h, w)
        self.collected_frames: List[np.ndarray] = list(pre_frames)
        self.is_completed = False

    def add_frame(self, frame: np.ndarray, frame_idx: int):
        if not self.is_completed:
            self.collected_frames.append(frame.copy())
            if frame_idx >= self.target_end_frame_idx:
                self.finalize_video()

    def finalize_video(self):
        if self.is_completed:
            return
        video_path = self.event_folder / "evidence.mp4"
        try:
            h, w = self.frame_shape
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(video_path), fourcc, max(10.0, self.fps), (w, h))
            for f in self.collected_frames:
                writer.write(f)
            writer.release()
            logger.info(f"Finalized evidence video clip: {video_path} ({len(self.collected_frames)} frames)")
        except Exception as e:
            logger.error(f"Failed to write evidence video {video_path}: {e}")
        finally:
            self.is_completed = True


class EvidenceRecorder:
    """
    Manages circular frame buffer and persists multi-modal violation evidence
    (annotated high-res frame, pre/post video clip, metadata.json).
    """

    def __init__(
        self,
        base_dir: str = "data/evidence",
        pre_seconds: int = 4,
        post_seconds: int = 4,
        fps_fallback: float = 25.0,
    ):
        self.base_dir = Path(base_dir)
        self.pre_seconds = pre_seconds
        self.post_seconds = post_seconds
        self.fps_fallback = fps_fallback
        self.buffer_size = int((pre_seconds + post_seconds + 2) * fps_fallback)
        self.frame_buffer: deque = deque(maxlen=self.buffer_size)  # [(frame_idx, frame)]
        self.pending_clips: List[PendingEvidenceClip] = []
        os.makedirs(self.base_dir, exist_ok=True)

    def push_frame(self, frame: np.ndarray, frame_idx: int):
        """Pushes current frame into the sliding memory buffer."""
        # Store a copy or reference in buffer
        self.frame_buffer.append((frame_idx, frame.copy()))

        # Update any pending evidence clips collecting post-frames
        active_clips = []
        for clip in self.pending_clips:
            clip.add_frame(frame, frame_idx)
            if not clip.is_completed:
                active_clips.append(clip)
        self.pending_clips = active_clips

    def record_violation(
        self,
        event_id: str,
        violation_type: str,
        track_id: Optional[int],
        plate_number: Optional[str],
        confidence: float,
        location: str,
        camera_id: Optional[int],
        current_frame: np.ndarray,
        frame_idx: int,
        fps: Optional[float] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """
        Generates folder structure: data/evidence/YYYY-MM-DD/EVT-XXXX/
        Saves violation_frame.jpg, metadata.json, and starts evidence.mp4 clip assembly.
        """
        actual_fps = fps if (fps and fps > 1.0) else self.fps_fallback
        today_str = datetime.datetime.utcnow().strftime("%Y-%m-%d")
        event_folder = self.base_dir / today_str / event_id
        event_folder.mkdir(parents=True, exist_ok=True)

        # 1. Save violation image frame
        frame_filename = "violation_frame.jpg"
        frame_path = event_folder / frame_filename
        cv2.imwrite(str(frame_path), current_frame)

        # 2. Save metadata.json
        meta = {
            "event_id": event_id,
            "violation_type": violation_type,
            "track_id": track_id,
            "plate_number": plate_number or "UNIDENTIFIED",
            "confidence": round(confidence, 3),
            "location": location,
            "camera_id": camera_id,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "frame_idx": frame_idx,
            "fps": actual_fps,
            "extra": extra_metadata or {},
        }
        meta_path = event_folder / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        # 3. Collect pre-violation frames from rolling buffer
        pre_frames_count = int(self.pre_seconds * actual_fps)
        buffer_frames = list(self.frame_buffer)
        selected_pre_frames = [
            f for (f_idx, f) in buffer_frames if f_idx <= frame_idx
        ][-pre_frames_count:]
        if not selected_pre_frames:
            selected_pre_frames = [current_frame.copy()]

        post_frames_count = int(self.post_seconds * actual_fps)
        target_end_frame = frame_idx + post_frames_count

        h, w = current_frame.shape[:2]
        pending_clip = PendingEvidenceClip(
            event_id=event_id,
            event_folder=event_folder,
            trigger_frame_idx=frame_idx,
            target_end_frame_idx=target_end_frame,
            fps=actual_fps,
            frame_shape=(h, w),
            pre_frames=selected_pre_frames,
        )
        self.pending_clips.append(pending_clip)

        logger.info(f"Generated violation evidence in {event_folder}")
        return {
            "folder": str(event_folder),
            "frame_path": str(frame_path),
            "video_path": str(event_folder / "evidence.mp4"),
            "metadata_path": str(meta_path),
        }

    def flush(self):
        """Finalizes any pending clips immediately."""
        for clip in self.pending_clips:
            clip.finalize_video()
        self.pending_clips.clear()
