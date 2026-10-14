from vision.tracker.base_tracker import BaseTracker, TrackedObject
from vision.tracker.byte_tracker import ByteTrackTracker
from vision.tracker.deepsort_tracker import DeepSORTTracker


def create_tracker(algorithm: str = "bytetrack", **kwargs) -> BaseTracker:
    algo = algorithm.lower().strip()
    if algo == "deepsort":
        return DeepSORTTracker(**kwargs)
    else:
        return ByteTrackTracker(**kwargs)


__all__ = ["BaseTracker", "TrackedObject", "ByteTrackTracker", "DeepSORTTracker", "create_tracker"]
