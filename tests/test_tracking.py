import pytest
import numpy as np
from vision.detector.detector import DetectionResult
from vision.tracker import create_tracker, TrackedObject
from vision.association.rider_association import RiderMotorcycleAssociator


def test_tracker_persistent_ids():
    tracker = create_tracker(algorithm="bytetrack")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Frame 1: Detection of a car
    dets1 = [
        DetectionResult(bbox=[100, 100, 200, 200], confidence=0.88, class_id=2, class_name="car")
    ]
    tracks1 = tracker.update(frame, dets1, frame_idx=1)
    assert len(tracks1) == 1
    t_id = tracks1[0].track_id

    # Frame 2: Car shifted slightly
    dets2 = [
        DetectionResult(bbox=[105, 102, 205, 202], confidence=0.91, class_id=2, class_name="car")
    ]
    tracks2 = tracker.update(frame, dets2, frame_idx=2)
    assert len(tracks2) == 1
    # Track ID must remain persistent!
    assert tracks2[0].track_id == t_id


def test_rider_motorcycle_association():
    associator = RiderMotorcycleAssociator(max_distance=150.0, min_overlap_ratio=0.1)

    # Motorcycle at [200, 300, 320, 480]
    bike = TrackedObject(track_id=1, class_name="motorcycle", class_id=3, bbox=[200, 300, 320, 480], confidence=0.9)

    # Rider seated directly on top of the bike: [220, 200, 300, 380]
    rider = TrackedObject(track_id=2, class_name="person", class_id=0, bbox=[220, 200, 300, 380], confidence=0.88)

    # Random pedestrian far away: [500, 100, 560, 250]
    pedestrian = TrackedObject(track_id=3, class_name="person", class_id=0, bbox=[500, 100, 560, 250], confidence=0.82)

    groups = associator.associate([bike, rider, pedestrian])
    assert len(groups) == 1
    group = groups[0]
    assert group.motorcycle.track_id == 1
    assert group.rider_count == 1
    assert group.riders[0].track_id == 2  # Only seated rider associated, pedestrian excluded!
