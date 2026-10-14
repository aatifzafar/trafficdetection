import math
from typing import List, Dict, Any, Tuple
from vision.tracker.base_tracker import TrackedObject


class MotorcycleRiderGroup:
    def __init__(self, motorcycle: TrackedObject):
        self.motorcycle = motorcycle
        self.riders: List[TrackedObject] = []

    def add_rider(self, rider: TrackedObject):
        self.riders.append(rider)

    @property
    def rider_count(self) -> int:
        return len(self.riders)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "motorcycle_id": self.motorcycle.track_id,
            "motorcycle_bbox": self.motorcycle.bbox,
            "rider_count": self.rider_count,
            "rider_ids": [r.track_id for r in self.riders],
        }


class RiderMotorcycleAssociator:
    """
    Explainable spatial association linking detected riders/persons to motorcycles.
    Prevents false counting of random pedestrians by verifying vertical alignment,
    horizontal span overlap, and proximity.
    """

    def __init__(
        self,
        max_distance: float = 120.0,
        min_overlap_ratio: float = 0.10,
    ):
        self.max_distance = max_distance
        self.min_overlap_ratio = min_overlap_ratio

    def associate(
        self,
        tracked_objects: List[TrackedObject],
    ) -> List[MotorcycleRiderGroup]:
        motorcycles: List[TrackedObject] = []
        persons: List[TrackedObject] = []

        two_wheeler_classes = {"motorcycle", "motorbike", "bicycle"}
        for obj in tracked_objects:
            cname = obj.class_name.lower()
            if cname in two_wheeler_classes:
                motorcycles.append(obj)
            elif cname == "person":
                persons.append(obj)

        groups: List[MotorcycleRiderGroup] = [
            MotorcycleRiderGroup(m) for m in motorcycles
        ]

        if not motorcycles or not persons:
            return groups

        # For each person, evaluate association score with all motorcycles
        for person in persons:
            best_group = None
            best_score = float("inf")

            px1, py1, px2, py2 = person.bbox
            pcx, pcy = person.center
            p_width = px2 - px1
            p_height = py2 - py1

            for group in groups:
                m = group.motorcycle
                mx1, my1, mx2, my2 = m.bbox
                mcx, mcy = m.center
                m_width = mx2 - mx1
                m_height = my2 - my1

                # 1. Horizontal constraint: Person's horizontal center must align with motorcycle
                horizontal_padding = 0.25 * m_width
                if not (mx1 - horizontal_padding <= pcx <= mx2 + horizontal_padding):
                    continue

                # 2. Vertical constraint: Rider sits above/overlapping the motorcycle
                # Rider head py1 must be above motorcycle center mcy
                if py1 > mcy:
                    continue
                # Rider feet/base py2 must reach down into the upper or middle section of the motorcycle
                if py2 < my1 - 0.2 * m_height:
                    continue

                # 3. Intersection over person area
                inter_x1 = max(px1, mx1)
                inter_y1 = max(py1, my1)
                inter_x2 = min(px2, mx2)
                inter_y2 = min(py2, my2)
                inter_w = max(0.0, inter_x2 - inter_x1)
                inter_h = max(0.0, inter_y2 - inter_y1)
                inter_area = inter_w * inter_h
                person_area = max(1.0, p_width * p_height)
                overlap_ratio = inter_area / person_area

                # 4. Center-to-center distance
                dist = math.sqrt((pcx - mcx) ** 2 + (pcy - mcy) ** 2)

                # Combined explainable score (lower is better)
                score = dist * (1.0 - overlap_ratio)

                if dist <= max(self.max_distance, max(m_width, m_height) * 1.2):
                    if score < best_score:
                        best_score = score
                        best_group = group

            if best_group is not None:
                best_group.add_rider(person)

        return groups
