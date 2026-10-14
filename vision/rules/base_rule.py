import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from typing import Dict, Any, Optional, List


class BaseViolationRule(ABC):
    """
    Abstract Base Rule with temporal confirmation window and cooldown throttling
    to eliminate frame-flicker false positives and duplicate penalty spam.
    """

    def __init__(
        self,
        rule_name: str,
        temporal_window: int = 10,
        min_positive_frames: int = 6,
        cooldown_seconds: float = 10.0,
    ):
        self.rule_name = rule_name
        self.temporal_window = temporal_window
        self.min_positive_frames = min_positive_frames
        self.cooldown_seconds = cooldown_seconds

        # track_id -> deque of bools (True for positive violation frame)
        self.history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=self.temporal_window))

        # track_id -> last violation trigger timestamp
        self.last_triggered: Dict[int, float] = {}

    def is_cooling_down(self, track_id: int) -> bool:
        """Returns True if this track_id was recently flagged and is in cooldown period."""
        if track_id not in self.last_triggered:
            return False
        return (time.time() - self.last_triggered[track_id]) < self.cooldown_seconds

    def record_frame_evidence(self, track_id: int, is_violating: bool) -> bool:
        """
        Records frame status into sliding window.
        Returns True if confirmed violation threshold reached and cooldown expired.
        """
        if self.is_cooling_down(track_id):
            return False

        dq = self.history[track_id]
        dq.append(is_violating)

        if len(dq) >= self.min_positive_frames:
            positives = sum(1 for x in dq if x)
            if positives >= self.min_positive_frames:
                self.last_triggered[track_id] = time.time()
                dq.clear()  # reset window after confirmed trigger
                return True

        return False

    @abstractmethod
    def evaluate(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Evaluates current frame context and returns list of confirmed violation dictionaries.
        """
        pass
