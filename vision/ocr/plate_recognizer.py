import re
import logging
from collections import defaultdict
from typing import Dict, Any, Optional, List, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class PlateRecognizer:
    """
    Automatic License Plate Recognition (ALPR) system with image preprocessing,
    Indian registration plate format normalization, and multi-frame voting.
    """

    def __init__(
        self,
        engine_type: str = "easyocr",
        plate_regex: str = r"^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{4}$",
        min_confidence: float = 0.35,
    ):
        self.plate_regex = re.compile(plate_regex)
        self.min_confidence = min_confidence
        self.ocr_reader = None
        self.engine_type = engine_type.lower()
        self._ocr_initialized = False

        # Multi-frame voting accumulator: track_id -> { plate_text: {"count": N, "total_conf": float} }
        self.track_votes: Dict[int, Dict[str, Dict[str, Any]]] = defaultdict(lambda: defaultdict(lambda: {"count": 0, "total_conf": 0.0}))

    def _ensure_ocr(self):
        if self._ocr_initialized:
            return
        self._ocr_initialized = True
        try:
            if self.engine_type == "easyocr":
                import easyocr
                logger.info("Initializing EasyOCR reader (English)...")
                # gpu=False by default for reliable cpu execution or falls back gracefully
                self.ocr_reader = easyocr.Reader(["en"], gpu=False)
                logger.info("EasyOCR reader initialized successfully.")
        except Exception as e:
            logger.warning(f"Could not initialize {self.engine_type}: {e}. Fallback OCR mode active.")

    def preprocess_plate(self, img: np.ndarray) -> np.ndarray:
        """Applies grayscale, bilateral filter, and adaptive thresholding for plate OCR clarity."""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # Resize for standard OCR input if too small
        h, w = gray.shape[:2]
        if w < 120 or h < 35:
            scale = max(120 / max(1, w), 35 / max(1, h))
            gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

        # Bilateral filter to remove noise while keeping edges sharp
        filtered = cv2.bilateralFilter(gray, 11, 17, 17)

        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        return thresh

    def normalize_indian_plate(self, raw_text: str) -> Optional[str]:
        """
        Cleans and normalizes OCR text using Indian High-Security Plate rules.
        Replaces ambiguous characters based on positional grammar:
        [State 2 Letters][District 1-2 Digits][Series 0-3 Letters][Number 4 Digits]
        """
        cleaned = re.sub(r"[^A-Za-z0-9]", "", raw_text).upper()
        if len(cleaned) < 8 or len(cleaned) > 11:
            return None

        # Positional character correction mappings
        to_letters = {"0": "O", "1": "I", "8": "B", "5": "S", "2": "Z"}
        to_digits = {"O": "0", "I": "1", "B": "8", "S": "5", "Z": "2", "G": "6", "Q": "0"}

        chars = list(cleaned)

        # Positions 0, 1 must be State Letters (e.g. KA, DL, MH)
        for i in [0, 1]:
            if chars[i] in to_letters:
                chars[i] = to_letters[chars[i]]

        # Position 2, 3 must be District Digits (e.g. 01, 05)
        for i in [2, 3]:
            if i < len(chars) and chars[i] in to_digits:
                chars[i] = to_digits[chars[i]]

        # Last 4 characters must be Number Digits
        for i in range(len(chars) - 4, len(chars)):
            if chars[i] in to_digits:
                chars[i] = to_digits[chars[i]]

        candidate = "".join(chars)
        if self.plate_regex.match(candidate):
            return candidate

        return None

    def read_plate(self, crop: np.ndarray) -> Optional[Tuple[str, float]]:
        """Runs OCR on cropped plate/bumper region."""
        self._ensure_ocr()
        if crop is None or crop.size == 0 or self.ocr_reader is None:
            return None

        try:
            preprocessed = self.preprocess_plate(crop)
            results = self.ocr_reader.readtext(preprocessed)

            for bbox, text, conf in results:
                if conf < self.min_confidence:
                    continue
                normalized = self.normalize_indian_plate(text)
                if normalized:
                    return (normalized, float(conf))

            # If preprocessed failed, try raw crop as fallback
            raw_results = self.ocr_reader.readtext(crop)
            for bbox, text, conf in raw_results:
                if conf < self.min_confidence:
                    continue
                normalized = self.normalize_indian_plate(text)
                if normalized:
                    return (normalized, float(conf))
        except Exception as e:
            logger.debug(f"OCR inference step error: {e}")

        return None

    def update_track_plate(
        self, track_id: int, plate_text: str, confidence: float
    ) -> str:
        """
        Registers OCR prediction into the multi-frame voting pool for this vehicle track,
        and returns the consensus winning plate string.
        """
        votes = self.track_votes[track_id]
        votes[plate_text]["count"] += 1
        votes[plate_text]["total_conf"] += confidence

        # Pick candidate with highest vote count, tie-broken by average confidence
        best_plate = max(
            votes.keys(),
            key=lambda p: (
                votes[p]["count"],
                votes[p]["total_conf"] / max(1, votes[p]["count"]),
            ),
        )
        return best_plate

    def get_track_plate(self, track_id: int) -> Optional[str]:
        """Returns current consensus plate for track_id if any."""
        votes = self.track_votes.get(track_id)
        if not votes:
            return None
        return max(
            votes.keys(),
            key=lambda p: (
                votes[p]["count"],
                votes[p]["total_conf"] / max(1, votes[p]["count"]),
            ),
        )
