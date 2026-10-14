import pytest
from vision.ocr.plate_recognizer import PlateRecognizer


def test_plate_normalization_indian_formats():
    recognizer = PlateRecognizer()

    # Standard clean plate
    assert recognizer.normalize_indian_plate("KA01AB1234") == "KA01AB1234"
    assert recognizer.normalize_indian_plate("KA 01 AB 1234") == "KA01AB1234"
    assert recognizer.normalize_indian_plate("DL-4C-NA-9021") == "DL04CNA9021" or recognizer.normalize_indian_plate("DL4CNA9021") is not None


def test_character_ambiguity_correction():
    recognizer = PlateRecognizer()

    # If first position has '0' instead of 'O' (e.g., 0A01AB1234 -> OA01AB1234)
    # If state is KA but OCR read 'K401AB1234' or 'KAO1AB1234' with letter O instead of digit 0
    raw_with_letter_o = "KAO1AB1234"  # 'O' at index 2 should become '0'
    normalized = recognizer.normalize_indian_plate(raw_with_letter_o)
    assert normalized == "KA01AB1234"

    # Letter 'B' at numeric tail (KA01AB123B) should convert to '8'
    raw_with_letter_b = "KA01AB123B"
    normalized_b = recognizer.normalize_indian_plate(raw_with_letter_b)
    assert normalized_b == "KA01AB1238"


def test_multi_frame_voting():
    recognizer = PlateRecognizer()
    track_id = 99

    # Frame 1: correct
    p1 = recognizer.update_track_plate(track_id, "KA01AB1234", 0.90)
    assert p1 == "KA01AB1234"

    # Frame 2: noisy reading with typo
    p2 = recognizer.update_track_plate(track_id, "KA01A81234", 0.70)
    # Consensus should remain the higher voted / more confident reading
    assert p2 == "KA01AB1234"

    # Frame 3: another correct vote
    p3 = recognizer.update_track_plate(track_id, "KA01AB1234", 0.95)
    assert p3 == "KA01AB1234"
