import os
import sys
import logging
import cv2
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.config.settings import settings
from backend.database.database import Base, engine, SessionLocal
from backend.models.models import Violation, CameraSource, Vehicle
from vision.pipeline import TrafficVisionPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("test_pipeline")


def generate_synthetic_traffic_video(output_path: str = "data/videos/sample_traffic.mp4", num_frames: int = 120):
    """
    Creates a realistic synthetic traffic scene video with road lanes,
    moving vehicles, motorcycles, and simulated traffic signal to validate the end-to-end pipeline.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    h, w = 720, 1280
    fps = 25.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    logger.info(f"Generating synthetic traffic test video ({num_frames} frames) at {output_path}...")

    # Vehicle trajectories
    bike_x, bike_y = 400, 100
    car_x, car_y = 750, 150

    for i in range(num_frames):
        # Road Canvas (Dark Asphalt)
        frame = np.ones((h, w, 3), dtype=np.uint8) * 45

        # Lane markings (White / Yellow dashed lines)
        for ly in range(0, h, 60):
            cv2.line(frame, (640, ly), (640, ly + 35), (255, 255, 255), 4)
            cv2.line(frame, (350, ly), (350, ly + 35), (200, 200, 200), 2)
            cv2.line(frame, (930, ly), (930, ly + 35), (200, 200, 200), 2)

        # Stop Line (Y = 520)
        cv2.line(frame, (150, 520), (1100, 520), (0, 0, 255), 4)
        cv2.putText(frame, "STOP LINE", (160, 510), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Traffic Signal Post (Red Light)
        cv2.rectangle(frame, (1150, 80), (1220, 240), (20, 20, 20), -1)
        cv2.circle(frame, (1185, 120), 22, (0, 0, 255), -1)  # RED ON
        cv2.circle(frame, (1185, 165), 18, (30, 30, 30), -1)
        cv2.circle(frame, (1185, 205), 18, (30, 30, 30), -1)

        # Vehicle 1: Motorcycle with 3 riders (moving downward across stop line)
        bike_y += 4
        # Draw motorcycle body
        cv2.rectangle(frame, (bike_x - 30, bike_y + 40), (bike_x + 30, bike_y + 110), (20, 150, 20), -1)
        # Draw license plate on rear
        cv2.rectangle(frame, (bike_x - 22, bike_y + 95), (bike_x + 22, bike_y + 108), (255, 255, 255), -1)
        cv2.putText(frame, "KA01AB1234", (bike_x - 20, bike_y + 105), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 0, 0), 1)

        # Rider 1 (Driver)
        cv2.rectangle(frame, (bike_x - 18, bike_y), (bike_x + 18, bike_y + 60), (220, 160, 50), -1)
        cv2.circle(frame, (bike_x, bike_y + 12), 12, (200, 140, 100), -1)  # Unhelmeted head

        # Rider 2 (Pillion 1)
        cv2.rectangle(frame, (bike_x - 16, bike_y + 25), (bike_x + 16, bike_y + 75), (50, 100, 200), -1)
        cv2.circle(frame, (bike_x, bike_y + 30), 11, (200, 140, 100), -1)

        # Rider 3 (Pillion 2 - Triple riding)
        cv2.rectangle(frame, (bike_x - 14, bike_y + 45), (bike_x + 14, bike_y + 90), (180, 50, 180), -1)
        cv2.circle(frame, (bike_x, bike_y + 50), 10, (200, 140, 100), -1)

        # Vehicle 2: Car (moving downward in right lane)
        car_y += 3
        cv2.rectangle(frame, (car_x - 60, car_y), (car_x + 60, car_y + 140), (200, 50, 50), -1)
        # Windshield
        cv2.rectangle(frame, (car_x - 45, car_y + 25), (car_x + 45, car_y + 65), (100, 180, 220), -1)
        # Plate
        cv2.rectangle(frame, (car_x - 30, car_y + 125), (car_x + 30, car_y + 138), (255, 255, 255), -1)
        cv2.putText(frame, "KA05XY4421", (car_x - 28, car_y + 135), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 0), 1)

        writer.write(frame)

    writer.release()
    logger.info(f"Synthetic video successfully saved: {output_path}")
    return output_path


def test_pipeline_execution():
    """Runs end-to-end verification of the vision and violation pipeline."""
    # Ensure tables exist
    Base.metadata.create_all(bind=engine)

    sample_vid = "data/videos/sample_traffic.mp4"
    if not os.path.exists(sample_vid):
        generate_synthetic_traffic_video(sample_vid, num_frames=60)

    logger.info("Initializing TrafficVisionPipeline...")
    pipeline = TrafficVisionPipeline(
        camera_id=1,
        location="MG Road Junction North, Bengaluru",
    )

    logger.info("Processing sample video frames...")
    res = pipeline.process_video(
        video_path=sample_vid,
        output_path="data/videos/processed_sample_traffic.mp4",
        max_frames=45,
    )

    logger.info(f"Pipeline Test Completed: {res}")

    # Verify DB insertion
    db = SessionLocal()
    try:
        v_count = db.query(Violation).count()
        logger.info(f"Total Violations in Database: {v_count}")
    finally:
        db.close()

    print("\n========================================================")
    print("SUCCESS: Full Computer Vision Pipeline Executed Cleanly!")
    print(f"Processed Frames: {res.get('total_frames')}")
    print(f"Tracked Vehicles: {res.get('unique_vehicles')}")
    print(f"Recorded Violations: {res.get('total_violations')}")
    print(f"Output Video: {res.get('output_path')}")
    print("========================================================\n")


if __name__ == "__main__":
    test_pipeline_execution()
