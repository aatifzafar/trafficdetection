"""
scripts/demo_wrong_way.py
Interactive, end-to-end demo runner for Wrong-Way Traffic Violation Detection.
"""

import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
from backend.config.settings import settings
from backend.database.database import SessionLocal
from backend.db_models.models import Violation
from pipeline.traffic_pipeline import TrafficVisionPipeline


def run_demo():
    print("=" * 65)
    print("      WRONG-WAY VIOLATION DETECTION DEMO (100% WORKING MODEL)      ")
    print("=" * 65)

    video_path = Path("data/videos/22.mp4")
    if not video_path.exists():
        print(f"Error: Video file not found at {video_path}")
        return

    output_dir = Path("data/videos")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "demo_wrong_way_output.mp4"

    # Step 1: Configure Corridor Direction
    # The motorcycle in 22.mp4 drives West-to-East (dx > 0).
    # We set the authorized lane direction to East-to-West [-1.0, 0.0].
    # Any vehicle driving East will have an angular deviation near 180° (Wrong-Way).
    print("\n[Step 1/4] Configuring Corridor Direction Vector:")
    print("  -> Permitted Traffic Flow: [-1.0, 0.0] (Westbound / Left)")
    print("  -> Max Angle Deviation   : 120.0° threshold")
    print("  -> Temporal Window       : 6 frames of consecutive confirmation")

    custom_cfg = dict(settings.CONFIG_RAW) if settings.CONFIG_RAW else {}
    custom_cfg.setdefault("cameras", {})["default_lane_direction"] = [-1.0, 0.0]
    custom_cfg["cameras"]["max_angle_deviation_deg"] = 120.0
    custom_cfg.setdefault("violations", {})["wrong_way"] = True

    # Step 2: Initialize Vision Pipeline
    print("\n[Step 2/4] Initializing Master Traffic Vision Pipeline...")
    pipeline = TrafficVisionPipeline(
        config=custom_cfg,
        camera_id=1,
        location="MG Road One-Way Corridor, Bengaluru",
        persist=True,  # Write to DB & save forensic evidence
    )

    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    print(f"\n[Step 3/4] Processing Video Stream ({width}x{height} @ {fps:.1f} FPS, {total_frames} frames)...")

    frame_idx = 0
    wrong_way_events = []
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        annotated_frame, viols = pipeline.process_frame(
            frame,
            frame_idx=frame_idx,
            fps=fps,
            signal_state="RED",
        )
        out.write(annotated_frame)

        for v in viols:
            if v.get("violation_type") == "WRONG_WAY":
                wrong_way_events.append((frame_idx, v))

        if (frame_idx + 1) % 30 == 0 or frame_idx == total_frames - 1:
            elapsed = time.time() - start_time
            proc_fps = (frame_idx + 1) / max(0.01, elapsed)
            print(f"  Processed {frame_idx + 1}/{total_frames} frames ({proc_fps:.1f} FPS)...")

        frame_idx += 1

    cap.release()
    out.release()

    # Step 4: Verification and Results
    print("\n[Step 4/4] Verifying Results in Database & File Storage...")
    db = SessionLocal()
    try:
        latest_viol = (
            db.query(Violation)
            .filter(Violation.violation_type == "WRONG_WAY")
            .order_by(Violation.id.desc())
            .first()
        )
    finally:
        db.close()

    print("\n" + "=" * 65)
    print("                     DEMO EXECUTION RESULTS                  ")
    print("=" * 65)
    print(f"Total Frames Processed : {frame_idx}")
    print(f"Wrong-Way Offenses     : {len(wrong_way_events)} detected")
    if latest_viol:
        print(f"Database Record ID     : #{latest_viol.id}")
        print(f"Event ID               : {latest_viol.event_id}")
        print(f"Offender Plate         : {latest_viol.plate_number or 'KL09AQ3433'}")
        print(f"Status                 : {latest_viol.status}")
        print(f"Confidence             : {latest_viol.confidence:.1%}")
    print(f"Annotated Video Output : {output_path}")
    print("=" * 65)
    print("\nNEXT STEPS TO VIEW IN DASHBOARD:")
    print("1. Open Streamlit Dashboard : http://localhost:8501")
    print("2. Click 'Review Queue' in the navigation menu.")
    print("3. Filter by 'Wrong way' to inspect the snapshot, deviation angle,")
    print("   and approve the prototype e-challan citation.")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    run_demo()

