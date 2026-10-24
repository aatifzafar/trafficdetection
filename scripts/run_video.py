"""
run_video.py — End-to-end traffic violation video processor.

CLI Arguments:
    --video: Path to input video file
    --output: Path to save annotated output video (optional)
    --signal: RED | GREEN | AUTO (traffic light state)
    --stop-line: x1,y1,x2,y2 (virtual stop line for red-light jump rule)
    --lane: dx,dy (primary traffic flow vector for wrong-way rule)
    --screenshots: Save key detection snapshot frames to data/evidence/
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List

import cv2
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.config.settings import settings
from vision.pipeline import TrafficVisionPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Run Traffic Violation Detection Pipeline on Video")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--output", type=str, default=None, help="Path to save annotated video")
    parser.add_argument("--signal", type=str, default="RED", choices=["RED", "GREEN", "AUTO"], help="Traffic signal state")
    parser.add_argument("--stop-line", type=str, default=None, help="Stop line coordinates: x1,y1,x2,y2")
    parser.add_argument("--lane", type=str, default=None, help="Lane direction vector: dx,dy (e.g. 0,1)")
    parser.add_argument("--save-screenshots", type=int, default=5, help="Number of detection screenshots to save")
    return parser.parse_args()


def run_pipeline_on_video(
    video_path: str,
    output_path: str = None,
    signal_state: str = "RED",
    stop_line_coords: str = None,
    lane_coords: str = None,
    num_screenshots: int = 5,
):
    video_file = Path(video_path)
    if not video_file.exists():
        print(f"Error: Video file not found at {video_file}")
        sys.exit(1)

    if output_path is None:
        output_dir = Path("data/videos")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / f"processed_{video_file.stem}.mp4")

    # Configure overrides
    custom_cfg = dict(settings.CONFIG_RAW) if settings.CONFIG_RAW else {}
    if stop_line_coords:
        try:
            x1, y1, x2, y2 = [float(v.strip()) for v in stop_line_coords.split(",")]
            custom_cfg.setdefault("cameras", {})["default_stop_line"] = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
            print(f"Configured Stop Line: ({x1}, {y1}) -> ({x2}, {y2})")
        except Exception as e:
            print(f"Warning: Failed to parse --stop-line: {e}")

    if lane_coords:
        try:
            dx, dy = [float(v.strip()) for v in lane_coords.split(",")]
            custom_cfg.setdefault("cameras", {})["default_lane_direction"] = [dx, dy]
            print(f"Configured Lane Vector: [{dx}, {dy}]")
        except Exception as e:
            print(f"Warning: Failed to parse --lane: {e}")

    print(f"\nInitializing TrafficVisionPipeline...")
    pipeline = TrafficVisionPipeline(config=custom_cfg, location=f"Video Analysis: {video_file.name}")

    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        print(f"Error: Failed to open video {video_file}")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames_in_file = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video metadata: {width}x{height} @ {fps:.2f} FPS | Total frames: {total_frames_in_file}")
    print(f"Signal State: {signal_state}")
    print(f"Output Video: {output_path}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    start_wall_time = time.time()
    violations_by_type: Dict[str, int] = {}
    saved_screenshots = 0
    screenshot_paths: List[str] = []

    shots_dir = Path("data/evidence/snapshots")
    shots_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- Processing Video Frames ---")
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        # In AUTO signal mode: cycle green / red every 150 frames
        curr_signal = signal_state
        if signal_state == "AUTO":
            curr_signal = "GREEN" if (frame_idx // 150) % 2 == 0 else "RED"

        annotated_frame, frame_violations = pipeline.process_frame(
            frame=frame,
            frame_idx=frame_idx,
            fps=fps,
            signal_state=curr_signal,
        )

        out_writer.write(annotated_frame)

        for v in frame_violations:
            v_type = v.get("violation_type", "UNKNOWN")
            violations_by_type[v_type] = violations_by_type.get(v_type, 0) + 1

            # Save screenshot of incident
            if saved_screenshots < num_screenshots:
                saved_screenshots += 1
                shot_path = shots_dir / f"detection_screenshot_{saved_screenshots}_{v_type}_{frame_idx}.jpg"
                cv2.imwrite(str(shot_path), annotated_frame)
                screenshot_paths.append(str(shot_path))

        frame_idx += 1
        if frame_idx % 30 == 0:
            elapsed = time.time() - start_wall_time
            proc_fps = frame_idx / max(0.001, elapsed)
            print(f"Processed {frame_idx}/{total_frames_in_file} frames ({proc_fps:.1f} FPS)...")

    cap.release()
    out_writer.release()

    total_time = time.time() - start_wall_time
    avg_fps = frame_idx / max(0.001, total_time)

    # Collect unique plate numbers recognized by ALPR
    plates_read = set()
    for tid in pipeline.total_vehicles_seen:
        plate = pipeline.plate_recognizer.get_track_plate(tid)
        if plate and plate != "UNKNOWN":
            plates_read.add(plate)

    # If fewer than num_screenshots saved from violations, capture regular detection frames
    if saved_screenshots < num_screenshots:
        cap_review = cv2.VideoCapture(output_path)
        sample_step = max(1, frame_idx // (num_screenshots - saved_screenshots + 1))
        f_i = 0
        while cap_review.isOpened() and saved_screenshots < num_screenshots:
            ret, frame = cap_review.read()
            if not ret:
                break
            if f_i % sample_step == 0 and f_i > 0:
                saved_screenshots += 1
                shot_path = shots_dir / f"detection_screenshot_{saved_screenshots}_frame_{f_i}.jpg"
                cv2.imwrite(str(shot_path), frame)
                screenshot_paths.append(str(shot_path))
            f_i += 1
        cap_review.release()

    # Print Summary Table
    print("\n" + "=" * 60)
    print("           VIDEO PROCESSING SUMMARY REPORT           ")
    print("=" * 60)
    print(f"Input Video:         {video_file.name}")
    print(f"Total Frames:        {frame_idx}")
    print(f"Processing Time:     {total_time:.2f} seconds")
    print(f"Average FPS:         {avg_fps:.2f} FPS")
    print(f"Vehicles Tracked:    {len(pipeline.total_vehicles_seen)}")
    print(f"Unique Plates Read:  {list(plates_read) if plates_read else 'None detected'}")
    print("-" * 60)
    print("Violations by Type:")
    if violations_by_type:
        for v_type, count in violations_by_type.items():
            print(f"  • {v_type:<24}: {count}")
    else:
        print("  • None detected")
    print("-" * 60)
    print(f"Output Video:        {output_path}")
    print(f"Saved Screenshots:   {len(screenshot_paths)} images in data/evidence/snapshots/")
    for sp in screenshot_paths:
        print(f"  -> {sp}")
    print("=" * 60 + "\n")

    return {
        "frames": frame_idx,
        "fps": avg_fps,
        "violations": violations_by_type,
        "plates": list(plates_read),
        "output_video": output_path,
        "screenshots": screenshot_paths,
    }


if __name__ == "__main__":
    args = parse_args()
    run_pipeline_on_video(
        video_path=args.video,
        output_path=args.output,
        signal_state=args.signal,
        stop_line_coords=args.stop_line,
        lane_coords=args.lane,
        num_screenshots=args.save_screenshots,
    )
