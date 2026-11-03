"""
Traffic Scenario Generator
Generates realistic benchmark scenario videos for demonstrating and validating
each primary traffic violation type:
1. Helmet & Triple Riding
2. Red Light Violation
3. Wrong-Way Driving
"""

import os
import sys
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def generate_helmet_and_triple_scenario(output_path: str = "data/videos/scenario_helmet_triple.mp4", num_frames: int = 100):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    w, h = 1280, 720
    fps = 25.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    bike_x, bike_y = 520, 100

    for i in range(num_frames):
        frame = np.ones((h, w, 3), dtype=np.uint8) * 45
        # Road markings
        for ly in range(0, h, 60):
            cv2.line(frame, (640, ly), (640, ly + 35), (255, 255, 255), 4)
            cv2.line(frame, (350, ly), (350, ly + 35), (200, 200, 200), 2)
            cv2.line(frame, (930, ly), (930, ly + 35), (200, 200, 200), 2)

        # Motorcycle moves downward
        bike_y += 5

        # Motorcycle chassis
        cv2.rectangle(frame, (bike_x - 30, bike_y + 30), (bike_x + 30, bike_y + 115), (30, 160, 30), -1)
        # License plate on rear
        cv2.rectangle(frame, (bike_x - 24, bike_y + 98), (bike_x + 24, bike_y + 112), (255, 255, 255), -1)
        cv2.putText(frame, "KA01AB1234", (bike_x - 22, bike_y + 109), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 0, 0), 1)

        # Rider 1 (Driver) - No helmet
        cv2.rectangle(frame, (bike_x - 18, bike_y - 10), (bike_x + 18, bike_y + 50), (220, 160, 50), -1)
        cv2.circle(frame, (bike_x, bike_y + 5), 14, (180, 130, 90), -1)  # Head texture
        # Hair texture for Canny edge detector
        cv2.circle(frame, (bike_x, bike_y), 10, (20, 20, 20), -1)

        # Rider 2 (Pillion 1)
        cv2.rectangle(frame, (bike_x - 16, bike_y + 20), (bike_x + 16, bike_y + 70), (50, 100, 200), -1)
        cv2.circle(frame, (bike_x, bike_y + 25), 12, (180, 130, 90), -1)

        # Rider 3 (Pillion 2 - Triggers Triple Riding)
        cv2.rectangle(frame, (bike_x - 14, bike_y + 45), (bike_x + 14, bike_y + 90), (180, 50, 180), -1)
        cv2.circle(frame, (bike_x, bike_y + 48), 11, (180, 130, 90), -1)

        # Telemetry overlay
        cv2.putText(frame, "SCENARIO 1: MOTORCYCLE TRIPLE RIDING & NO HELMET", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.putText(frame, f"Frame: {i+1}/{num_frames} | Target: KA01AB1234", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        writer.write(frame)

    writer.release()
    print(f"Generated: {output_path}")


def generate_red_light_scenario(output_path: str = "data/videos/scenario_red_light.mp4", num_frames: int = 100):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    w, h = 1280, 720
    fps = 25.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    car_x, car_y = 780, 180

    for i in range(num_frames):
        frame = np.ones((h, w, 3), dtype=np.uint8) * 45
        for ly in range(0, h, 60):
            cv2.line(frame, (640, ly), (640, ly + 35), (255, 255, 255), 4)

        # Virtual Stop line at Y = 500
        cv2.line(frame, (150, 500), (1150, 500), (0, 0, 255), 4)
        cv2.putText(frame, "STOP LINE [RED]", (160, 490), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

        # Traffic Signal Post
        cv2.rectangle(frame, (1150, 60), (1220, 220), (20, 20, 20), -1)
        cv2.circle(frame, (1185, 100), 22, (0, 0, 255), -1)  # RED ACTIVE
        cv2.circle(frame, (1185, 145), 18, (30, 30, 30), -1)
        cv2.circle(frame, (1185, 185), 18, (30, 30, 30), -1)

        # Car moving downward across the red stop line
        car_y += 6
        cv2.rectangle(frame, (car_x - 65, car_y), (car_x + 65, car_y + 140), (200, 40, 40), -1)
        cv2.rectangle(frame, (car_x - 50, car_y + 25), (car_x + 50, car_y + 65), (100, 180, 220), -1)
        cv2.rectangle(frame, (car_x - 32, car_y + 124), (car_x + 32, car_y + 138), (255, 255, 255), -1)
        cv2.putText(frame, "KA05XY4421", (car_x - 30, car_y + 135), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 0), 1)

        cv2.putText(frame, "SCENARIO 2: RED LIGHT JUMPING AT INTERSECTION", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.putText(frame, f"Frame: {i+1}/{num_frames} | Target: KA05XY4421", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        writer.write(frame)

    writer.release()
    print(f"Generated: {output_path}")


def generate_wrong_way_scenario(output_path: str = "data/videos/scenario_wrong_way.mp4", num_frames: int = 100):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    w, h = 1280, 720
    fps = 25.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # Vehicle driving UPWARDS against downward traffic flow
    car_x, car_y = 480, 600

    for i in range(num_frames):
        frame = np.ones((h, w, 3), dtype=np.uint8) * 45
        for ly in range(0, h, 60):
            cv2.line(frame, (640, ly), (640, ly + 35), (255, 255, 255), 4)

        # Designated lane arrows pointing DOWN
        cv2.arrowedLine(frame, (480, 120), (480, 200), (0, 200, 0), 4)
        cv2.arrowedLine(frame, (800, 120), (800, 200), (0, 200, 0), 4)
        cv2.putText(frame, "LANE DIRECTION (SOUTHBOUND)", (430, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Car moving UPWARDS (wrong way)
        car_y -= 5
        cv2.rectangle(frame, (car_x - 65, car_y), (car_x + 65, car_y + 140), (40, 100, 220), -1)
        cv2.rectangle(frame, (car_x - 50, car_y + 70), (car_x + 50, car_y + 110), (100, 180, 220), -1)
        cv2.rectangle(frame, (car_x - 32, car_y + 10), (car_x + 32, car_y + 24), (255, 255, 255), -1)
        cv2.putText(frame, "DL04CNA9021", (car_x - 30, car_y + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 0), 1)

        cv2.putText(frame, "SCENARIO 3: WRONG-WAY DRIVING AGAINST LANE VECTOR", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.putText(frame, f"Frame: {i+1}/{num_frames} | Target: DL04CNA9021", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        writer.write(frame)

    writer.release()
    print(f"Generated: {output_path}")


def generate_all_scenarios():
    print("Generating comprehensive test scenarios for traffic surveillance validation...")
    generate_helmet_and_triple_scenario()
    generate_red_light_scenario()
    generate_wrong_way_scenario()
    print("All scenarios generated successfully in data/videos/")


if __name__ == "__main__":
    generate_all_scenarios()
