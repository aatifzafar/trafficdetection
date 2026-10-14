"""
Seed script to initialize and populate the SQLite database with rich realistic traffic surveillance data:
- Realistic Bengaluru Junction Cameras (MG Road, Silk Board, Hebbal, Indiranagar, etc.)
- Confirmed and review-pending traffic violation events across all 6 categories
- Registered vehicle profiles with HSRP number plates
- Forensic evidence links and snapshots
- Alert records dispatched to command center
"""

import os
import sys
import uuid
import datetime
from pathlib import Path
import numpy as np
import cv2

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.database.database import Base, engine, SessionLocal
from backend.models.models import CameraSource, Vehicle, Violation, Evidence, Alert


def create_sample_evidence_image(filepath: str, violation_type: str, plate: str, location: str):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    img = np.ones((720, 1280, 3), dtype=np.uint8) * 40

    # Road lanes
    for ly in range(0, 720, 60):
        cv2.line(img, (640, ly), (640, ly + 35), (255, 255, 255), 3)
        cv2.line(img, (320, ly), (320, ly + 35), (200, 200, 200), 2)
        cv2.line(img, (960, ly), (960, ly + 35), (200, 200, 200), 2)

    # Stop line
    cv2.line(img, (100, 520), (1180, 520), (0, 0, 255), 4)

    # Vehicle / Motorcycle representation
    if "HELMET" in violation_type or "TRIPLE" in violation_type:
        # Motorcycle box
        cv2.rectangle(img, (460, 320), (580, 520), (0, 0, 255), 3)
        cv2.rectangle(img, (460, 290), (600, 320), (0, 0, 255), -1)
        cv2.putText(img, f"ID:104 MOTORCYCLE [VIOLATION]", (465, 312), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        # Rider head crop box
        cv2.rectangle(img, (490, 220), (550, 280), (0, 255, 255), 2)
        cv2.putText(img, "HEAD: NO HELMET", (460, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    else:
        # Car / Four wheeler box
        cv2.rectangle(img, (420, 380), (680, 620), (0, 0, 255), 3)
        cv2.rectangle(img, (420, 345), (600, 380), (0, 0, 255), -1)
        cv2.putText(img, f"ID:88 CAR [VIOLATION]", (425, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    # License plate badge
    cv2.rectangle(img, (490, 480), (590, 510), (255, 255, 255), -1)
    cv2.putText(img, plate, (495, 502), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # Overlay Telemetry HUD
    cv2.rectangle(img, (0, 0), (1280, 70), (15, 23, 42), -1)
    cv2.putText(img, f"TRAFFIC SURVEILLANCE EVIDENCE ARCHIVE | {violation_type}", (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 215, 255), 2)
    cv2.putText(img, f"Plate: {plate} | Loc: {location} | Timestamp: {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}", (20, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)

    cv2.imwrite(filepath, img)


def seed_database():
    print("Initializing Database tables...")
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    try:
        # 1. Cameras
        if db.query(CameraSource).count() == 0:
            print("Seeding Cameras...")
            cameras = [
                CameraSource(
                    name="MG Road Junction North",
                    location="MG Road - Brigade Rd Crossing, Bengaluru",
                    source_type="VIDEO_FILE",
                    source_url="data/videos/sample_traffic.mp4",
                    latitude=12.9754,
                    longitude=77.6067,
                    direction="Northbound",
                    status="ONLINE",
                ),
                CameraSource(
                    name="Silk Board Flyover Entrance",
                    location="Silk Board Junction, Hosur Road, Bengaluru",
                    source_type="RTSP",
                    source_url="rtsp://demo:demo@stream.traffic.mock/silkboard",
                    latitude=12.9176,
                    longitude=77.6238,
                    direction="Southbound",
                    status="ONLINE",
                ),
                CameraSource(
                    name="Hebbal Outer Ring Rd",
                    location="Hebbal Flyover, Bellary Road, Bengaluru",
                    source_type="VIDEO_FILE",
                    source_url="data/videos/sample_hebbal.mp4",
                    latitude=13.0358,
                    longitude=77.5970,
                    direction="Inbound",
                    status="OFFLINE",
                ),
                CameraSource(
                    name="Indiranagar 100ft Road",
                    location="100 Feet Road, 12th Main Crossing, Bengaluru",
                    source_type="RTSP",
                    source_url="rtsp://demo:demo@stream.traffic.mock/indiranagar",
                    latitude=12.9784,
                    longitude=77.6408,
                    direction="Eastbound",
                    status="ONLINE",
                ),
                CameraSource(
                    name="Koramangala Sony World Signal",
                    location="Sony World Junction, 80ft Road, Bengaluru",
                    source_type="RTSP",
                    source_url="rtsp://demo:demo@stream.traffic.mock/koramangala",
                    latitude=12.9345,
                    longitude=77.6256,
                    direction="Westbound",
                    status="ONLINE",
                ),
            ]
            db.add_all(cameras)
            db.commit()

        # 2. Vehicles & Violations
        if db.query(Violation).count() == 0:
            print("Seeding Vehicles and Violations...")
            now = datetime.datetime.utcnow()

            sample_events = [
                {
                    "plate": "KA01AB1234",
                    "vtype": "NO_HELMET",
                    "type_name": "motorcycle",
                    "conf": 0.94,
                    "loc": "MG Road - Brigade Rd Crossing, Bengaluru",
                    "cam_id": 1,
                    "status": "CONFIRMED",
                    "hours_ago": 1,
                },
                {
                    "plate": "KA05XY4421",
                    "vtype": "TRIPLE_RIDING",
                    "type_name": "motorcycle",
                    "conf": 0.91,
                    "loc": "Silk Board Junction, Hosur Road, Bengaluru",
                    "cam_id": 2,
                    "status": "CONFIRMED",
                    "hours_ago": 3,
                },
                {
                    "plate": "KA03MN9901",
                    "vtype": "RED_LIGHT_JUMP",
                    "type_name": "car",
                    "conf": 0.96,
                    "loc": "Indiranagar 100ft Road, Bengaluru",
                    "cam_id": 4,
                    "status": "CONFIRMED",
                    "hours_ago": 5,
                },
                {
                    "plate": "DL04CNA9021",
                    "vtype": "WRONG_WAY",
                    "type_name": "car",
                    "conf": 0.89,
                    "loc": "Hebbal Flyover, Bellary Road, Bengaluru",
                    "cam_id": 3,
                    "status": "CONFIRMED",
                    "hours_ago": 8,
                },
                {
                    "plate": "KA02GH3312",
                    "vtype": "MOBILE_PHONE_WHILE_DRIVING",
                    "type_name": "car",
                    "conf": 0.87,
                    "loc": "Koramangala Sony World Signal, Bengaluru",
                    "cam_id": 5,
                    "status": "PENDING_REVIEW",
                    "hours_ago": 12,
                },
                {
                    "plate": "MH12DE7890",
                    "vtype": "NO_HELMET",
                    "type_name": "motorcycle",
                    "conf": 0.92,
                    "loc": "MG Road - Brigade Rd Crossing, Bengaluru",
                    "cam_id": 1,
                    "status": "CONFIRMED",
                    "hours_ago": 16,
                },
                {
                    "plate": "KA04JK5544",
                    "vtype": "RED_LIGHT_JUMP",
                    "type_name": "motorcycle",
                    "conf": 0.95,
                    "loc": "Silk Board Junction, Hosur Road, Bengaluru",
                    "cam_id": 2,
                    "status": "CONFIRMED",
                    "hours_ago": 22,
                },
                {
                    "plate": "KA51EF2211",
                    "vtype": "NO_SEATBELT",
                    "type_name": "car",
                    "conf": 0.85,
                    "loc": "Indiranagar 100ft Road, Bengaluru",
                    "cam_id": 4,
                    "status": "PENDING_REVIEW",
                    "hours_ago": 26,
                },
            ]

            for ev in sample_events:
                # Vehicle
                veh = db.query(Vehicle).filter(Vehicle.plate_number == ev["plate"]).first()
                if not veh:
                    veh = Vehicle(
                        plate_number=ev["plate"],
                        vehicle_type=ev["type_name"],
                        first_seen=now - datetime.timedelta(days=7),
                        last_seen=now - datetime.timedelta(hours=ev["hours_ago"]),
                        total_violations=1,
                    )
                    db.add(veh)
                    db.flush()
                else:
                    veh.total_violations += 1

                event_id = f"EVT-{(now - datetime.timedelta(hours=ev['hours_ago'])).strftime('%Y%m%d%H%M')}-{uuid.uuid4().hex[:4].upper()}"
                evidence_img = f"data/evidence/{now.strftime('%Y-%m-%d')}/{event_id}/violation_frame.jpg"
                create_sample_evidence_image(evidence_img, ev["vtype"], ev["plate"], ev["loc"])

                viol = Violation(
                    event_id=event_id,
                    violation_type=ev["vtype"],
                    track_id=100 + ev["cam_id"] * 5,
                    vehicle_id=veh.id,
                    plate_number=ev["plate"],
                    confidence=ev["conf"],
                    timestamp=now - datetime.timedelta(hours=ev["hours_ago"]),
                    location=ev["loc"],
                    camera_id=ev["cam_id"],
                    evidence_path=evidence_img,
                    status=ev["status"],
                )
                db.add(viol)
                db.flush()

                evidence = Evidence(
                    event_id=event_id,
                    violation_id=viol.id,
                    image_path=evidence_img,
                    video_path=None,
                    metadata_json=f'{{"event_id": "{event_id}", "plate": "{ev["plate"]}", "fine_amount": 1000}}',
                )
                db.add(evidence)

                alert = Alert(
                    event_id=event_id,
                    violation_id=viol.id,
                    violation_type=ev["vtype"],
                    channel="DASHBOARD",
                    message=f"Violation {ev['vtype']} detected for {ev['plate']} at {ev['loc']}",
                    status="SENT",
                    sent_at=now - datetime.timedelta(hours=ev["hours_ago"]),
                )
                db.add(alert)

            db.commit()
            print(f"Successfully seeded {len(sample_events)} violations and vehicles.")

    finally:
        db.close()


if __name__ == "__main__":
    seed_database()
