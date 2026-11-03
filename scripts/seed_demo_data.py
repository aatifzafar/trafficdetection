"""
Fill the database with DEMO records so every dashboard page can be shown before real cameras run.

    python scripts/seed_demo_data.py            # add demo cameras + ~60 demo violations over 3 weeks
    python scripts/seed_demo_data.py --remove   # delete every demo record again

Demo records are clearly marked: event ids start with EVT-DEMO-, evidence images carry a
"DEMO RECORD" watermark, and nothing here is presented as a real detection or a real accuracy figure.
Evidence frames are taken from the videos in data/videos/ when available.
"""

import argparse
import datetime
import os
import random
import sys
import uuid
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.database.database import SessionLocal, ensure_schema  # noqa: E402
from backend.db_models.models import Alert, CameraSource, Evidence, Vehicle, Violation  # noqa: E402

DEMO_PREFIX = "EVT-DEMO-"
CAMERAS = [
    ("Silk Board Jn · Cam 2", "Silk Board Junction, Hosur Rd", "VIDEO_FILE", "data/videos/22.mp4", 12.9177, 77.6233),
    ("Hebbal Flyover · Cam 1", "Hebbal Flyover, Bellary Rd", "VIDEO_FILE", "data/videos/scenario_red_light.mp4", 13.0358, 77.5970),
    ("Koramangala 80ft Rd · Cam 4", "80 Feet Rd, Koramangala", "VIDEO_FILE", "data/videos/scenario_wrong_way.mp4", 12.9345, 77.6256),
    ("MG Road · Cam 1", "MG Road – Brigade Rd crossing", "VIDEO_FILE", "data/videos/scenario_helmet_triple.mp4", 12.9754, 77.6067),
]
VIDEO_FOR = {"NO_HELMET": "data/videos/22.mp4", "TRIPLE_RIDING": "data/videos/scenario_helmet_triple.mp4",
             "RED_LIGHT_JUMP": "data/videos/scenario_red_light.mp4", "WRONG_WAY": "data/videos/scenario_wrong_way.mp4"}
STATES = ["KA05", "KA01", "KA03", "KA51", "KA04", "KA53", "KA02", "KA41", "KL09", "TN07", "AP16"]
OFFICERS = ["Insp. R. Kumar", "SI P. Shetty", "Insp. S. Rao"]
REASONS = {"NO_HELMET": "Not a violation", "TRIPLE_RIDING": "Wrong vehicle or rider",
           "RED_LIGHT_JUMP": "Image too unclear to decide", "WRONG_WAY": "Not a violation"}


def _frame(video: str, idx: int):
    if not os.path.exists(video):
        return None
    cap = cv2.VideoCapture(video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx % n)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def _evidence(event_id: str, vtype: str, i: int, night: bool) -> str:
    folder = Path("data/evidence/demo") / event_id
    folder.mkdir(parents=True, exist_ok=True)
    img = _frame(VIDEO_FOR[vtype], 7 + i * 11)
    if img is None:
        img = np.full((720, 1280, 3), 60, np.uint8)
    if night:
        img = cv2.convertScaleAbs(cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR), alpha=0.55, beta=0)
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, h - 34), (w, h), (0, 0, 0), -1)
    cv2.putText(img, "DEMO RECORD - not a real detection", (12, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    path = folder / "violation_frame.jpg"
    cv2.imwrite(str(path), img)
    return str(path)


def seed(n: int = 60):
    ensure_schema()
    rnd = random.Random(42)
    db = SessionLocal()
    try:
        cam_ids = []
        for name, loc, kind, url, lat, lon in CAMERAS:
            cam = db.query(CameraSource).filter(CameraSource.name == name).first()
            if cam is None:
                cam = CameraSource(name=name, location=loc, source_type=kind, source_url=url,
                                   latitude=lat, longitude=lon, status="ONLINE", direction="Inbound")
                db.add(cam)
                db.flush()
            cam_ids.append((cam.id, loc))
        db.commit()

        now = datetime.datetime.utcnow()
        weights = {"NO_HELMET": 5, "RED_LIGHT_JUMP": 3, "TRIPLE_RIDING": 2, "WRONG_WAY": 1}
        types = [t for t, k in weights.items() for _ in range(k)]
        plates = [f"{rnd.choice(STATES)}{rnd.choice('ABCDEFHJKMNPRSTUVWXYZ')}{rnd.choice('ABCDEFHJKMNPRSTUVWXYZ')}{rnd.randint(1000, 9999)}"
                  for _ in range(35)]
        for i in range(n):
            vtype = rnd.choice(types)
            cam_id, loc = cam_ids[{"NO_HELMET": 0, "TRIPLE_RIDING": 3, "RED_LIGHT_JUMP": 1, "WRONG_WAY": 2}[vtype]]
            hour = rnd.choice([8, 9, 9, 10, 13, 17, 18, 18, 19, 21, 22])
            ts = (now - datetime.timedelta(days=rnd.randint(0, 20))).replace(hour=hour, minute=rnd.randint(0, 59))
            if i < 6:
                ts = now - datetime.timedelta(minutes=10 * (i + 1))  # a few from today, still waiting
            night = hour >= 19
            event_id = f"{DEMO_PREFIX}{ts:%Y%m%d%H%M}-{uuid.uuid4().hex[:4].upper()}"
            plate = None if rnd.random() < (0.35 if night else 0.12) else rnd.choice(plates)
            conf = round(rnd.uniform(0.70, 0.96), 2)
            status = "PENDING_REVIEW" if i < 9 else rnd.choices(["APPROVED", "REJECTED", "DISPUTED"], [0.78, 0.18, 0.04])[0]
            reviewed = None if status == "PENDING_REVIEW" else ts + datetime.timedelta(minutes=rnd.randint(1, 9), seconds=rnd.randint(0, 59))
            img = _evidence(event_id, vtype, i, night)
            v = Violation(event_id=event_id, violation_type=vtype, track_id=rnd.randint(1, 90), plate_number=plate,
                          confidence=conf, timestamp=ts, created_at=ts, location=loc, camera_id=cam_id,
                          evidence_path=img, status=status, source="CITIZEN" if i in (3, 17, 29) else "CCTV",
                          light_condition="night" if night else "day", merged_count=rnd.choice([0, 0, 0, 1, 2]),
                          reviewed_by=None if not reviewed else rnd.choice(OFFICERS), reviewed_at=reviewed,
                          reject_reason=REASONS[vtype] if status == "REJECTED" else None)
            db.add(v)
            db.flush()
            db.add(Evidence(event_id=event_id, violation_id=v.id, image_path=img, video_path=None, metadata_json=None))
            db.add(Alert(event_id=event_id, violation_id=v.id, violation_type=vtype, channel="DASHBOARD",
                         message=f"DEMO {vtype} at {loc}", status="SENT", sent_at=ts))
            if plate and status in ("APPROVED", "DISPUTED"):
                veh = db.query(Vehicle).filter(Vehicle.plate_number == plate).first()
                if veh is None:
                    veh = Vehicle(plate_number=plate, total_violations=0, first_seen=ts, last_seen=ts)
                    db.add(veh)
                    db.flush()
                veh.total_violations += 1
                v.vehicle_id = veh.id
        db.commit()
        print(f"Added {n} demo violations (event ids start with {DEMO_PREFIX}) and {len(CAMERAS)} demo cameras.")
    finally:
        db.close()


def remove():
    ensure_schema()
    db = SessionLocal()
    try:
        ids = [v.id for v in db.query(Violation).filter(Violation.event_id.like(f"{DEMO_PREFIX}%")).all()]
        if ids:
            db.query(Evidence).filter(Evidence.violation_id.in_(ids)).delete(synchronize_session=False)
            db.query(Alert).filter(Alert.violation_id.in_(ids)).delete(synchronize_session=False)
            db.query(Violation).filter(Violation.id.in_(ids)).delete(synchronize_session=False)
        names = [c[0] for c in CAMERAS]
        db.query(CameraSource).filter(CameraSource.name.in_(names)).delete(synchronize_session=False)
        db.commit()
        print(f"Removed {len(ids)} demo violations and the demo cameras.")
    finally:
        db.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--remove", action="store_true", help="delete all demo records")
    ap.add_argument("-n", type=int, default=60, help="number of demo violations")
    a = ap.parse_args()
    remove() if a.remove else seed(a.n)
