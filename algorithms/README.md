# algorithms/ — the AI and computer-vision logic

Each sub-folder is one algorithm. The pipeline (`pipeline/traffic_pipeline.py`) calls them in this order:

| Step | Folder | Algorithm | Main file | What it does |
|---|---|---|---|---|
| 1 | `detection/` | **YOLO26n** object detection (NMS-free) | `detector.py` | Finds cars, motorcycles, buses, trucks, people and traffic lights in each frame. Also runs the custom helmet / plate model when `models/traffic_custom.pt` exists. |
| 2 | `tracking/` | **ByteTrack** (DeepSORT as an option, IoU matcher as fallback) | `byte_tracker.py`, `deepsort_tracker.py`, `base_tracker.py` | Gives every vehicle and person a stable track id across frames. Needs `pip install lap`. |
| 3 | `association/` | **Rider–motorcycle association** (box overlap + position) | `rider_association.py` | Links each person to the motorcycle they are riding and counts riders per bike. |
| 4 | `plate_ocr/` | **EasyOCR** + Indian plate regex + multi-frame voting | `plate_recognizer.py` | Reads the number plate of a violating vehicle and keeps the most-voted reading per track. |
| 5 | `violation_rules/` | **Rule engine** with N-of-M temporal confirmation (6 of 10 frames) and cooldown | `rule_engine.py` + one file per rule | Decides whether a tracked object is breaking a rule. |
| — | `calibration/` | Per-camera calibration, **HSV signal-colour** reading, day/night check | `camera_calibration.py` | Stop line, lane direction, signal area and rule switches per camera (saved from the dashboard to `data/camera_calibration.yaml`). |

## Violation rules (`violation_rules/`)

| File | Violation | Status |
|---|---|---|
| `helmet_rule.py` | Rider without helmet | Needs the trained `models/traffic_custom.pt` (off until then) |
| `triple_riding_rule.py` | More than two riders on one motorcycle | Working |
| `red_light_rule.py` | Crossing the stop line on red (line-segment intersection) | Working (needs stop line + signal) |
| `wrong_way_rule.py` | Moving against the lane direction (velocity angle) | Working (needs lane direction) |
| `seatbelt_rule.py` | No seatbelt | Placeholder, switched off (future work) |
| `mobile_phone_rule.py` | Phone while riding | Placeholder, switched off (future work) |
| `base_rule.py` | Shared temporal-confirmation and cooldown logic | — |
