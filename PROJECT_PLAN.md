# B.Tech Design Project-I: Project Plan & Architectural Blueprint

**Project Title:** AI-Driven Smart Traffic Violation Detection and Alert System  
**Lead AI Engineer & Full-Stack Developer:** Antigravity AI  
**Scope:** Real-Time Traffic Violation Detection, Multi-Object Tracking, Evidence Generation, Multi-Frame License Plate Recognition (ALPR), and Command Center Dashboard.

---

## 1. System Overview & Objectives
The system automates the detection and reporting of traffic safety and traffic order violations using video streams from surveillance cameras, dashcams, and mobile video feeds. 

### Primary Violation Types Supported:
1. **Riding Without Helmet:** Detection of motorcycle riders and pillion passengers without safety helmets, with spatial rider-motorcycle association.
2. **Triple Riding:** Detection of >2 riders on a single motorcycle via spatial overlap and centroid-distance clustering.
3. **Red-Light Jumping:** Dynamic virtual stop-line crossing detection correlated with traffic signal phase (RED/GREEN).
4. **Wrong-Way Driving:** Lane vector and trajectory centroid angle tracking across sliding frame windows.
5. **Driving Without Seatbelt (Car):** Driver cabin region extraction and seatbelt detection rule (with graceful model availability checks).
6. **Mobile-Phone Use While Driving:** Hand-to-ear / device detection in the driver cabin (with graceful model availability checks).
7. **Extensibility:** Open rule-engine architecture allowing rapid onboarding of illegal parking, speeding, zebra-crossing obstruction, etc.

---

## 2. Technical Stack
- **Language:** Python 3.11+ (Active Environment: Python 3.13)
- **Computer Vision & Deep Learning:** Ultralytics YOLOv8 / YOLO11, PyTorch, OpenCV
- **Multi-Object Tracking (MOT):** ByteTrack (Primary) & DeepSORT (Alternative Abstraction)
- **OCR Engine:** EasyOCR / PaddleOCR / PyTesseract with regex-based Indian High-Security Registration Plate (HSRP) format validation and multi-frame voting
- **Backend API:** FastAPI with async ASGI handlers and CORS
- **Database & ORM:** SQLAlchemy ORM with SQLite (Local prototype/Zero-setup) and PostgreSQL/TimescaleDB support
- **Monitoring Dashboard:** Streamlit with Plotly visual analytics, live annotated video feed, filterable violation triage, and vehicle search
- **Alert System:** Modular AlertService dispatching to Dashboard (live queue), Console logger, with extensible SMS/Email notification hooks

---

## 3. Model Weight & Dataset Reality Assessment

### Currently Available Off-The-Shelf:
- **Pretrained YOLOv8 / YOLO11 Models (`yolov8n.pt`, `yolov8s.pt`):**
  - Classes available in COCO: `person`, `bicycle`, `car`, `motorcycle`, `bus`, `truck`, `traffic light`, `cell phone`.
  - Capable directly of: Vehicle detection, Rider detection, Traffic light detection, Phone detection (if visible), Trajectory tracking, Red light jump rule, Wrong-way rule, Triple riding (spatial rider-to-bike association).

### Custom Weights Required for Production Grade:
- `helmet_model.pt`: Binary helmet / no-helmet detector, head-crop classifier.
- `plate_model.pt`: License plate localization model (Indian HSRP plates).
- `seatbelt_model.pt`: Car cabin windshield crop & seatbelt wear classifier.

### Prototype / Demo Mode Strategy:
- **No Fake AI:** When custom trained weights are not in the `models/` directory, the system automatically uses a robust fallback pipeline:
  - Motorcycle rider head-crop analysis + YOLO person bounding boxes.
  - License plate detector uses candidate bumper contours + OCR fallback, or custom lightweight plate weights when supplied.
  - Clear UI indicators showing whether detections originate from "COCO Baseline + Spatial Heuristics" vs "Dedicated Custom Model".
  - If a model is missing (e.g. seatbelt classifier), the system logs *"Seatbelt detection model not configured"* instead of fabricating outputs.

---

## 4. Phase-by-Phase Roadmap

### Phase 1: Foundation, Pipeline Core & Live Dashboard (Current)
- Directory layout matching specification.
- Configuration management (`config.yaml`, `.env.example`).
- Database schema (SQLAlchemy): `users`, `vehicles`, `detections`, `tracks`, `violations`, `evidence`, `alerts`, `camera_sources`.
- ByteTrack / Tracker abstraction.
- YOLO Detector wrapper with PyTorch CPU/CUDA auto-detection.
- Video upload & processing pipeline.
- FastAPI backend endpoints (`/health`, `/api/cameras`, `/api/violations`, `/api/analytics`, `/api/upload-video`).
- Interactive Traffic Monitoring Dashboard with KPI cards, live feed player, and violation alerts.

### Phase 2: Helmet & Triple Riding Engine with Rider-Bike Association
- Spatial overlap and IoU-based rider-motorcycle association.
- Pillion count calculation (`riders > 2` triggers `TRIPLE_RIDING`).
- Rider head region extraction and helmet classification rule.
- Temporal confirmation window ($N$ out of $M$ frames) and violation cooldown ($10s$) to eliminate double counting.
- Evidence generator: Snapshot framing + pre/post event video buffer extraction.

### Phase 3: Number Plate Detection (ALPR) & Multi-Frame OCR Voting
- License plate localization on vehicle crops.
- Image preprocessing (bilateral filter, thresholding, deskewing).
- Indian plate regex validation (`^[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{4}$`).
- Multi-frame OCR voting accumulator (resolving `0` vs `O`, `8` vs `B`, `1` vs `I`).

### Phase 4: Traffic Rules (Red-Light, Wrong-Way, Cabin Rules)
- Configurable virtual stop lines and intersection polygons.
- Red light state tracker (traffic light detection + bounding box color analysis).
- Centroid movement vector estimation and lane angle deviation check.
- Driver cabin region heuristics for seatbelt and mobile phone rules.

### Phase 5: Production Polish, Camera Management & Alerts
- Virtual cameras and RTSP streaming handler.
- Hotspot mapping (Plotly geo/scatter visualization of Bengaluru junction coordinates).
- Vehicle search page with violation history ledger and evidence previews.
- Notification dispatch abstraction (Console, Dashboard, SMS/Email mock).

### Phase 6: Automated Testing & Benchmarking Suite
- Unit tests: Rules, Tracking, OCR normalization, API endpoints.
- `scripts/evaluate.py` & `scripts/benchmark.py` for mAP, latency, and FPS profiling.
- Comprehensive technical documentation in `README.md`.
