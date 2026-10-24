# B.Tech Design Project-I: Project Plan & Architectural Blueprint

**Project Title:** AI-Driven Smart Traffic Violation Detection and Alert System  
**Lead AI Engineer & Full-Stack Developer:** Aatif Zafar  
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
- **Language:** Python 3.11+ (Active Environment: Python 3.14)
- **Computer Vision & Deep Learning:** Ultralytics YOLO26 (`yolo26n.pt`), PyTorch, OpenCV
- **Multi-Object Tracking (MOT):** ByteTrack (Ultralytics native engine with IoU fallback)
- **OCR Engine:** EasyOCR / PyTesseract with regex-based Indian High-Security Registration Plate (HSRP) format validation and multi-frame voting
- **Backend API:** FastAPI with async ASGI handlers and CORS
- **Database & ORM:** SQLAlchemy ORM with SQLite (Local prototype/Zero-setup) and PostgreSQL/TimescaleDB support
- **Monitoring Dashboard:** Streamlit with Plotly visual analytics, live annotated video feed, filterable violation triage, and vehicle search
- **Alert System:** Modular AlertService dispatching to Dashboard (live queue), Console logger, and Prototype Violation Notice Generator

---

## 3. Model Weight & Dataset Reality Assessment

### Currently Available Off-The-Shelf:
- **Pretrained YOLO26 Models (`yolo26n.pt`):**
  - Architecture: NMS-free, end-to-end real-time object detection (Ultralytics >= 8.4.0).
  - Classes available in COCO: `person`, `bicycle`, `car`, `motorcycle`, `bus`, `truck`, `traffic light`, `cell phone`.
  - Capable directly of: Vehicle detection, Rider detection, Traffic light detection, Phone detection (if visible), Trajectory tracking, Red light jump rule, Wrong-way rule, Triple riding (spatial rider-to-bike association).

### Custom Weights & Training (Merged YOLO Dataset):
- `models/traffic_custom.pt`: Unified custom model detecting 3 target classes:
  - Class 0: `helmet`
  - Class 1: `no_helmet`
  - Class 2: `number_plate`
- Built from 4,016 merged and deduplicated images across 6 source archives (`datasets/merged/`).
- Trained on Google Colab T4 GPU via `notebooks/train_colab.ipynb`.

### Prototype / Fallback Strategy:
- **No Fake AI:** When custom trained weights are not in the `models/` directory, the system automatically uses a robust fallback pipeline:
  - Motorcycle rider head-crop analysis + YOLO person bounding boxes (labeled `BASELINE_HEURISTIC`).
  - License plate detector uses candidate bumper contours + OCR fallback, or custom lightweight plate weights when supplied.
  - Clear UI indicators showing whether detections originate from "COCO Baseline + Spatial Heuristics" vs "Dedicated Custom Model".
  - If a model is missing (e.g. seatbelt classifier), the system logs *"Seatbelt detection model not configured"* instead of fabricating outputs.

---

## 4. Phase-by-Phase Roadmap & Current Status

### Phase 0: YOLO26 Upgrade (COMPLETED)
- [x] Upgraded to `ultralytics>=8.4.0` supporting YOLO26.
- [x] Integrated `yolo26n.pt` into `models/` and updated `config.yaml`.
- [x] Verified NMS-free box parsing in `detector.py` and validated pipeline execution.

### Phase 1: Merged YOLO Dataset Construction (COMPLETED)
- [x] Consolidated 6 raw dataset archives into a unified 3-class YOLO dataset (`0: helmet`, `1: no_helmet`, `2: number_plate`).
- [x] Image byte deduplication (1,440 duplicate images filtered).
- [x] 80/20 train/validation split (3,213 train / 803 val images) preserving video sequence groups.
- [x] Generated `datasets/merged/data.yaml` and 12-sample annotated preview `datasets/merged/preview.jpg`.

### Phase 2: Google Colab Training Pipeline (COMPLETED)
- [x] Packaged training dataset into `datasets/merged.zip` via `scripts/zip_dataset.py`.
- [x] Created `notebooks/train_colab.ipynb` for T4 GPU training with automated Drive export for weights, confusion matrices, and metrics.

### Phase 3: Core Pipeline Code Fixes & Robustness (COMPLETED)
- [x] Custom model full-frame single-pass inference (`models/traffic_custom.pt`).
- [x] Spatial top-35% rider bounding box rule for helmet detection.
- [x] Targeted 10% padded plate crop OCR triggered exclusively for confirmed violation tracks.
- [x] Synchronized video-time cooldown calculation (`frame_idx / fps`).
- [x] ByteTrack integration with native Ultralytics tracker.
- [x] Eliminated all hardcoded metrics from Dashboard and Analytics API.
- [x] Rebranded notices to "Prototype violation notice" (`PROTO-`).

### Phase 4: Video Processing CLI & Hardware Benchmarking (COMPLETED)
- [x] End-to-end video inference CLI (`scripts/run_video.py`) with stop-line and traffic signal state controls.
- [x] Processed test video `data/videos/22.mp4` and generated 5 forensic detection snapshots.
- [x] Real hardware inference benchmark (`scripts/benchmark.py`) achieving ~23.9 FPS on CPU with YOLO26n.
- [x] Comprehensive documentation in `docs/MODEL_RESULTS.md` detailing dataset distributions and honest limitations.

