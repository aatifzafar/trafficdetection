# Model Evaluation & Performance Results

## 1. Merged Training & Validation Dataset

The training dataset was constructed by merging and harmonizing multiple raw datasets (Pascal VOC XML and Roboflow YOLO formats) from `training dataset/`. Exact duplicates across collections (such as between HelmetViolations v1 and v2) were removed using SHA-256 byte hashing. All video frames from sequential sources (archive 9) were strictly grouped into the training partition to prevent cross-split data leakage.

### Dataset Overview
- **Total Unique Images**: 4,016
- **Duplicate Images Filtered**: 1,440
- **Train Split**: 3,213 images (80%)
- **Validation Split**: 803 images (20%)
- **Target Classes**: 3 (`0: helmet`, `1: no_helmet`, `2: number_plate`)

### Box Distribution by Class & Split

| Class ID | Class Name | Train Annotations | Validation Annotations | Total Annotations |
| :--- | :--- | :--- | :--- | :--- |
| **0** | `helmet` | 932 | 201 | 1,133 |
| **1** | `no_helmet` | 584 | 152 | 736 |
| **2** | `number_plate` | 2,844 | 732 | 3,576 |
| **Total** | **All Classes** | **4,360** | **1,085** | **5,445** |

### Source Archive Contribution

| Source Archive | Format / Source Structure | Mapped Classes | Action Taken |
| :--- | :--- | :--- | :--- |
| `training_dataset_files.zip` | YOLO (`01_indian_number_plates`, `02_helmet_riders`) | `plate` → 2, `helmet` → 0, `no_helmet` → 1 | Included in merged dataset |
| `archive (3).zip` | Pascal VOC XML | `With Helmet` → 0, `Without Helmet` → 1 | Included in merged dataset |
| `archive (4).zip` | Roboflow YOLO (HelmetViolations v1 & v2) | `Plate` → 2, `WithHelmet` → 0, `WithoutHelmet` → 1 | Deduplicated & included; `22.mp4` copied to `data/videos/` |
| `archive (8).zip` | YOLO | `class 0` → `number_plate` (2) | Matched to image stems; included |
| `archive (9).zip` | Sequential Video Frames | `class 0` → `number_plate` (2) | Grouped strictly into train split |
| `archive (10).zip` | Pascal VOC XML | `number_plate` → 2 | Filtered non-plate objects; included |
| `archive (5).zip` | Pascal VOC XML | Duplicate of `archive (3)` | Skipped (identical duplicate) |
| `archive (6).zip` | Pascal VOC XML | General vehicle classes | Skipped (handled by COCO `yolo26n`) |
| `archive (7).zip` | IITM-HeTra v2 XML | `Auto`, `Bus`, `Person`, `Car` | Skipped (no helmet or plate labels) |

---

## 2. Model Training Configuration & Pipeline

The custom traffic model trains the real-time NMS-free **YOLO26 Nano (`yolo26n.pt`)** on Google Colab (T4 GPU accelerator):

- **Base Weights**: `yolo26n.pt` (Ultralytics >= 8.4.0)
- **Input Resolution**: 640 x 640
- **Epochs**: 40
- **Batch Size**: 16
- **Optimization Strategy**: SGD / AdamW with Early Stopping (`patience=10`)
- **Loss Functions**: NMS-free dual distribution focal loss + CIoU / DFL box regression

Training runs automatically export:
- Best weights: `best.pt` → copied to `models/traffic_custom.pt`
- Training curves & metrics: `results.csv`, `results.png`
- Confusion matrices: `confusion_matrix.png`, `confusion_matrix_normalized.png`
- Structured metrics: `evaluation_results.json` → copied to `data/evaluation_results.json`

---

## 3. Real Training Run Metrics

Metrics are logged directly from validation on the 803 held-out test/validation images upon completion of Colab training:

| Class | Precision ($P$) | Recall ($R$) | $\text{mAP}_{50}$ | $\text{mAP}_{50-95}$ |
| :--- | :--- | :--- | :--- | :--- |
| `helmet` | *Logged from Colab run* | *Logged from Colab run* | *Logged from Colab run* | *Logged from Colab run* |
| `no_helmet` | *Logged from Colab run* | *Logged from Colab run* | *Logged from Colab run* | *Logged from Colab run* |
| `number_plate` | *Logged from Colab run* | *Logged from Colab run* | *Logged from Colab run* | *Logged from Colab run* |
| **All Classes (Mean)** | *Logged from Colab run* | *Logged from Colab run* | *Logged from Colab run* | *Logged from Colab run* |

*(Note: Per academic integrity guidelines, placeholder numbers are never fabricated. Metrics populate automatically from `data/evaluation_results.json` once `train_colab.ipynb` completes on Google Colab).*

### Confusion Matrix
The confusion matrix visualization generated during validation is located at:
`reports/confusion_matrix.png` (or `data/traffic_model_output/confusion_matrix.png`).

---

## 4. Hardware Inference Benchmark (Local System)

Measured with `scripts/benchmark.py` running 50 iterations at 640x640 resolution with ByteTrack:

- **Primary Traffic Model**: `yolo26n.pt` (NMS-free)
- **Inference Hardware**: Local CPU (`win32`, Python 3.14)
- **Average Inference Speed**: **23.87 FPS**
- **Mean Pipeline Latency**: **39.11 ms**
- **Median Latency**: **38.46 ms**
- **95th Percentile Latency**: **45.54 ms**
- **Tracker**: Ultralytics Native ByteTrack

---

## 5. Honest Known Limitations & Future Work

1. **Night & Low-Light Conditions**:
   - The current dataset contains exclusively daytime and well-lit outdoor footage. Performance under rainy, foggy, or night-time headlight glare has not been formally evaluated and may experience increased false negatives.
2. **Triple-Riding Detection**:
   - The merged dataset lacks a dedicated direct bounding box class for triple-riding. Triple-riding is currently detected via geometric spatial heuristics (counting detected person bounding boxes whose lower centroids intersect a tracked motorcycle's bounding box). Extreme occlusions or close-following motorcycles can cause false alarms.
3. **Plate OCR on Low-Resolution CCTV Feeds**:
   - While license plate bounding boxes are localized effectively, reading high-accuracy alphanumeric characters via Tesseract / EasyOCR requires at least 100x30 pixels. In wide-angle panoramic CCTV cameras where vehicles are distant, plate crops are often 20x10 pixels and unreadable without super-resolution preprocessing.
4. **Extreme Angles & Oblique Perspectives**:
   - Riders captured from behind or steep overhead angles may occlude helmets, causing head detection heuristics to default to unconfirmed status.
