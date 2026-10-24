# Training Dataset — TrafficAI

Collected: 24 Oct 2026. All data below is for academic (B.Tech Design Project) use only.
Cite each source in your report. Do not redistribute publicly without checking each source's licence.

## What is already downloaded (in this folder)

| Folder | What it is | Size | Format | Use in the project | Source (downloaded from) | Licence |
|---|---|---|---|---|---|---|
| `01_indian_number_plates/` | 602 photos of Indian vehicles from 35 states/UTs, each with the plate box **and the plate text** | 602 images (482 train / 120 val) | Original Pascal VOC XML + converted YOLO (`yolo/`, class 0 = `number_plate`) + `plate_text_ground_truth.csv` | Train the plate detector; test OCR accuracy using the plate text column | https://github.com/ChSatyaSavith/NumberPlate_Annotated_Dataset (cleaned copy of https://www.kaggle.com/datasets/saisirishan/indian-vehicle-dataset) | None stated by the author — academic use, cite the source |
| `02_helmet_riders/` | Motorcycle riders with and without helmets, hand-labelled with LabelImg | 50 images, 155 boxes (102 helmet, 53 no_helmet) | Original VOC + YOLO, converted split in `yolo/` (0 = `helmet`, 1 = `no_helmet`) | Starter set for the helmet model; far too small on its own — combine with the Kaggle sets below | https://github.com/naveena-as/helmet-detection-yolo | None stated by the author — academic use, cite the source |
| `03_traffic_light_state/` | Cropped traffic-light images labelled red / yellow / green | 1,488 images (1,190 training, 298 test) | Folders per class (image classification) | Train a small classifier for the signal state in the red-light rule, instead of the manual dropdown | https://github.com/ShashankKumbhare/traffic-light-classifier | MIT (see `LICENSE_MIT.md`) |

### Known issues — fix before training
- **Plates:** a few boxes in the original source are misplaced (e.g. on the bumper, not the plate). Open `01_indian_number_plates/yolo` in CVAT or Label Studio and correct them; it takes about an hour for 602 images.
- **Plates:** images are parked cars photographed for listings, mostly front view in daylight. They do not look like CCTV footage, so the detector still needs your own CCTV-angle and night images.
- **Helmet:** 50 images is only a starter. The target in the plan is 3,000–5,000 labelled rider heads, about 30% at night.
- **Traffic lights:** these are close-up crops from non-Indian roads. Good for learning the red/green classifier; re-test on crops from your own junction video.

## Still to download (needs your Kaggle / Roboflow account)

Kaggle, Roboflow, Mendeley and IDD could not be downloaded from Claude's workspace; they need your login.
Run `download_kaggle_datasets.py` (instructions inside) to fetch all Kaggle sets into `04_kaggle/` in one go.

### Kaggle (automatic, via the script)

| Purpose | Dataset | Link |
|---|---|---|
| Helmet | Helmet Detection (764 images, 2 classes) | https://www.kaggle.com/datasets/andrewmvd/helmet-detection |
| Helmet + plate | HelmetViolations (YOLO-ready, helmet compliance + licence plates) | https://www.kaggle.com/datasets/pkdarabi/helmet |
| Helmet | Helmet Detection — person with and without helmet | https://www.kaggle.com/datasets/abhi3022/helmet-detection-person-with-and-without-helmet |
| Plate | Indian License Plates with Labels | https://www.kaggle.com/datasets/kedarsai/indian-license-plates-with-labels |
| Plate | Indian vehicle number plate YOLO annotation | https://www.kaggle.com/datasets/deepakat002/indian-vehicle-number-plate-yolo-annotation |
| Plate | Indian Number Plates Dataset (DataCluster Labs) | https://www.kaggle.com/datasets/dataclusterlabs/indian-number-plates-dataset |
| Plate OCR | Synthetic Indian License Plates | https://www.kaggle.com/datasets/abtexp/synthetic-indian-license-plates |
| Vehicles | Indian Vehicle Dataset (DataCluster Labs) | https://www.kaggle.com/datasets/dataclusterlabs/indian-vehicle-dataset |
| Vehicles | IITM-HeTra — vehicle detection in heterogeneous (Indian) traffic | https://www.kaggle.com/datasets/deepak242424/iitmhetra |
| Vehicles | Autorickshaw Image Dataset | https://www.kaggle.com/datasets/dataclusterlabs/autorickshaw-image-dataset-niche-vehicle-dataset |
| Vehicles | IRUVD — vehicle detection dataset | https://www.kaggle.com/datasets/asfakali2/iruvd-dataset-for-automatic-vehicle-detection |

Image counts and licences for most Kaggle sets are shown only after login; check each page's licence and note it here after downloading.

### Manual download (website login or registration)

| Purpose | Dataset | Link | How |
|---|---|---|---|
| Triple riding | triple riding detection (Roboflow Universe) | https://universe.roboflow.com/ai-lisrp/triple-riding-detection-ysta9 | Sign in → Download → format "YOLOv8" |
| Triple riding | triple riding detection 3 class | https://universe.roboflow.com/krokky/triple-riding-detection-3-class | Same |
| Helmet | Helmet Detection Project (Roboflow) | https://universe.roboflow.com/vath/helmet-detection-project-cxlvd | Same |
| Helmet | Dataset for the Detection of Helmet Use in Motorcycle Drivers (Mendeley Data) | https://data.mendeley.com/datasets/bmy35m25pw/1 | Download all files |
| Vehicles (India, main) | India Driving Dataset (IDD) — Bangalore/Hyderabad roads, autos, two-wheelers | https://idd.insaan.iiit.ac.in/ (info: https://cvit.iiit.ac.in/ncvpripg19/idd-challenge/index.html) | Free registration; download "IDD Detection" |
| Helmet (research) | AI City Challenge 2023, Track 5 (helmet violation) — see paper 06 in `ASSETS/` | https://www.aicitychallenge.org/ | Request access |

## Suggested final layout after you download everything
```
training dataset/
  01_indian_number_plates/      (done)
  02_helmet_riders/             (done, starter)
  03_traffic_light_state/       (done)
  04_kaggle/                    (script output)
  05_roboflow/                  (manual)
  06_idd/                       (manual)
  07_own_footage/               (your own CCTV-angle videos + labels — most important for the test set)
```
Keep the frozen test set (`07_own_footage/test`) separate and never train on it.
