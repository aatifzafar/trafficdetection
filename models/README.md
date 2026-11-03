# models/ — trained model weights (.pt files)

| File | Model | Classes | Status |
|---|---|---|---|
| `yolo26n.pt` | YOLO26 nano, pretrained on COCO (Ultralytics, 2026) | car, motorcycle, bus, truck, person, traffic light (and other COCO classes) | In use |
| `traffic_custom.pt` | YOLO26n fine-tuned on our merged dataset (`datasets/merged/`) | 0 helmet, 1 no_helmet, 2 number_plate | **Not trained yet** — `training/train_colab.ipynb` trains YOLO26n and YOLOv8n; copy `yolo26n/best.pt` here with this name |
| `yolov8n.pt` | YOLOv8 nano, pretrained on COCO (Ultralytics, 2023) — the Review 1 plan | same as above | Kept for comparison: switch on Live cameras or Settings; compared on the Evaluation page |

**More models:** the dashboard's Models page lists 13 detection models (YOLO26, YOLO11, YOLOv10, YOLOv8, YOLOv5 in nano / small / medium). Their `.pt` files are downloaded here automatically the first time one is compared or put live. The catalogue is in `algorithms/detection/model_registry.py`.

EasyOCR (plate text reading) downloads its own weights automatically on first run to `~/.EasyOCR/`.

Paths are set in `config.yaml` under `detection:`. Large `.pt` files should not be committed to GitHub if they are over 100 MB.
