# models/ — trained model weights (.pt files)

| File | Model | Classes | Status |
|---|---|---|---|
| `wrong_way_yolo.pt` | **YOLO Wrong-Way Driving Detector** (Ultralytics YOLO, fine-tuned on 1,228 images, 640 px) | 0: right-side, 1: wrong-side | **In use on citizen photo uploads.** Validation: mAP50 0.969, Precision 0.936, Recall 0.940 |
| `wrong_way_lstm.pt` | **Bi-LSTM + Temporal Attention** (Colab T4, 20 epochs, seed 42) — Trajectory Classifier | 0: COMPLIANT_FLOW, 1: WRONG_WAY | Extra check inside wrong-way rule. Trained on synthetic trajectories; validation F1: 1.000 |
| `triple_riding_best.pt` | YOLOv8n, team's triple-riding model | 5 classes; only MORE_THAN_TWO_PERSONS / TRIPLE_RIDING / USING_MOBILE are used | **In use.** Triple riding mAP50 0.923 |
| `seatbelt_best.pt` | YOLO26n, 50 epochs | seatbelt, no_seatbelt | Experimental (no-seatbelt F1 0.56) |
| `yolo26n.pt` | YOLO26 nano, pretrained on COCO (Ultralytics, 2026) | car, motorcycle, bus, truck, person, traffic light (and other COCO classes) | In use |
| `traffic_custom.pt` | **YOLO26n fine-tuned on our merged dataset** (Colab T4, 40 epochs, 640 px, seed 0) — the live helmet / plate model | 0 helmet, 1 no_helmet, 2 number_plate | **In use.** Validation: P 0.865, R 0.889, F1 0.877, mAP50 0.922 |
| `traffic_custom_yolo26n.pt` | Same file as `traffic_custom.pt` (kept under its model name) | same | Copy |
| `traffic_custom_yolov8n.pt` | YOLOv8n fine-tuned with the identical settings — the Review 1 baseline | same | For comparison. Validation: P 0.865, R 0.835, F1 0.850, mAP50 0.909 |
| `old/traffic_custom_cpu_yolov8n_5epochs.pt` | Earlier quick CPU run (YOLOv8n, 5 epochs, 512 px) | same | Replaced |
| `yolov8n.pt` | YOLOv8 nano, pretrained on COCO (Ultralytics, 2023) — the Review 1 plan | same as above | Kept for comparison: switch on Live cameras or Settings; compared on the Evaluation page |

**More models:** the dashboard's Models page lists 13 detection models (YOLO26, YOLO11, YOLOv10, YOLOv8, YOLOv5 in nano / small / medium). Their `.pt` files are downloaded here automatically the first time one is compared or put live. The catalogue is in `algorithms/detection/model_registry.py`.

EasyOCR (plate text reading) downloads its own weights automatically on first run to `~/.EasyOCR/`.

Paths are set in `config.yaml` under `detection:`. Large `.pt` files should not be committed to GitHub if they are over 100 MB.

**Training outputs** (from Google Drive `traffic_model_output`): `traffic_model_output/` holds the raw export; results files are
copied to `data/evaluation_results_yolo26n.json`, `data/evaluation_results_yolov8n.json`, and curves / confusion matrices to
`data/training_results/<model>/`. The Evaluation page shows them side by side.
