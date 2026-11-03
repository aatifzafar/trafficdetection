# training/ — building the dataset and training the custom model

Run in this order from the project root:

| Step | Command | What it does |
|---|---|---|
| 1 | put the Kaggle zips in `datasets/raw/` (see `datasets/raw/README.md`) | Raw datasets |
| 2 | `python training/build_dataset.py` | Merges, de-duplicates and converts them to YOLO format in `datasets/merged/` (classes helmet / no_helmet / number_plate) |
| 3 | `python training/zip_dataset.py` | Makes `datasets/merged.zip` to upload to Google Drive |
| 4 | open `training/train_colab.ipynb` in Google Colab (T4 GPU) | Trains **YOLO26n and YOLOv8n** on the same data and settings; saves each model's `best.pt` and metrics to Drive (`traffic_model_output/yolo26n/`, `.../yolov8n/`) |
| 5 | copy `yolo26n/best.pt` → `models/traffic_custom.pt` | The pipeline picks it up automatically |
| 6 | copy `evaluation_results_yolo26n.json` and `evaluation_results_yolov8n.json` → `data/` | The Evaluation page shows both models' precision, recall, F1 and mAP side by side |

If you only have a `results.csv`: `python training/parse_training_results.py --csv results.csv --model yolov8n.pt` writes `data/evaluation_results_yolov8n.json` (overall numbers only).

`train.py` is the same training as the notebook, for a local GPU.
