# evaluation/ — measuring how well the system works

| File | Command | What it measures |
|---|---|---|
| `evaluate.py` | `python evaluation/evaluate.py --dataset datasets/merged/data.yaml` | Precision, recall, F1 and mAP of the model on the test split |
| `compare_models.py` | `python evaluation/compare_models.py --video data/videos/traffic_junction_cctv.mp4 --frames 150 --models yolo26n yolo11n yolov8n` | **YOLO26n vs YOLOv8n** on the same video: detection time, pipeline FPS, objects found, confidence, vehicles tracked, violations flagged, and a sample frame from each (also a button on the Evaluation page) |
| `benchmark.py` | `python evaluation/benchmark.py --model all --frames 50 --size 640` | Speed (FPS / latency) of detection + tracking on this computer, for each model |
| `scenario_generator.py` | `python evaluation/scenario_generator.py` | Makes synthetic test videos (red light, wrong way, triple riding) |
| `test_pipeline.py` | `python evaluation/test_pipeline.py` | End-to-end run of the pipeline on a sample video |

Unit tests for each algorithm are in `tests/` (`python -m pytest tests`).
