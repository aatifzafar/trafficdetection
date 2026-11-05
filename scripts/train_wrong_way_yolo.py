"""
scripts/train_wrong_way_yolo.py
Production Training and Fine-Tuning Pipeline for YOLO Wrong-Way Driving Detection Models.

Supports YOLO26 and YOLOv8 architectures on custom annotated datasets.
Deploys best checkpoint to models/wrong_way_yolo.pt and generates metrics report.
"""

import os
import sys
import shutil
import argparse
import logging
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_wrong_way_yolo")


def train_wrong_way_yolo(
    data_yaml: str = "datasets/wrong_way_roboflow/data.yaml",
    base_model: str = "models/yolov8n.pt",
    epochs: int = 30,
    batch_size: int = 16,
    img_size: int = 640,
    device: str = None,
    workers: int = 4,
    patience: int = 10,
    project_name: str = "runs/detect",
    run_name: str = "wrong_way_yolo_detector",
):
    """
    Fine-tunes YOLO on wrong-way driving dataset (classes: right-side, wrong-side).
    Deploys best weights to models/wrong_way_yolo.pt and generates evaluation JSON.
    """
    yaml_path = Path(data_yaml)
    if not yaml_path.is_absolute():
        yaml_path = PROJECT_ROOT / yaml_path

    if not yaml_path.exists():
        logger.error(f"Dataset configuration not found at {yaml_path}")
        return False

    model_path = Path(base_model)
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path

    if not model_path.exists():
        if (PROJECT_ROOT / "models" / "yolov8n.pt").exists():
            model_path = PROJECT_ROOT / "models" / "yolov8n.pt"
        elif (PROJECT_ROOT / "models" / "yolo26n.pt").exists():
            model_path = PROJECT_ROOT / "models" / "yolo26n.pt"
        else:
            model_path = "yolov8n.pt"

    from ultralytics import YOLO

    if device is None:
        if torch.cuda.is_available():
            device = "0"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    logger.info("=" * 65)
    logger.info("       YOLO WRONG-WAY DRIVING DETECTION MODEL TRAINING        ")
    logger.info("=" * 65)
    logger.info(f"Dataset Config   : {yaml_path}")
    logger.info(f"Base Weights     : {model_path}")
    logger.info(f"Target Device    : {device.upper()}")
    logger.info(f"Hyperparameters  : epochs={epochs}, batch={batch_size}, imgsz={img_size}, workers={workers}")

    model = YOLO(str(model_path))

    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        workers=workers,
        patience=patience,
        project=project_name,
        name=run_name,
        exist_ok=True,
        plots=True,
        verbose=True,
    )

    save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else PROJECT_ROOT / project_name / run_name
    best_weights = save_dir / "weights" / "best.pt"
    dest_weights = PROJECT_ROOT / "models" / "wrong_way_yolo.pt"

    if best_weights.exists():
        dest_weights.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_weights, dest_weights)
        logger.info(f"✅ Deployed best checkpoint -> {dest_weights}")
    else:
        last_weights = save_dir / "weights" / "last.pt"
        if last_weights.exists():
            dest_weights.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(last_weights, dest_weights)
            logger.info(f"✅ Deployed last checkpoint -> {dest_weights}")

    # Archive training plots and confusion matrices
    out_results_dir = PROJECT_ROOT / "data" / "training_results" / "wrong_way_yolo"
    out_results_dir.mkdir(parents=True, exist_ok=True)

    artifacts = [
        "results.png",
        "results.csv",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "BoxF1_curve.png",
        "BoxPR_curve.png",
        "BoxP_curve.png",
        "BoxR_curve.png",
    ]
    for artifact in artifacts:
        src = save_dir / artifact
        if src.exists():
            shutil.copy2(src, out_results_dir / artifact)
            logger.info(f"Archived training artifact: {artifact}")

    # Run evaluation on test split and export metrics JSON
    try:
        from scripts.evaluate_wrong_way_yolo import evaluate_wrong_way_yolo
        evaluate_wrong_way_yolo(
            model_path=str(dest_weights),
            data_yaml=str(yaml_path),
            img_size=img_size,
            output_json=str(PROJECT_ROOT / "data" / "evaluation_results_wrong_way_yolo.json"),
        )
    except Exception as e:
        logger.warning(f"Note: Post-training evaluation export skipped: {e}")

    logger.info("🎉 Training and deployment completed successfully!")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train custom YOLO model for wrong-way driving detection")
    parser.add_argument("--data", type=str, default="datasets/wrong_way_roboflow/data.yaml", help="Path to data.yaml")
    parser.add_argument("--model", type=str, default="models/yolov8n.pt", help="Base model checkpoint (yolov8n / yolo26n)")
    parser.add_argument("--epochs", type=int, default=30, help="Epoch count (default: 30)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    parser.add_argument("--device", type=str, default=None, help="Device ('mps', '0', 'cpu')")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")

    args = parser.parse_args()
    train_wrong_way_yolo(
        data_yaml=args.data,
        base_model=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.imgsz,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
    )
