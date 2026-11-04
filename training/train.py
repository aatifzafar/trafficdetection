import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from training.parse_training_results import parse_results_csv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train")


def train_model(
    data_yaml: str = "datasets/merged/data.yaml",
    base_model: str = "models/yolov8n.pt",
    epochs: int = 5,
    batch_size: int = 16,
    img_size: int = 512,
    workers: int = 4,
    project_name: str = "runs/detect",
    run_name: str = "traffic_custom",
):
    """
    Fine-tunes YOLO on custom traffic violation dataset (helmet, no_helmet, number_plate)
    and automatically deploys the resulting weights and metrics.
    """
    yaml_path = Path(data_yaml)
    if not yaml_path.is_absolute():
        yaml_path = PROJECT_ROOT / yaml_path

    if not yaml_path.exists():
        logger.error(f"Dataset config not found at: {yaml_path}")
        return False

    from ultralytics import YOLO

    device = "0" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device.upper()} (CUDA: {torch.cuda.is_available()})")
    logger.info(f"Loading base model: {base_model}")
    model = YOLO(base_model)

    logger.info(f"Starting training on {yaml_path}")
    logger.info(f"Params: epochs={epochs}, batch={batch_size}, imgsz={img_size}, workers={workers}")

    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        workers=workers,
        project=project_name,
        name=run_name,
        exist_ok=True,
        verbose=True,
    )

    save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else PROJECT_ROOT / project_name / run_name
    best_weights = save_dir / "weights" / "best.pt"
    results_csv = save_dir / "results.csv"

    # Deploy best weights to models/traffic_custom.pt
    dest_weights = PROJECT_ROOT / "models" / "traffic_custom.pt"
    if best_weights.exists():
        dest_weights.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_weights, dest_weights)
        logger.info(f"✅ Deployed best weights -> {dest_weights}")
    else:
        last_weights = save_dir / "weights" / "last.pt"
        if last_weights.exists():
            shutil.copy2(last_weights, dest_weights)
            logger.info(f"✅ Deployed last weights -> {dest_weights}")

    # Generate evaluation metrics JSON for the dashboard
    if results_csv.exists():
        out_json = PROJECT_ROOT / "data" / "evaluation_results_yolo26n.json"
        parse_results_csv(results_csv, out_json, model_name=Path(base_model).name)
        # Also copy for yolov8n baseline if applicable
        out_v8 = PROJECT_ROOT / "data" / "evaluation_results_yolov8n.json"
        shutil.copy2(out_json, out_v8)
        logger.info(f"✅ Exported evaluation metrics -> {out_json}")

    logger.info("🎉 Local training and deployment completed successfully!")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train custom YOLO model for traffic violation detection")
    parser.add_argument("--data", type=str, default="datasets/merged/data.yaml", help="Path to data.yaml dataset config")
    parser.add_argument("--model", type=str, default="models/yolov8n.pt", help="Base model checkpoint")
    parser.add_argument("--epochs", type=int, default=5, help="Epoch count (default: 5 for fast local run)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--imgsz", type=int, default=512, help="Image size (default: 512)")
    parser.add_argument("--workers", type=int, default=4, help="Data loader worker count (default: 4)")
    args = parser.parse_args()

    train_model(
        data_yaml=args.data,
        base_model=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.imgsz,
        workers=args.workers,
    )
