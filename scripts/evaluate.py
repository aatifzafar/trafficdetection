import os
import sys
import json
import logging
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.config.settings import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate")


def evaluate_dataset(dataset_path: str = None):
    """
    Evaluates detector and rules against an annotated dataset (COCO / YOLO format).
    Adheres strictly to the project engineering rule: DO NOT FABRICATE NUMBERS.
    If dataset path is missing or annotations are not found, reports 'Evaluation not performed'.
    """
    if not dataset_path or not Path(dataset_path).exists():
        msg = (
            "Evaluation not performed: No annotated validation dataset found.\n"
            "To perform an empirical evaluation with Precision, Recall, F1, and mAP@0.5,\n"
            "supply an annotated dataset YAML (e.g., IDD, RideSafe-400, or UFPR-ALPR) using:\n"
            "  python scripts/evaluate.py --dataset path/to/dataset.yaml"
        )
        print("\n" + "=" * 60)
        print("          SURVEILLANCE ACCURACY EVALUATION          ")
        print("=" * 60)
        print(msg)
        print("=" * 60 + "\n")
        return {"status": "Evaluation not performed", "reason": "No annotated validation dataset provided"}

    # If dataset exists, run Ultralytics validation
    logger.info(f"Evaluating model '{settings.MODEL_TRAFFIC}' on dataset '{dataset_path}'...")
    try:
        from ultralytics import YOLO
        model = YOLO(settings.MODEL_TRAFFIC)
        metrics = model.val(data=dataset_path, device=settings.DEVICE)

        results = {
            "status": "Evaluation Completed",
            "model": settings.MODEL_TRAFFIC,
            "dataset": dataset_path,
            "precision": round(float(metrics.box.mp), 4),
            "recall": round(float(metrics.box.mr), 4),
            "f1": round(2 * (metrics.box.mp * metrics.box.mr) / max(1e-6, (metrics.box.mp + metrics.box.mr)), 4),
            "map50": round(float(metrics.box.map50), 4),
            "map50_95": round(float(metrics.box.map), 4),
        }

        out_path = Path("data/evaluation_results.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        print("\n" + "=" * 60)
        print("          SURVEILLANCE ACCURACY EVALUATION          ")
        print("=" * 60)
        print(f"Dataset      : {dataset_path}")
        print(f"Precision    : {results['precision']:.4f}")
        print(f"Recall       : {results['recall']:.4f}")
        print(f"F1-Score     : {results['f1']:.4f}")
        print(f"mAP@0.5      : {results['map50']:.4f}")
        print(f"mAP@0.5:0.95 : {results['map50_95']:.4f}")
        print("=" * 60)
        print(f"Results saved to: {out_path}\n")
        return results

    except Exception as e:
        logger.error(f"Error during validation: {e}")
        return {"status": "Error", "detail": str(e)}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Traffic Detection & Rules against ground truth")
    parser.add_argument("--dataset", type=str, default=None, help="Path to dataset YAML file")
    args = parser.parse_args()

    evaluate_dataset(args.dataset)
