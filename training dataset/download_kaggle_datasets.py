"""
Download all Kaggle datasets for TrafficAI into training dataset/04_kaggle/.

One-time setup (Windows):
  1. pip install kaggle
  2. Kaggle website -> your profile picture -> Settings -> API -> "Create New Token".
     This downloads kaggle.json.
  3. Move kaggle.json to C:\\Users\\LENOVO\\.kaggle\\kaggle.json
  4. For each dataset, open its page once in the browser while signed in
     (some datasets ask you to accept their terms first).

Run (from inside the "training dataset" folder):
  python download_kaggle_datasets.py
  python download_kaggle_datasets.py helmet      # only one group
"""
import subprocess
import sys
from pathlib import Path

DATASETS = {
    "helmet": [
        "andrewmvd/helmet-detection",
        "pkdarabi/helmet",
        "abhi3022/helmet-detection-person-with-and-without-helmet",
    ],
    "plates": [
        "kedarsai/indian-license-plates-with-labels",
        "deepakat002/indian-vehicle-number-plate-yolo-annotation",
        "dataclusterlabs/indian-number-plates-dataset",
        "abtexp/synthetic-indian-license-plates",
    ],
    "vehicles": [
        "dataclusterlabs/indian-vehicle-dataset",
        "deepak242424/iitmhetra",
        "dataclusterlabs/autorickshaw-image-dataset-niche-vehicle-dataset",
        "asfakali2/iruvd-dataset-for-automatic-vehicle-detection",
    ],
}

ROOT = Path(__file__).resolve().parent / "04_kaggle"


def main() -> None:
    groups = sys.argv[1:] or list(DATASETS)
    failed = []
    for group in groups:
        for slug in DATASETS[group]:
            target = ROOT / group / slug.replace("/", "__")
            if target.exists() and any(target.iterdir()):
                print(f"[skip] {slug} already downloaded")
                continue
            target.mkdir(parents=True, exist_ok=True)
            print(f"[download] {slug} -> {target}")
            result = subprocess.run(
                ["kaggle", "datasets", "download", "-d", slug, "-p", str(target), "--unzip"]
            )
            if result.returncode != 0:
                failed.append(slug)
            else:
                (target / "SOURCE.txt").write_text(
                    f"https://www.kaggle.com/datasets/{slug}\n", encoding="utf-8"
                )
    print("\nDone.")
    if failed:
        print("These failed (open the page in a browser, accept terms, then re-run):")
        for slug in failed:
            print(f"  https://www.kaggle.com/datasets/{slug}")


if __name__ == "__main__":
    main()
