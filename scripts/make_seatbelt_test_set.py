"""
make_seatbelt_test_set.py — Creates a curated test set in data/seatbelt/test_samples/
from the held-out TEST split (strictly zero training images used).
Copies up to 3000 labeled NO-SEATBELT images (or all available if fewer),
plus a small 10-20% mix of WITH-SEATBELT images to evaluate false positives.
"""

import os
import sys
import json
import random
import shutil
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("make_test_set")


def make_test_set(
    test_dir: str = "data/seatbelt/test",
    output_dir: str = "data/seatbelt/test_samples",
    max_no_seatbelt: int = 3000,
    with_seatbelt_ratio: float = 0.15,
):
    test_path = PROJECT_ROOT / test_dir
    out_path = PROJECT_ROOT / output_dir

    if not test_path.exists():
        logger.error(f"Held-out test split not found at: {test_path}")
        return False

    out_path.mkdir(parents=True, exist_ok=True)

    img_dir = test_path / "images"
    lbl_dir = test_path / "labels"

    no_seatbelt_candidates = []
    seatbelt_only_candidates = []

    for lbl_file in sorted(lbl_dir.glob("*.txt")):
        stem = lbl_file.stem
        # find matching image
        img_candidates = list(img_dir.glob(f"{stem}.*"))
        if not img_candidates:
            continue
        img_file = img_candidates[0]

        lines = lbl_file.read_text(encoding="utf-8").splitlines()
        classes = set()
        for l in lines:
            if l.strip():
                classes.add(int(l.split()[0]))

        if 1 in classes:  # has no_seatbelt
            no_seatbelt_candidates.append((img_file, lbl_file, classes))
        elif 0 in classes:  # seatbelt only
            seatbelt_only_candidates.append((img_file, lbl_file, classes))

    total_ns = len(no_seatbelt_candidates)
    logger.info(f"Identified {total_ns} labeled NO-SEATBELT images in held-out test split.")
    logger.info(f"Identified {len(seatbelt_only_candidates)} WITH-SEATBELT images in held-out test split.")

    # Select all available up to max_no_seatbelt
    selected_ns = no_seatbelt_candidates[:max_no_seatbelt]
    if total_ns < max_no_seatbelt:
        logger.info(f"Fewer than {max_no_seatbelt} NO-SEATBELT images exist in test split. Using all {total_ns} available.")

    # Calculate mix of with-seatbelt images (10-20%)
    target_s_count = int(len(selected_ns) * with_seatbelt_ratio)
    random.seed(42)
    selected_s = random.sample(seatbelt_only_candidates, min(target_s_count, len(seatbelt_only_candidates)))

    logger.info(f"Selected {len(selected_s)} WITH-SEATBELT images ({len(selected_s)/len(selected_ns)*100:.1f}% mix) for false-positive validation.")

    manifest = []
    copied_count = 0

    for img_file, lbl_file, classes in selected_ns:
        dest_img = out_path / img_file.name
        shutil.copy2(img_file, dest_img)
        manifest.append({
            "filename": img_file.name,
            "ground_truth": "no_seatbelt" if 1 in classes else "seatbelt",
            "classes": sorted(list(classes)),
            "is_violation_ground_truth": True,
        })
        copied_count += 1

    for img_file, lbl_file, classes in selected_s:
        dest_img = out_path / img_file.name
        shutil.copy2(img_file, dest_img)
        manifest.append({
            "filename": img_file.name,
            "ground_truth": "seatbelt",
            "classes": sorted(list(classes)),
            "is_violation_ground_truth": False,
        })
        copied_count += 1

    manifest_path = out_path / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"✅ Successfully prepared {copied_count} test samples in {out_path} (manifest written to {manifest_path}).")
    return True


if __name__ == "__main__":
    make_test_set()

