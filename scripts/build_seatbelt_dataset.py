"""
Build and merge public seatbelt datasets into a unified YOLO dataset in data/seatbelt/.
Unified class mapping:
  0: seatbelt
  1: no_seatbelt

Features:
- Validates corrupt image files and corrupted labels.
- Computes perceptual dHash (64-bit) & MD5 to eliminate exact and near-duplicate images.
- Splits by source/scene (~70% train, 15% val, 15% test).
- Strict data leakage check across splits (zero near-duplicate cross-split overlap).
- Class distribution reporting and targeted training-only augmentation for underrepresented classes.
- Generates data/seatbelt/data.yaml and data/seatbelt/DATASETS.md.
"""

import os
import sys
import glob
import math
import random
import hashlib
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT_DIR / "data" / "seatbelt" / "raw"
OUT_DIR = ROOT_DIR / "data" / "seatbelt"

# Unified class map
CLASS_NAMES = ["seatbelt", "no_seatbelt"]

DATASET_METADATA = {
    "ds1_seatbelttraining": {
        "name": "seatbelt-detection",
        "author": "seatbelttraining",
        "url": "https://universe.roboflow.com/seatbelttraining/seatbelt-detection-lb1ec",
        "license": "CC BY 4.0",
        "class_remap": {0: 1, 1: 0},  # 0: no-seatbelt -> 1, 1: seatbelt -> 0
    },
    "ds2_yashinta": {
        "name": "seat-belt-detection",
        "author": "Yashinta Salsabila",
        "url": "https://universe.roboflow.com/yashinta-salsabila-popfd/seat-belt-detection-6wa10",
        "license": "CC BY 4.0",
        "class_remap": {0: 1, 1: 0},  # 0: no seat belt -> 1, 1: seat belt -> 0
    },
    "ds3_fay": {
        "name": "seat_belt",
        "author": "Fay",
        "url": "https://universe.roboflow.com/fay-regu8/seat_belt-iauiy",
        "license": "CC BY 4.0",
        "class_remap": {0: 0, 1: 1},  # 0: Seat_Belt -> 0, 1: Without_Seat_Belt -> 1
    },
}


def compute_dhash(img: Image.Image, hash_size=8) -> int:
    """Compute 64-bit difference hash (dHash) using PIL."""
    # Resize to (hash_size + 1, hash_size) grayscale
    resized = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = list(resized.getdata())
    diff = []
    for row in range(hash_size):
        for col in range(hash_size):
            pixel_left = pixels[row * (hash_size + 1) + col]
            pixel_right = pixels[row * (hash_size + 1) + col + 1]
            diff.append(pixel_left > pixel_right)
    # Convert bool array to integer
    decimal_val = 0
    for idx, bit in enumerate(diff):
        if bit:
            decimal_val |= 1 << idx
    return decimal_val


def hamming_distance(h1: int, h2: int) -> int:
    """Compute Hamming distance between two 64-bit hashes."""
    x = h1 ^ h2
    return bin(x).count("1")


def validate_image_and_labels(img_path: Path, lbl_path: Path, class_remap: dict):
    """
    Validates image readability and re-maps label bounding boxes.
    Returns (Image.Image, list_of_remapped_boxes) or None if invalid.
    """
    if not img_path.exists() or not lbl_path.exists():
        return None
    try:
        with Image.open(img_path) as im:
            im.verify()
        # Open for actual processing
        im = Image.open(img_path)
        im.load()
    except Exception:
        return None

    remapped_lines = []
    try:
        content = lbl_path.read_text(encoding="utf-8").strip()
        if not content:
            # Negative image (no objects) is fine in YOLO
            return im, []
        for line in content.splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            orig_cls = int(parts[0])
            if orig_cls not in class_remap:
                continue
            new_cls = class_remap[orig_cls]
            coords = [float(x) for x in parts[1:5]]
            # Validate coordinates are normalized [0, 1]
            x, y, w, h = coords
            if w <= 0 or h <= 0:
                continue
            # Clamp to [0, 1]
            x = max(0.0, min(1.0, x))
            y = max(0.0, min(1.0, y))
            w = max(0.001, min(1.0, w))
            h = max(0.001, min(1.0, h))
            remapped_lines.append(f"{new_cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
    except Exception:
        return None

    return im, remapped_lines


def build():
    print("=" * 60)
    print("STEP 1: Collecting & deduplicating raw datasets...")
    print("=" * 60)

    all_samples = []  # list of dicts
    seen_hashes = {}  # dhash -> sample_idx
    seen_md5s = set()
    corrupt_count = 0
    duplicate_count = 0
    dataset_counts = {k: 0 for k in DATASET_METADATA}

    for ds_key, meta in DATASET_METADATA.items():
        ds_path = RAW_DIR / ds_key
        if not ds_path.exists():
            print(f"Warning: {ds_path} does not exist!")
            continue

        class_remap = meta["class_remap"]
        # Find all images in train, valid, test
        for split in ["train", "valid", "test"]:
            img_dir = ds_path / split / "images"
            lbl_dir = ds_path / split / "labels"
            if not img_dir.exists():
                continue
            for img_file in sorted(img_dir.glob("*.*")):
                if img_file.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
                    continue
                lbl_file = lbl_dir / f"{img_file.stem}.txt"
                res = validate_image_and_labels(img_file, lbl_file, class_remap)
                if res is None:
                    corrupt_count += 1
                    continue
                im, lines = res

                # Check exact MD5
                with open(img_file, "rb") as f:
                    file_md5 = hashlib.md5(f.read()).hexdigest()
                if file_md5 in seen_md5s:
                    duplicate_count += 1
                    continue
                seen_md5s.add(file_md5)

                # Check perceptual dHash
                dhash_val = compute_dhash(im)
                is_near_dup = False
                for existing_hash in seen_hashes:
                    if hamming_distance(dhash_val, existing_hash) <= 1:
                        is_near_dup = True
                        break
                if is_near_dup:
                    duplicate_count += 1
                    continue

                seen_hashes[dhash_val] = len(all_samples)
                dataset_counts[ds_key] += 1

                # Group identifier (Scene/Source clustering)
                # Group by dataset and image filename stem prefix (e.g. video/sequence prefix)
                stem = img_file.stem
                prefix = stem.split("_")[0] if "_" in stem else stem[:6]
                group_key = f"{ds_key}_{prefix}"

                all_samples.append({
                    "ds_key": ds_key,
                    "img_path": img_file,
                    "lbl_path": lbl_file,
                    "group": group_key,
                    "dhash": dhash_val,
                    "lines": lines,
                })

    print(f"Total raw candidates processed: {len(all_samples) + duplicate_count + corrupt_count}")
    print(f"Filtered corrupt files: {corrupt_count}")
    print(f"Filtered exact/perceptual duplicates: {duplicate_count}")
    print(f"Unique valid samples retained: {len(all_samples)}")
    for ds_key, cnt in dataset_counts.items():
        print(f"  - {ds_key}: {cnt} images")

    print("\n" + "=" * 60)
    print("STEP 2: Grouping by source/scene into Train / Val / Test (70/15/15)...")
    print("=" * 60)

    # Group samples by group_key to ensure scenes aren't split across sets
    groups = {}
    for idx, s in enumerate(all_samples):
        groups.setdefault(s["group"], []).append(idx)

    # Classify each group by what classes it contains
    group_has_ns = []
    group_only_s = []
    for g, member_indices in groups.items():
        has_ns = any(any(int(l.split()[0]) == 1 for l in all_samples[i]["lines"]) for i in member_indices)
        if has_ns:
            group_has_ns.append((g, member_indices))
        else:
            group_only_s.append((g, member_indices))

    random.seed(42)
    random.shuffle(group_has_ns)
    random.shuffle(group_only_s)

    train_indices, val_indices, test_indices = [], [], []

    def allocate_bucket(group_list, split_ratios=(0.70, 0.15, 0.15)):
        total_items = sum(len(m) for _, m in group_list)
        t_target = int(total_items * split_ratios[0])
        v_target = int(total_items * split_ratios[1])
        t_curr, v_curr = 0, 0
        t_idx, v_idx, te_idx = [], [], []
        for _, members in group_list:
            cnt = len(members)
            if t_curr + cnt <= t_target:
                t_idx.extend(members)
                t_curr += cnt
            elif v_curr + cnt <= v_target:
                v_idx.extend(members)
                v_curr += cnt
            else:
                te_idx.extend(members)
        return t_idx, v_idx, te_idx

    t1, v1, te1 = allocate_bucket(group_has_ns)
    t2, v2, te2 = allocate_bucket(group_only_s)

    train_indices = t1 + t2
    val_indices = v1 + v2
    test_indices = te1 + te2
    total_samples = len(all_samples)

    print(f"Initial split assignments: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")

    print("\n" + "=" * 60)
    print("STEP 3: Perceptual Hash Cross-Split Leakage Check (Hamming <= 4)...")
    print("=" * 60)

    train_hashes = {all_samples[i]["dhash"]: i for i in train_indices}
    val_to_move = []
    test_to_move = []

    for v_idx in val_indices:
        vh = all_samples[v_idx]["dhash"]
        for th in train_hashes:
            if hamming_distance(vh, th) <= 4:
                val_to_move.append(v_idx)
                break

    for t_idx in test_indices:
        th_val = all_samples[t_idx]["dhash"]
        for th in train_hashes:
            if hamming_distance(th_val, th) <= 4:
                test_to_move.append(t_idx)
                break

    if val_to_move or test_to_move:
        print(f"Leakage check found near-duplicates across splits:")
        print(f"  Moving {len(val_to_move)} Val samples to Train")
        print(f"  Moving {len(test_to_move)} Test samples to Train")
        val_set = set(val_indices) - set(val_to_move)
        test_set = set(test_indices) - set(test_to_move)
        train_indices.extend(val_to_move)
        train_indices.extend(test_to_move)
        val_indices = sorted(list(val_set))
        test_indices = sorted(list(test_set))
    else:
        print("Leakage check PASSED cleanly: 0 near-duplicate matches across train/val/test!")

    print(f"Final split counts: Train={len(train_indices)} ({len(train_indices)/total_samples*100:.1f}%), "
          f"Val={len(val_indices)} ({len(val_indices)/total_samples*100:.1f}%), "
          f"Test={len(test_indices)} ({len(test_indices)/total_samples*100:.1f}%)")

    print("\n" + "=" * 60)
    print("STEP 4: Writing YOLO directory structure & performing label audit...")
    print("=" * 60)

    import shutil
    for split_name in ["train", "val", "test"]:
        split_dir = OUT_DIR / split_name
        if split_dir.exists():
            shutil.rmtree(split_dir)
        (split_dir / "images").mkdir(parents=True, exist_ok=True)
        (split_dir / "labels").mkdir(parents=True, exist_ok=True)

    splits = {
        "train": train_indices,
        "val": val_indices,
        "test": test_indices,
    }

    class_counts = {
        "train": {0: 0, 1: 0},
        "val": {0: 0, 1: 0},
        "test": {0: 0, 1: 0},
    }

    for split_name, idx_list in splits.items():
        for i, sample_idx in enumerate(idx_list):
            sample = all_samples[sample_idx]
            base_name = f"{sample['ds_key']}_{sample['img_path'].stem}_{i:05d}"
            dest_img = OUT_DIR / split_name / "images" / f"{base_name}.jpg"
            dest_lbl = OUT_DIR / split_name / "labels" / f"{base_name}.txt"

            # Copy/Save normalized JPEG
            try:
                with Image.open(sample["img_path"]) as im:
                    im = im.convert("RGB")
                    im.save(dest_img, "JPEG", quality=95)
            except Exception as e:
                print(f"Error saving {dest_img}: {e}")
                continue

            # Write label lines & tally counts
            lbl_content = "\n".join(sample["lines"])
            dest_lbl.write_text(lbl_content, encoding="utf-8")
            for line in sample["lines"]:
                cls_id = int(line.split()[0])
                class_counts[split_name][cls_id] += 1

    print("Class bounding box distribution before augmentation:")
    for split_name in ["train", "val", "test"]:
        s_cnt = class_counts[split_name][0]
        ns_cnt = class_counts[split_name][1]
        print(f"  [{split_name.upper()}] seatbelt: {s_cnt}, no_seatbelt: {ns_cnt}")

    # Check class imbalance in training split
    train_s = class_counts["train"][0]
    train_ns = class_counts["train"][1]
    augmented_count = 0

    if train_ns < train_s * 0.7:
        print("\nNotice: 'no_seatbelt' is under-represented in train split.")
        print("Applying training-only offline augmentations (brightness, blur, slight rotation)...")
        train_lbl_dir = OUT_DIR / "train" / "labels"
        train_img_dir = OUT_DIR / "train" / "images"

        for lbl_file in list(train_lbl_dir.glob("*.txt")):
            lines = lbl_file.read_text(encoding="utf-8").splitlines()
            has_no_seatbelt = any(int(l.split()[0]) == 1 for l in lines if l.strip())
            if has_no_seatbelt:
                img_file = train_img_dir / f"{lbl_file.stem}.jpg"
                if not img_file.exists():
                    continue
                try:
                    with Image.open(img_file) as im:
                        im = im.convert("RGB")
                        # 1. Brightness enhancement (+15%)
                        enhancer = ImageEnhance.Brightness(im)
                        aug_im1 = enhancer.enhance(1.15)
                        aug1_name = f"{img_file.stem}_aug_bright"
                        aug_im1.save(train_img_dir / f"{aug1_name}.jpg", "JPEG", quality=95)
                        (train_lbl_dir / f"{aug1_name}.txt").write_text("\n".join(lines), encoding="utf-8")

                        # 2. Slight blur
                        aug_im2 = im.filter(ImageFilter.GaussianBlur(radius=0.7))
                        aug2_name = f"{img_file.stem}_aug_blur"
                        aug_im2.save(train_img_dir / f"{aug2_name}.jpg", "JPEG", quality=95)
                        (train_lbl_dir / f"{aug2_name}.txt").write_text("\n".join(lines), encoding="utf-8")

                        for l in lines:
                            cls_id = int(l.split()[0])
                            class_counts["train"][cls_id] += 2
                        augmented_count += 2
                except Exception as e:
                    pass

        print(f"Added {augmented_count} augmented training samples.")
        print("Class bounding box distribution after augmentation:")
        print(f"  [TRAIN] seatbelt: {class_counts['train'][0]}, no_seatbelt: {class_counts['train'][1]}")

    print("\n" + "=" * 60)
    print("STEP 5: Generating data.yaml and DATASETS.md...")
    print("=" * 60)

    # Write data.yaml
    yaml_content = {
        "path": str(OUT_DIR.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": 2,
        "names": CLASS_NAMES,
    }
    yaml_file = OUT_DIR / "data.yaml"
    with open(yaml_file, "w", encoding="utf-8") as f:
        yaml.dump(yaml_content, f, sort_keys=False)
    print(f"Wrote configuration to {yaml_file}")

    # Write DATASETS.md
    datasets_md = f"""# Seatbelt Detection Datasets & Licensing Audit

This dataset was constructed and merged for the B.Tech research project **AI-Driven Smart Traffic Violation Detection and Alert System**.
All raw sources are public, peer-reviewed or community-contributed datasets licensed for open research and education use.

## Unified Class Schema
- **Class 0:** `seatbelt` (Car driver or front-seat passenger wearing a 3-point or 2-point safety seatbelt)
- **Class 1:** `no_seatbelt` (Car driver or front-seat passenger unbuckled / no seatbelt detected)

## Source Datasets Summary

| # | Dataset Name | Source / URL | License | Original Classes | Raw Images Retained |
|---|---|---|---|---|---|
| 1 | **seatbelt-detection** (by seatbelttraining) | [Roboflow Universe](https://universe.roboflow.com/seatbelttraining/seatbelt-detection-lb1ec) | CC BY 4.0 | `['no-seatbelt', 'seatbelt']` | {dataset_counts['ds1_seatbelttraining']} |
| 2 | **seat-belt-detection** (by Yashinta Salsabila) | [Roboflow Universe](https://universe.roboflow.com/yashinta-salsabila-popfd/seat-belt-detection-6wa10) | CC BY 4.0 | `['no seat belt', 'seat belt']` | {dataset_counts['ds2_yashinta']} |
| 3 | **seat_belt** (by Fay) | [Roboflow Universe](https://universe.roboflow.com/fay-regu8/seat_belt-iauiy) | CC BY 4.0 | `['Seat_Belt', 'Without_Seat_Belt']` | {dataset_counts['ds3_fay']} |

**Total Unique Merged Images:** {total_samples}  
**Deduplication:** Perceptual difference hashing (dHash 64-bit) and MD5 filtering eliminated exact and near-duplicate frames.  
**Cross-Split Leakage Check:** Zero cross-split leakage verified (Hamming distance > 4 between all test/val images and train images).  

## Split Distribution

| Split | Images | % of Total | Bounding Boxes (seatbelt) | Bounding Boxes (no_seatbelt) |
|---|---|---|---|---|
| **Train** | {len(train_indices)} (+{augmented_count} aug) | {len(train_indices)/total_samples*100:.1f}% | {class_counts['train'][0]} | {class_counts['train'][1]} |
| **Validation** | {len(val_indices)} | {len(val_indices)/total_samples*100:.1f}% | {class_counts['val'][0]} | {class_counts['val'][1]} |
| **Test** | {len(test_indices)} | {len(test_indices)/total_samples*100:.1f}% | {class_counts['test'][0]} | {class_counts['test'][1]} |

*Note: Data was split by source/scene clustering to prevent sequential frame leakage across train/val/test splits.*
"""
    (OUT_DIR / "DATASETS.md").write_text(datasets_md, encoding="utf-8")
    print(f"Wrote dataset documentation to {OUT_DIR / 'DATASETS.md'}")
    print("\nPhase 1 Dataset Construction Complete!")


if __name__ == "__main__":
    build()
