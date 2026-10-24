"""
zip_dataset.py — Compresses datasets/merged into datasets/merged.zip for Google Colab upload.
"""

import os
import sys
import zipfile
from pathlib import Path
from tqdm import tqdm


def zip_merged_dataset():
    repo_root = Path(__file__).resolve().parent.parent
    merged_dir = repo_root / "datasets" / "merged"
    output_zip = repo_root / "datasets" / "merged.zip"

    if not merged_dir.exists():
        print(f"Error: {merged_dir} does not exist. Run scripts/build_dataset.py first.")
        sys.exit(1)

    print(f"Collecting files from {merged_dir}...")
    files_to_zip = []
    for root, dirs, files in os.walk(merged_dir):
        for f in files:
            file_path = Path(root) / f
            # Skip any existing zip or preview inside
            if file_path.suffix == ".zip":
                continue
            arcname = file_path.relative_to(merged_dir)
            files_to_zip.append((file_path, arcname))

    print(f"Total files to compress: {len(files_to_zip)}")
    print(f"Creating {output_zip}...")

    # Write zip file with standard DEFLATED compression
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for file_path, arcname in tqdm(files_to_zip, desc="Zipping dataset"):
            z.write(file_path, str(arcname))

    size_mb = output_zip.stat().st_size / (1024 * 1024)
    print(f"\nSuccessfully created: {output_zip}")
    print(f"Archive size: {size_mb:.2f} MB")
    print("You can now upload datasets/merged.zip to Google Drive (MyDrive/merged.zip).")


if __name__ == "__main__":
    zip_merged_dataset()
