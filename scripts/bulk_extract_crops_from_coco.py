#!/usr/bin/env python3

"""
bulk_extract_crops.py

Iterates through a raw_data directory and runs extract_crops_from_coco.py
twice per trial:

1. Using the highest-numbered instances_train_v*.json
2. Using incorrect_predictions.json

Expected folder structure:

raw_data/
├── trial_001/
│   ├── val_annotations/
│   │   ├── instances_train.json
│   │   ├── instances_train_v1.json
│   │   ├── instances_train_v2.json
│   │   └── incorrect_predictions.json
│   └── images/
├── trial_002/
│   └── ...

Usage:
    python bulk_extract_crops.py --raw-data-path ./raw_data
"""

import argparse
import re
import subprocess
from pathlib import Path

output_dir="/scratch/alpine/maha7624/AMC/training/extracted_crops_2023"
def find_highest_versioned_json(annotation_dir: Path):
    """
    Finds the highest instances_train_v*.json file.

    Returns:
        Path or None
    """
    pattern = re.compile(r"instances_train_v(\d+)\.json$")

    versioned_files = []

    for file in annotation_dir.glob("instances_train_v*.json"):
        match = pattern.match(file.name)
        if match:
            version = int(match.group(1))
            versioned_files.append((version, file))

    if not versioned_files:
        return None

    # Return file with highest version number
    versioned_files.sort(key=lambda x: x[0], reverse=True)
    return versioned_files[0][1]


def run_extract_script(
    coco_file: Path,
    images_dir: Path,
    output_dir: Path,
    obs_id: str,
):
    """
    Executes extract_crops_from_coco.py
    """
    cmd = [
        "python",
        "extract_crops_from_coco.py",
        "--coco-file",
        str(coco_file),
        "--images-dir",
        str(images_dir),
        "--output-dir",
        str(output_dir),
        "--obs-id",
        obs_id,
    ]

    print("\nRunning:")
    print(" ".join(cmd))

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"ERROR: Script failed for {coco_file}")

def process_trial(trial_dir: Path):
    """
    Processes a single trial directory.
    """
    annotation_dir = trial_dir / "reviewed_annotations"
    images_dir = trial_dir / "images" / "train"
    obs_id = trial_dir.name

    if not annotation_dir.exists():
        print(f"Skipping {trial_dir.name}: no reviewed_annotations/")
        return

    if not images_dir.exists():
        print(f"Skipping {trial_dir.name}: no images/")
        return

    # ------------------------------------------------------------
    # Run highest instances_train_v*.json
    # ------------------------------------------------------------
    latest_json = find_highest_versioned_json(annotation_dir)

    if latest_json:
        #output_dir = trial_dir / "output_latest_train"

        run_extract_script(
            coco_file=latest_json,
            images_dir=images_dir,
            output_dir=output_dir,
            obs_id=obs_id,

        )
    else:
        print(f"No instances_train_v*.json found in {trial_dir.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-data-path",
        required=True,
        help="Path to raw_data directory",
    )

    args = parser.parse_args()

    raw_data_path = Path(args.raw_data_path)

    if not raw_data_path.exists():
        raise FileNotFoundError(f"{raw_data_path} does not exist")

    # Iterate through trial folders
    for trial_dir in sorted(raw_data_path.iterdir()):
        if trial_dir.is_dir():
            print(f"\n{'=' * 60}")
            print(f"Processing: {trial_dir.name}")
            print(f"{'=' * 60}")

            process_trial(trial_dir)


if __name__ == "__main__":
    main()