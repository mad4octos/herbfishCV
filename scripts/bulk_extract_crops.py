"""
Iterates over every (frames_folder, mask_file) pair in the matched CSV and invokes extract_crops.py once per row 
via subprocess.

Usage:
    python bulk_process_crops.py
"""

import subprocess
import pandas as pd

FRAMES_BASE = "/scratch/alpine/maha7624/3D_Tracking/2024_FF/frames"
MASKS_BASE = "/scratch/alpine/maha7624/3D_Tracking/2024_FF/masks"
ERRORS_CSV = "/scratch/alpine/maha7624/3D_Tracking/2024_FF/SAM2_errors_ff_2024 - SAM2_errors.csv"
OUTPUT_FOLDER = "/scratch/alpine/maha7624/3D_Tracking/2024_FF/automatic_mask_cleaner_data"
CSV_PATH = "frames_masks_matched.csv"

df = pd.read_csv(CSV_PATH, encoding="utf-8")
df = df.dropna(subset=["frames"])

for _, row in df.iterrows():
    frames = row["frames"].strip()
    masks = row["masks"].strip()

    # Strip annotator prefix up to and including the first underscore
    # e.g. "CR_AK_..." -> "AK_..."
    obs_id = masks.split("_", 1)[1]

    # Strip trailing "_mask.pkl" suffix
    obs_id = obs_id.removesuffix("_mask.pkl")

    cmd = [
        "python",
        "extract_crops.py",
        "--images-dirpath",
        f"{FRAMES_BASE}/{frames}",
        "--masks-filepath",
        f"{MASKS_BASE}/{masks}",
        "--errors-csv-filepath",
        ERRORS_CSV,
        "--errors-obs-id",
        obs_id,
        "--output-folder",
        OUTPUT_FOLDER,
        "--overlay"
    ]

    # print(" ".join(f'"{c}"' if " " in c else c for c in cmd))
    subprocess.run(cmd)
