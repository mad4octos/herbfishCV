"""
Convert a COCO detection dataset to Ultralytics YOLO format.

Expected CSV layout (three columns, with header):

    dir_path,split,observation_id
    /path/to/240101_01_label_L,train,240101_01_L
    /path/to/240101_02_label_R,val,240101_02_R
    ...

Each `dir_path` must be an observation subfolder that contains `annotations/`
and `images/train/` subdirs. The `observation_id` value is used as a filename
prefix in the output to avoid collisions across folders.

Output directory will contain the YOLO layout expected by Ultralytics:

    <output_dir>/
    ├── images/
    │   └── <split>/
    ├── labels/
    │   └── <split>/
    └── data.yaml

Usage:
    python scripts/convert_coco_to_yolo.py \\
        --csv /path/to/dirs.csv \\
        --output-dir /path/to/yolo_dataset
"""

# Standard Library imports
import argparse
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

# External imports
import pandas as pd
import yaml

EXPECTED_CSV_COLUMNS = {"dir_path", "split", "observation_id"}


def verify_csv(df: pd.DataFrame, csv_path: Path) -> None:
    """Validate the CSV dataframe before processing.

    Raises
    ------
    ValueError
        If the CSV is empty, required columns are missing, any cell is empty,
        or observation_ids are not unique.
    """
    if df.empty:
        raise ValueError(f"CSV '{csv_path}' contains no data rows.")

    missing = EXPECTED_CSV_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV '{csv_path}' is missing required column(s): {sorted(missing)}. "
            f"Expected columns: {sorted(EXPECTED_CSV_COLUMNS)}, got: {sorted(df.columns)}"
        )

    for col in EXPECTED_CSV_COLUMNS:
        empty_rows = df.index[
            df[col].isna() | (df[col].astype(str).str.strip() == "")
        ].tolist()
        if empty_rows:
            raise ValueError(
                f"CSV '{csv_path}': column '{col}' has empty value(s) at row(s): "
                f"{[r + 2 for r in empty_rows]} (1-indexed, including header)"
            )

    duplicates = df["observation_id"][df["observation_id"].duplicated()].tolist()
    if duplicates:
        raise ValueError(
            f"CSV '{csv_path}': 'observation_id' must be unique, "
            f"but found duplicate value(s): {duplicates}"
        )


def find_latest_coco_annotation(ann_dir: Path, split: str = "train") -> Path | None:
    """Return the highest-versioned `instances_<split>_vN.json` file in `ann_dir`.
    Fall back to `instances_<split>.json`.
    """
    if not ann_dir.is_dir():
        return None

    pattern = re.compile(rf"^instances_{split}_v(\d+)\.json$")
    best_version, best_path = -1, None
    for f in ann_dir.iterdir():
        if m := pattern.match(f.name):
            v = int(m.group(1))
            if v > best_version:
                best_version, best_path = v, f

    if best_path is not None:
        return best_path

    fallback = ann_dir / f"instances_{split}.json"
    return fallback if fallback.exists() else None


def coco_bbox_to_yolo(
    x: float,
    y: float,
    bbox_width: float,
    bbox_height: float,
    image_w: int,
    image_h: int,
) -> tuple[float, float, float, float] | None:
    """Convert a COCO bbox to normalised YOLO format.

    Parameters
    ----------
    x, y:
        Top-left corner of the COCO bbox in pixels.
    bbox_width, bbox_height:
        Width and height of the COCO bbox in pixels.
    image_w, image_h:
        Image dimensions used for normalisation.

    Returns
    -------
    tuple[float, float, float, float] or None
        ``(cx, cy, w, h)`` normalised to [0, 1], or ``None`` if the bbox is
        degenerate (zero or negative area).
    """
    norm_bbox_width = bbox_width / image_w
    norm_bbox_height = bbox_height / image_h
    if norm_bbox_width <= 0 or norm_bbox_height <= 0:
        return None
    cx = (x + bbox_width / 2) / image_w
    cy = (y + bbox_height / 2) / image_h
    return cx, cy, norm_bbox_width, norm_bbox_height


def _parse_categories(coco_categories: list[dict]) -> tuple[dict[int, int], list[str]]:
    """Return a COCO-id-to-YOLO-class-index map and an ordered list of class names."""
    categories = sorted(coco_categories, key=lambda c: c["id"])
    coco_id_to_cls = {c["id"]: i for i, c in enumerate(categories)}
    labels = [c["name"] for c in categories]
    return coco_id_to_cls, labels


def _annotation_lines(
    anns: list[dict], coco_id_to_cls: dict[int, int], image_w: int, image_h: int
) -> list[str]:
    lines: list[str] = []
    for ann in anns:
        yolo = coco_bbox_to_yolo(*ann["bbox"], image_w=image_w, image_h=image_h)
        if yolo is None:
            continue
        cx, cy, bw, bh = yolo
        cls = coco_id_to_cls[ann["category_id"]]
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def convert_observation(
    ann_path: Path, images_dir: Path, split: str, observation_id: str, output_dir: Path
) -> tuple[list[str], int]:
    """Convert one observation's COCO JSON to YOLO label files and copy images.

    Parameters
    ----------
    ann_path:
        Path to the COCO instances JSON file.
    images_dir:
        Directory containing the image files listed in the JSON.
    split:
        Subset name (e.g. `train`, `val`).
    observation_id:
        Unique identifier used as a filename prefix to avoid collisions across
        observation folders (e.g. `240101_01_L`).
    output_dir:
        Root of the output YOLO dataset.

    Returns
    -------
    tuple[list[str], int]
        Class names ordered by their 0-based YOLO label index, and the number
        of images processed.
    """
    labels_split = output_dir / "labels" / split
    images_split = output_dir / "images" / split
    labels_split.mkdir(parents=True, exist_ok=True)
    images_split.mkdir(parents=True, exist_ok=True)

    with open(ann_path, encoding="utf-8") as f:
        coco = json.load(f)

    coco_id_to_cls, labels = _parse_categories(coco["categories"])

    # Group annotations by image_id for fast lookup
    annotations_by_image: dict[int, list] = defaultdict(list)
    for ann in coco.get("annotations", []):
        annotations_by_image[ann["image_id"]].append(ann)

    for image_info in coco["images"]:
        annotation = annotations_by_image[image_info["id"]]
        image_h = image_info["height"]
        image_w = image_info["width"]
        image_fp = Path(image_info["file_name"])
        output_name = f"{observation_id}_{image_fp.stem}"

        # Write label file
        label_filepath = labels_split / f"{output_name}.txt"
        lines = _annotation_lines(annotation, coco_id_to_cls, image_w, image_h)
        label_filepath.write_text("\n".join(lines), encoding="utf-8")

        # Copy image
        src = images_dir / image_fp
        dst = images_split / f"{output_name}{image_fp.suffix}"
        shutil.copy2(src, dst)

    return labels, len(coco["images"])


def write_data_yaml(output_dir: Path, labels: list[str]) -> None:
    """Write the Ultralytics data.yaml for the converted dataset."""
    splits_present = [
        s for s in ("train", "val", "test") if (output_dir / "images" / s).is_dir()
    ]
    data_yaml: dict = {"path": str(output_dir.resolve())}
    for s in splits_present:
        data_yaml[s] = f"images/{s}"
    data_yaml["nc"] = len(labels)
    data_yaml["names"] = labels
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)
    print(f"Wrote {yaml_path}")


def convert(csv_path: Path, output_dir: Path) -> None:
    """Convert all observation dirs listed in a CSV to a YOLO dataset.

    Parameters
    ----------
    csv_path:
        Path to a CSV file with columns `dir_path`, `split`, and
        `observation_id`. Each row specifies an observation subfolder, the
        split it belongs to, and the unique identifier used to prefix output
        filenames.
    output_dir:
        Destination directory for the converted YOLO dataset.
    """
    df = pd.read_csv(csv_path)
    verify_csv(df, csv_path)

    labels: list[str] = []
    total_images = 0

    for _, row in df.iterrows():
        subdir = Path(row["dir_path"])
        split = row["split"]
        observation_id = row["observation_id"]

        # Source images always live under images/train/ regardless of the target split,
        # because Labelme exports all images into that fixed subdirectory.
        images_dir = subdir / "images" / "train"
        annotations_dir = subdir / "annotations"

        annotation_file = find_latest_coco_annotation(annotations_dir)
        if annotation_file is None:
            raise FileNotFoundError(
                f"No annotation file found for split '{split}' in {annotations_dir}"
            )

        if not images_dir.is_dir():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")

        print(
            f"Processing {subdir.name} | observation_id={observation_id} | split={split}"
        )

        observation_labels, image_count = convert_observation(
            annotation_file, images_dir, split, observation_id, output_dir
        )

        if not labels:
            labels = observation_labels
        elif observation_labels != labels:
            raise ValueError(
                f"Category mismatch in '{observation_id}': "
                f"expected {labels}, got {observation_labels}"
            )

        total_images += image_count

    print(f"Converted {total_images} images total.")
    print(f"Labels: {labels}")

    write_data_yaml(output_dir, labels)
    print("Done.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a COCO detection dataset to Ultralytics YOLO format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--csv",
        required=True,
        type=Path,
        help="CSV file with columns 'dir_path', 'split', and 'observation_id'.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Destination directory for the converted YOLO dataset.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert(csv_path=args.csv, output_dir=args.output_dir)
