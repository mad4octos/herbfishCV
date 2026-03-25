"""
Convert a COCO detection dataset to Ultralytics YOLO format using Datumaro.

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
import os
import re
from collections import defaultdict
from pathlib import Path

# External imports
import datumaro.components.dataset_base
import datumaro.components.media
import pandas as pd
import yaml
from datumaro.components.annotation import (
    Annotation,
    AnnotationType,
    Bbox,
    LabelCategories,
)
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import CategoriesInfo

EXPECTED_COLUMNS = {"dir_path", "split", "observation_id"}


def verify_csv(df: pd.DataFrame, csv_path: Path) -> None:
    """Validate the CSV dataframe before processing.

    Raises
    ------
    ValueError
        If required columns are missing, any cell is empty, or observation_ids
        are not unique.
    """
    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV '{csv_path}' is missing required column(s): {sorted(missing)}. "
            f"Expected columns: {sorted(EXPECTED_COLUMNS)}, got: {sorted(df.columns)}"
        )

    for col in EXPECTED_COLUMNS:
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

    pattern = re.compile(rf"^instances_{re.escape(split)}_v(\d+)\.json$")
    best_version, best_path = -1, None
    for f in ann_dir.iterdir():
        m = pattern.match(f.name)
        if m:
            v = int(m.group(1))
            if v > best_version:
                best_version, best_path = v, f

    if best_path is not None:
        return best_path

    fallback = ann_dir / f"instances_{split}.json"
    return fallback if fallback.exists() else None


def _build_label_categories(
    coco_categories: list[dict],
) -> tuple[CategoriesInfo, dict[int, int]]:
    """Build Datumaro CategoriesInfo from a COCO categories list.

    Parameters
    ----------
    coco_categories:
        The `categories` list from a COCO JSON, each entry having at least
        `id` and `name` keys.

    Returns
    -------
    categories:
        Datumaro CategoriesInfo with one label per COCO category, ordered by
        COCO category id.
    coco_cat_id_to_dat_label_id:
        Mapping from COCO category id to the corresponding Datumaro label id.
    """
    label_cat = LabelCategories()
    coco_cat_id_to_dat_label_id: dict[int, int] = {}
    for label_id, cat in enumerate(sorted(coco_categories, key=lambda c: c["id"])):
        label_cat.add(cat["name"])
        coco_cat_id_to_dat_label_id[cat["id"]] = label_id
    return {AnnotationType.label: label_cat}, coco_cat_id_to_dat_label_id


def build_items_from_coco_json(
    ann_path: Path, images_dir: Path, split: str, observation_id: str
) -> tuple[list[datumaro.components.dataset_base.DatasetItem], CategoriesInfo]:
    """Parse a COCO instances JSON and return DatasetItems and categories.

    Parameters
    ----------
    ann_path:
        Path to the COCO instances JSON file.
    images_dir:
        Directory that contains the image files listed in the JSON.
    split:
        Subset name to assign to every item (e.g. `train`).
    observation_id:
        Unique identifier for this folder as specified in the CSV
        `observation_id` column (e.g. `240101_01_L`). Used as a filename
        prefix to avoid collisions across observation folders.
    """
    with open(ann_path, encoding="utf-8") as f:
        coco = json.load(f)

    label_categories, coco_cat_id_to_dat_label_id = _build_label_categories(
        coco["categories"]
    )

    # Group COCO annotations by image_id for fast lookup.
    anns_by_image: dict[int, list] = defaultdict(list)
    for ann in coco.get("annotations", []):
        anns_by_image[ann["image_id"]].append(ann)

    items: list[datumaro.components.dataset_base.DatasetItem] = []

    for img_info in coco["images"]:
        img_id = img_info["id"]
        h = img_info["height"]
        w = img_info["width"]
        file_name = img_info["file_name"]

        image_path = images_dir / file_name

        annotations: list[Annotation] = []
        for ann in anns_by_image[img_id]:
            x, y, bw, bh = ann["bbox"]
            label = coco_cat_id_to_dat_label_id[ann["category_id"]]
            annotations.append(Bbox(x, y, bw, bh, label=label))

        items.append(
            datumaro.components.dataset_base.DatasetItem(
                id=f"{observation_id}_{Path(file_name).stem}",
                subset=split,
                media=datumaro.components.media.Image.from_file(
                    str(image_path), size=(h, w)
                ),
                annotations=annotations,
            )
        )

    return items, label_categories


def convert(csv_path: Path, output_dir: Path) -> None:
    """Build a Datumaro dataset from observation dirs listed in a CSV and export as YOLO.

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
    all_items: list[datumaro.components.dataset_base.DatasetItem] = []
    categories: CategoriesInfo | None = None

    df = pd.read_csv(csv_path)
    verify_csv(df, csv_path)

    for _, row in df.iterrows():
        subdir = Path(row["dir_path"])
        split = row["split"]
        observation_id = row["observation_id"]

        annotations_dir = subdir / "annotations"
        images_dir = subdir / "images" / "train"

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

        items, label_categories = build_items_from_coco_json(
            annotation_file, images_dir, split, observation_id
        )
        all_items.extend(items)

        if categories is None:
            categories = label_categories

    dataset = Dataset.from_iterable(all_items, categories=categories)

    print(f"Loaded {len(dataset)} items total.")
    labels = dataset.categories().get(AnnotationType.label)
    names = [item.name for item in labels.items]
    print(f"Labels: {names}")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Exporting to YOLO Ultralytics format at: {output_dir}")
    dataset.export(
        str(output_dir),
        format="yolo_ultralytics_detection",
        save_media=True,
        add_path_prefix=False,
    )

    # Datumaro skips label files for items with no annotations.
    # Create empty .txt files for any exported image that has no label file.
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    images_root = output_dir / "images"
    labels_root = output_dir / "labels"
    empty_count = 0
    for image_file in images_root.rglob("*"):
        if image_file.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label_file = labels_root / image_file.relative_to(images_root).with_suffix(
            ".txt"
        )
        if not label_file.exists():
            label_file.parent.mkdir(parents=True, exist_ok=True)
            label_file.touch()
            empty_count += 1
    if empty_count:
        print(f"Created {empty_count} empty label file(s) for unannotated images.")

    # Datumaro generates train.txt/val.txt with forward-slash relative paths.
    # On Windows, Ultralytics concatenates the parent dir (backslash) with these
    # paths producing mixed separators, so the \images\ -> \labels\ replacement
    # that Ultralytics uses for label lookup fails silently.
    # Fix: rewrite data.yaml to reference image directories directly instead of
    # txt files. Ultralytics then scans each directory with fully resolved absolute
    # paths, so the replacement works correctly on all platforms.
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, encoding="utf-8") as f:
        data_yaml = yaml.safe_load(f)

    data_yaml["path"] = str(output_dir.resolve())
    for split in ("train", "val", "test"):
        if split not in data_yaml:
            continue
        split_dir = output_dir / "images" / split
        if split_dir.is_dir():
            data_yaml[split] = f"images/{split}"
        else:
            del data_yaml[split]

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

    for split in ("train", "val", "test"):
        txt_file = output_dir / f"{split}.txt"
        if txt_file.exists():
            txt_file.unlink()

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
    os._exit(0)
