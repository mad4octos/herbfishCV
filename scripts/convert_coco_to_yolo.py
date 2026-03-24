"""
Convert a COCO detection dataset to Ultralytics YOLO format using Datumaro.

Expected CSV layout (two columns, with header):

    dir_path,split
    /path/to/240101_01_label_L,train
    /path/to/240101_02_label_R,val
    ...

Each `dir_path` must be an observation subfolder whose name matches the pattern
YYMMDD_NN…_L|R and must contain `annotations/` and `images/<split>/` subdirs.

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

# External imports
import pandas as pd
from datumaro.components.annotation import (
    Annotation,
    AnnotationType,
    Bbox,
    LabelCategories,
)
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import CategoriesInfo


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


FOLDER_PATTERN = re.compile(r"(\d{6}_\d{2}).*_(L|R)")


def build_items_from_coco_json(
    ann_path: Path, images_dir: Path, split: str, folder_id: str, rl: str
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
    folder_id:
        Observation id extracted from the subfolder name (e.g. `240101_01`).
    rl:
        Side extracted from the subfolder name (`L` or `R`).
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
                id=f"{folder_id}_{rl}_{Path(file_name).stem}",
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
        Path to a CSV file with columns `dir_path` and `split`. Each row
        specifies an observation subfolder and the split it belongs to.
    output_dir:
        Destination directory for the converted YOLO dataset.
    """
    all_items: list[datumaro.components.dataset_base.DatasetItem] = []
    categories: CategoriesInfo | None = None

    df = pd.read_csv(csv_path)

    for _, row in df.iterrows():
        subdir = Path(row["dir_path"])
        split = row["split"]

        m = FOLDER_PATTERN.search(str(subdir.resolve()))
        if not m:
            print(f"Skipping {subdir.name}: does not match observation folder pattern.")
            continue

        folder_id, rl = m.groups()
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
            f"Processing {subdir.name} | prepend id={folder_id}_{rl}] | target split={split}"
        )

        items, label_categories = build_items_from_coco_json(
            annotation_file, images_dir, split, folder_id, rl
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
        str(output_dir), format="yolo_ultralytics_detection", save_media=True
    )
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
        help="CSV file with columns 'dir_path' and 'split'.",
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
