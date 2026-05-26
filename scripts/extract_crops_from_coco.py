"""
Extract cropped RGBA image regions from COCO annotations.

This script is intended to processes COCO annotations generated with LabelMe, that is, on files:
- instances_train_vn.json
- instances_val_vn.json
- incorrect_predictions.json

Annotations from `incorrect_predictions.json` are always classified as *incorrect*
(with a `rejection_type` suffix in the filename); all others as *correct*.

Output directory structure:

    output_dir/
    └── {obs_id}/
        ├── correct/
        │   └── {stem}_ann{id}_{category}_crop.png
        └── incorrect/
            └── {stem}_ann{id}_{category}_{rejection_type}_crop.png

Usage:
    python extract_crops_from_coco.py --coco-file annotations.json \
        --images-dir ./images --output-dir ./output --obs-id observation_id
"""

# Standard Library imports
import argparse
import json
import logging
import sys
from pathlib import Path

# External imports
import cv2
import numpy as np
import numpy.typing as npt
import supervision as sv

# Local imports
from coco_types import CocoAnnotation, CocoCategories, CocoImage, CompressedRLE
from supervision.dataset.utils import rle_to_mask
from supervision.detection.utils.converters import polygon_to_mask
from supervision.utils.image import crop_image

# Logging
_log_formatter = logging.Formatter(
    fmt="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_console_handler = logging.StreamHandler()
_console_handler.setFormatter(_log_formatter)
logger.addHandler(_console_handler)

CLASS_CORRECT = "correct"
CLASS_INCORRECT = "incorrect"


def coco_annotations_to_masks(
    image_annotations: list[CocoAnnotation], resolution_wh: tuple[int, int]
) -> npt.NDArray[np.bool_]:
    """Convert a list of COCO annotations to a stack of boolean masks.

    Handles both crowd annotations (RLE-encoded) and instance annotations
    (polygon-encoded). Annotations missing a valid segmentation field produce
    an all-zero mask of the requested resolution.

    Args:
        image_annotations: List of COCO annotations for a single image.
        resolution_wh: Output mask resolution as `(width, height)`.

    Returns:
        Boolean array of shape `(N, height, width)` where `N` is the
        number of annotations.
    """
    masks = []
    for annotation in image_annotations:
        if annotation.iscrowd:
            assert isinstance(annotation.segmentation, CompressedRLE)
            rle = np.array(annotation.segmentation.counts)
            mask = rle_to_mask(rle=rle, resolution_wh=resolution_wh)
            masks.append(mask)
        else:
            if not isinstance(annotation.segmentation, list):
                mask = np.zeros((resolution_wh[1], resolution_wh[0]), dtype=np.bool_)
            else:
                polygon = np.reshape(
                    np.asarray(annotation.segmentation, dtype=np.int32),
                    (-1, 2),
                )
                mask = polygon_to_mask(polygon=polygon, resolution_wh=resolution_wh)
            masks.append(mask)
    return np.array(masks, dtype=bool)


def process_coco_file(
    coco_file: Path, images_dir: Path, output_dir: Path, obs_id: str
) -> None:
    """Process a COCO annotation file and save RGBA crops.

    For each annotation with a bbox, crops the image region and saves it as a
    4-channel BGRA PNG with the segmentation mask as the alpha channel.

    Args:
        coco_file: Path to the COCO JSON file.
        images_dir: Directory containing the source images.
        output_dir: Directory to save outputs.
        obs_id: Observation ID used as the top-level output subdirectory name.
    """
    with open(coco_file, encoding="utf-8") as f:
        coco_data = json.load(f)

    # Build image lookup
    images: dict[int, CocoImage] = {
        img["id"]: CocoImage.from_dict(img) for img in coco_data.get("images", [])
    }
    categories: dict[int, CocoCategories] = {
        cat["id"]: cat for cat in coco_data.get("categories", [])
    }
    raw_annotations: list[dict] = coco_data.get("annotations", [])

    # Create output directories
    for class_name in [CLASS_CORRECT, CLASS_INCORRECT]:
        (output_dir / obs_id / class_name).mkdir(parents=True, exist_ok=True)

    all_incorrect = coco_file.name == "incorrect_predictions.json"

    logger.info(f"Processing {len(raw_annotations)} annotations from {coco_file.name}")

    for raw_ann in raw_annotations:
        ann = CocoAnnotation.from_dict(raw_ann)

        if ann.image_id not in images:
            logger.warning(
                f"Image ID {ann.image_id} not found, skipping annotation {ann.id}"
            )
            continue

        image_info = images[ann.image_id]
        image_path = images_dir / image_info.filepath
        if not image_path.exists():
            logger.warning(
                f"Image file not found: {image_path}, skipping annotation {ann.id}"
            )
            continue

        # Skip those annotations where the ground-truth click-location attributes on error frames were preserved
        # without including a real mask.
        if ann.area == 0:
            continue

        category_name = categories.get(ann.category_id, {}).get("name")
        assert category_name is not None

        # Get rejection type if present (for incorrect_predictions.json)
        rejection_type = raw_ann.get("rejection_type", "") if all_incorrect else ""
        suffix = f"_{rejection_type}" if rejection_type else ""

        # Base filename for outputs
        frame_idx = int(image_info.filepath.stem)
        obj_id = ann.attributes.get("ObjID")
        base_name = f"frame_{frame_idx}_obj_{obj_id}{suffix}"

        class_name = CLASS_INCORRECT if all_incorrect else CLASS_CORRECT
        crops_dir = output_dir / obs_id / class_name

        # Generate mask from segmentation (needed for RGBA alpha channel)
        masks = coco_annotations_to_masks([ann], (image_info.width, image_info.height))
        mask = (masks[0] * 255).astype(np.uint8)

        # Crop image using bbox
        if ann.bbox:
            image = cv2.imread(str(image_path))
            if image is None:
                logger.warning(f"Could not read image: {image_path}")
                continue

            xywh = np.array([ann.bbox])
            xyxy = sv.xywh_to_xyxy(xywh)[0]
            mask_crop = crop_image(image=mask, xyxy=xyxy)
            crop = crop_image(image=image, xyxy=xyxy)
            rgba_crop = np.dstack([crop, mask_crop])
            crop_path = crops_dir / f"{base_name}.png"
            cv2.imwrite(str(crop_path), rgba_crop)
            logger.info(f"Saved crop: {crop_path.name}")

    logger.info(f"Done! Outputs saved to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert COCO segmentations to SAM2 masks and crop frames using bbox"
    )
    parser.add_argument(
        "--coco-file",
        type=Path,
        required=True,
        help="Path to COCO annotation JSON file (e.g., instances_train_v1.json, incorrect_predictions.json)",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        required=True,
        help="Directory containing source images",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./sam2_outputs"),
        help="Output directory for crops (default: ./sam2_outputs)",
    )
    parser.add_argument(
        "--obs-id",
        type=str,
        required=True,
        help="Observation ID used as the top-level output subdirectory name.",
    )
    args = parser.parse_args()

    if not args.coco_file.exists():
        logger.error(f"COCO file not found: {args.coco_file}")
        sys.exit(1)

    if not args.images_dir.exists():
        logger.error(f"Images directory not found: {args.images_dir}")
        sys.exit(1)

    process_coco_file(
        coco_file=args.coco_file,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        obs_id=args.obs_id,
    )


if __name__ == "__main__":
    main()
