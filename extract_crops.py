"""Extract blob crops from SAM2 segmentation masks and classify them by error status.

Reads video frames and their corresponding SAM2 masks, filters blobs by area
and size thresholds, and saves cropped regions as RGBA PNGs. Each crop is
sorted into 'correct' or 'incorrect' subdirectories based on a manually
annotated errors CSV.
"""

# Standard Library imports
import argparse
from pathlib import Path

# External imports
import cv2
import numpy as np
import numpy.typing as npt
import pandas as pd
from tqdm import tqdm

# Local imports
from blob_filter_rules import MinAreaRule, MinSizeRule
from common import sparse_mask_tensor_to_dense_numpy
from configuration import Config
from convert_utils import (
    BlobInfo,
    _get_frame_filename,
    extract_error_frames,
    get_blobs_from_mask,
    load_masks,
)

# Constants
CLASS_CORRECT = "correct"
CLASS_INCORRECT = "incorrect"
ERROR_TYPES_OF_INTEREST = [
    "Coral/rock/rubble masked without focal fish",
    "Coral/rock/rubble masked with fish",
]


def partition_frames_by_errors(
    images_dirpath: Path,
    errors_df: pd.DataFrame,
    error_types: list[str],
) -> tuple[list[int], list[int]]:
    """Classify frames as correct or incorrect based on error types.

    Parameters
    ----------
    images_dirpath : Path
        Directory containing the image files.
    errors_df : pd.DataFrame
        DataFrame containing error annotations for the observation.
    error_types : list[str]
        List of specific error types to classify as incorrect
        (e.g., ["Coral/rock/rubble masked without focal fish"]).

    Returns
    -------
    tuple[list[int], list[int]]
        A tuple containing:
        - incorrect_frames: List of frame indices with errors of the specified types
        - correct_frames: List of frame indices without any errors
    """
    # Total number of frames
    total_frames = sum(1 for _ in images_dirpath.iterdir())
    all_frames = set(range(0, total_frames))

    # Get all frames with any errors
    all_error_frames = set(extract_error_frames(errors_df, include_end=False))

    # Get frames with specific error types of interest
    incorrect_frames = extract_error_frames(
        errors_df, include_end=False, error_type=error_types
    )

    # Correct frames are those without any errors
    correct_frames = list(all_frames - all_error_frames)

    return incorrect_frames, correct_frames


def get_first_frame_info(images_path: Path) -> tuple[Path, int, int]:
    """Find the first image file and extract its frame number.

    Parameters
    ----------
    images_path : Path
        Directory containing the image files.

    Returns
    -------
    tuple[Path, int, int]
        A tuple containing:
        - first_image: Path to the first image file
        - first_frame_number: The 1-indexed frame number from the filename (e.g., 21 from "0021.jpg")
        - first_frame_idx: The 0-indexed frame index (first_frame_number - 1)

    Raises
    ------
    FileNotFoundError
        If no image files are found in the directory.
    """
    image_files = sorted(
        [
            f
            for f in images_path.iterdir()
            if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ]
    )

    if not image_files:
        raise FileNotFoundError(f"No image files found in directory '{images_path}'")

    first_image = image_files[0]
    # Extract the numeric part from the filename (e.g., "0021.jpg" -> "0021" -> 21)
    first_frame_number = int(first_image.stem)
    # Frame indices are 1-indexed in the filename, so convert to 0-indexed
    first_frame_idx = first_frame_number - 1

    return first_frame_idx


def get_filtered_blobs(
    dense_object_mask: np.ndarray,
    obj_id: int,
    frame_idx: int,
    min_size_threshold: int,
    area_threshold: int,
) -> list[BlobInfo]:
    """Get filtered blobs from dense mask.
    The blobs are filtered based on a minimal area, height and width."""

    valid_blobs: list[BlobInfo] = []
    blob_rules = [MinAreaRule(area_threshold), MinSizeRule(min_size_threshold)]
    for blob in get_blobs_from_mask(dense_object_mask, obj_id, frame_idx):
        if all(rule(blob) for rule in blob_rules):
            valid_blobs.append(blob)
    return valid_blobs


def extract_blobs(
    object_mask: npt.NDArray[np.uint8],
    frame_idx: int,  # 0-indexed
    obj_id: int,
    input_image: npt.NDArray[np.uint8],
    area_threshold: int = 200,
    min_size_threshold: int = 20,
    output_folder: Path | str = "output_blobs",
    do_mask=False,
    overlay=False,
):
    """Crop fish and mask with blob object mask, then save to disk.

    - Filters blobs by area and minimum dimension
    - crops each valid blob from the input image.
    - If `do_mask`, the background outside the blob is preserved but the crop is tightly bounded by the blob;
    - otherwise a plain bounding-box crop is used.
    - Crops are saved as PNGs

    """

    # Create output folder if required
    output_dirpath = Path(output_folder)
    output_dirpath.mkdir(exist_ok=True, parents=True)

    filtered_blobs = get_filtered_blobs(
        object_mask,
        obj_id,
        frame_idx,
        area_threshold=area_threshold,
        min_size_threshold=min_size_threshold,
    )

    for blob in filtered_blobs:
        observation_id = f"frame_{frame_idx}_obj_{obj_id}_blob_{blob.blob_num}"
        if do_mask:
            if overlay:
                crop = blob.mask_and_crop_blob(input_image, remove_background=False)
            else:
                crop = blob.crop_blob_rgba(input_image)
        else:
            crop = blob.crop_from_image(input_image)

        blob_filename = Path(output_folder, f"{observation_id}.png")
        cv2.imwrite(str(blob_filename), crop)


def main(
    obs_id: str,
    images_path: Path,
    masks_filepath: Path,
    output_folder: Path,
    incorrect_frames: list[int],
    correct_frames: list[int],
    filename_num_zeros: int,
    area_threshold: int,
    size_threshold: int,
    overlay=False,
):
    """ """

    if not masks_filepath.exists() or not masks_filepath.is_file():
        raise FileNotFoundError(
            f"Couldn't find masks file or is not a file '{masks_filepath}'"
        )

    for subdir in [CLASS_CORRECT, CLASS_INCORRECT]:
        dir_path = output_folder / obs_id / subdir
        if dir_path.exists() and any(dir_path.iterdir()):
            raise ValueError(f"Destination directory '{dir_path}' is not empty")

    if (not incorrect_frames) or (not correct_frames):
        raise ValueError(
            "Correct/Incorrect frames must be provided to distinguish correct from incorrect blobs."
        )

    masks = load_masks(masks_filepath)
    print(f"Loading masks from '{masks_filepath.absolute()}'")

    num_image_files = sum(1 for _ in images_path.iterdir())
    assert num_image_files == len(masks), (
        f"Number of image files on disk ({num_image_files}) does not match number of masks ({len(masks)})"
    )

    first_frame_idx = get_first_frame_info(images_path)

    for extracted_frame_idx, frame_masks in tqdm(  # 0-indexed!
        masks.items(), desc="Processing frames", unit="frame"
    ):
        if extracted_frame_idx in incorrect_frames:
            class_name = CLASS_INCORRECT
        elif extracted_frame_idx in correct_frames:
            class_name = CLASS_CORRECT
        else:
            print(f"Skipping frame {extracted_frame_idx}: not in subset of interest")
            continue

        dst_dir = output_folder / obs_id / class_name

        # - The number of zeros is important. For the focal follow data is 4, for the stationary data is 5
        # - The mask indices are not neccesarily in sync with the frame indices (even when moved from 1-index to
        #   0-index space). This is because sometimes, some frames were cut-off when generating the SAM2 masks.
        #   The relationship between frame numbers (e.g. 1 in 0001.jpg) and mask indices (extracted_frame_idx) is:
        #       extracted_frame_idx = frame_number - first_frame_idx (0-index) - 1
        filename = _get_frame_filename(
            extracted_frame_idx + first_frame_idx, filename_num_zeros
        )
        image_filepath = images_path / filename

        if not image_filepath.exists():
            print(f"Frame doesn't exist! '{image_filepath.name}'")
            continue

        # Load frame
        input_frame = cv2.imread(str(image_filepath), cv2.IMREAD_COLOR)

        for obj_id in frame_masks.keys():
            # Load mask corresponding to one object (can be multiple blobs!)
            sparse_mask = frame_masks[obj_id]
            dense_object_mask = sparse_mask_tensor_to_dense_numpy(sparse_mask)

            extract_blobs(
                dense_object_mask,
                extracted_frame_idx,
                obj_id,
                input_frame,
                area_threshold=area_threshold,
                min_size_threshold=size_threshold,
                output_folder=dst_dir,
                do_mask=True,
                overlay=overlay,
            )


def parse_args():
    """ """

    parser = argparse.ArgumentParser(
        description="Extract blob crops with an alpha layer for the segmentation mask from SAM2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_crops.py \\
    --images-dirpath "/path/to/MH_JM_060624_145_L" \\
    --masks-filepath "/path/to/CR_JM_060624_145_playa_largu_scuba_IPSpv_L_mask.pkl" \\
    --errors-csv-filepath "/path/to/SAM2_errors_ff_2024 - SAM2_errors.csv"
    --errors-obs-id "JM_060624_145_playa_largu_scuba_IPSpv_L" \\
    --output-folder "/path/to/automatic_mask_cleaner_anomaly_detector_data" \\
    """,
    )
    parser.add_argument(
        "--images-dirpath",
        type=Path,
        help="The absolute path to the directory containing the image frames.",
    )
    parser.add_argument(
        "--masks-filepath",
        type=Path,
        help="The absolute file path for the masks file (e.g., '/path/to/video_name_mask.pkl').",
    )
    parser.add_argument(
        "--errors-csv-filepath",
        type=Path,
        help="The absolute file path for the errors CSV file (e.g., '/path/to/SAM2_errors.csv').",
    )
    parser.add_argument(
        "--output-folder",
        type=Path,
        help="Output folder for extracted crops.",
    )
    parser.add_argument(
        "--errors-obs-id",
        type=str,
        help="The exact observation ID string as it appears in the errors CSV file.",
    )
    parser.add_argument(
        "--filename-num-zeros",
        type=int,
        default=4,
        help="Number of zero-padded digits in frame filenames (4 for focal follow, 5 for stationary).",
    )
    parser.add_argument(
        "--area-threshold",
        type=int,
        default=Config.area_threshold,
        help="Minimum blob area in pixels.",
    )
    parser.add_argument(
        "--size-threshold",
        type=int,
        default=Config.min_size_threshold,
        help="Minimum blob width/height in pixels.",
    )
    parser.add_argument(
        "--overlay",
        action="store_true",
        default=False,
        help="Whether to produce an overlaid mask crop rather than a RGBA",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Validate masks file
    if not args.masks_filepath.is_absolute():
        raise ValueError(f"Masks file path must be absolute: {args.masks_filepath}")
    if not args.masks_filepath.exists():
        raise FileNotFoundError(f"Masks file not found: {args.masks_filepath}")
    if not args.masks_filepath.is_file():
        raise ValueError(f"Masks file path is not a file: {args.masks_filepath}")

    # Validate images directory
    if not args.images_dirpath.is_absolute():
        raise ValueError(
            f"Images directory path must be absolute: {args.images_dirpath}"
        )
    if not args.images_dirpath.exists():
        raise FileNotFoundError(f"Images directory not found: {args.images_dirpath}")
    if not args.images_dirpath.is_dir():
        raise ValueError(f"Images path is not a directory: {args.images_dirpath}")

    ####################################################################################################################
    # Load errors CSV
    ####################################################################################################################
    video_errors_df = pd.read_csv(args.errors_csv_filepath)
    video_errors_df = video_errors_df.loc[video_errors_df.obsID == args.errors_obs_id]

    # Classify frames based on error types
    incorrect_frames, correct_frames = partition_frames_by_errors(
        args.images_dirpath, video_errors_df, ERROR_TYPES_OF_INTEREST
    )

    assert not set(correct_frames).intersection(set(incorrect_frames)), (
        "There should be no intersection between correct frames and incorrect frames of interest"
    )

    ####################################################################################################################

    # TODO: this function will consider ALL fish within an erroneous frame as an incorrectly masked fish
    # That's fine for the Focal Follow project because the CSV shows only one ObjID, but if this is used in a project
    # where there may be both correctly and incorrectly labeled fish in a single frame, that would be a problem.
    main(
        obs_id=args.errors_obs_id,
        images_path=args.images_dirpath,
        masks_filepath=args.masks_filepath,
        output_folder=args.output_folder,
        filename_num_zeros=args.filename_num_zeros,
        incorrect_frames=incorrect_frames,
        correct_frames=correct_frames,
        area_threshold=args.area_threshold,
        size_threshold=args.size_threshold,
        overlay=args.overlay,
    )
