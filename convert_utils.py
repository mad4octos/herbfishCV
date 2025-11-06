# Standard Library imports
import pickle
from enum import IntEnum
from pathlib import Path

# External imports
import cv2
import datumaro.components.dataset_base
from datumaro.components.annotation import AnnotationType, LabelCategories
import numpy as np
import pandas as pd
import torch

# Local imports
from blob import BlobInfo

# Type aliases for the masks file
FrameIndex = int
ObjectIndex = int
MasksType = dict[FrameIndex, dict[ObjectIndex, torch.Tensor]]


class ClickType(IntEnum):
    ENTER = 3
    EXIT = 4


def get_blobs_from_mask(
    object_mask: np.ndarray, obj_id: int, extracted_frame_idx: int
) -> list[BlobInfo]:
    """ """

    num_blobs, labeled_mask, stats, centroids = cv2.connectedComponentsWithStats(
        object_mask, connectivity=8, ltype=cv2.CV_32S
    )

    blobs = []

    # Iterate labels skipping background (label 0)
    for blob_num in range(1, num_blobs):
        x = int(stats[blob_num, cv2.CC_STAT_LEFT])
        y = int(stats[blob_num, cv2.CC_STAT_TOP])
        w = int(stats[blob_num, cv2.CC_STAT_WIDTH])
        h = int(stats[blob_num, cv2.CC_STAT_HEIGHT])
        area = int(stats[blob_num, cv2.CC_STAT_AREA])
        cx, cy = centroids[blob_num]

        blob = BlobInfo(
            frame_idx=extracted_frame_idx,
            blob_num=blob_num,
            obj_id=obj_id,
            x=x,
            y=y,
            w=w,
            h=h,
            area=area,
            centroid_x=float(round(cx, 1)),
            centroid_y=float(round(cy, 1)),
        )

        blob.store_mask(labeled_mask)
        blobs.append(blob)

    return blobs


def get_frame_chunks_df(
    df: pd.DataFrame,
    instance_id="ObjID",
    class_name="ObjType",
    frame_name="Frame",
    click_type_name="ClickType",
):
    """
    Modified from:
    https://github.com/mad4octos/Annotator_GUI/blob/7ea8b4aeb53e69622f23b6a24c466a39b5a3ff0e/SAM2_Tracking/utils.py#L518

    Using `click_type_name` obtain the enter and exit frame values for each `instance_id`.
    """

    # For each instance_id get frame where the object enters the scene
    enter_frame = df[df[click_type_name] == ClickType.ENTER][
        [instance_id, class_name, frame_name]
    ]
    enter_frame = enter_frame.sort_values(by=[instance_id, frame_name], ascending=True)

    # For each instance_id get frame where the object exits the scene
    exit_frame = df[df[click_type_name] == ClickType.EXIT][
        [instance_id, class_name, frame_name]
    ]
    exit_frame = exit_frame.sort_values(by=[instance_id, frame_name], ascending=True)

    # Check that each enter point has a corresponding exit point
    if (enter_frame.shape != exit_frame.shape) or (
        not np.array_equal(
            enter_frame[instance_id].values, exit_frame[instance_id].values
        )
    ):
        raise RuntimeError(
            f"A {instance_id} does not have both an enter and exit point!"
        )

    # Drop instance_id from exit_frame, now that we have sorted and compared them
    exit_frame.drop(columns=[instance_id, class_name], axis=1, inplace=True)

    # Turn instance_id column back to a string
    enter_frame[instance_id] = enter_frame[instance_id].astype(str)

    # Concatenate columns to improve ease of use later
    obj_frame_chunks = pd.concat(
        [enter_frame.reset_index(drop=True), exit_frame.reset_index(drop=True)], axis=1
    )
    obj_frame_chunks.columns = [instance_id, class_name, "EnterFrame", "ExitFrame"]

    return obj_frame_chunks


def load_masks(masks_filepath: Path) -> MasksType:
    """Load SAM2 masks from a .pkl file."""
    print(f"Loading masks file: '{masks_filepath}'")
    if not masks_filepath.exists():
        raise FileNotFoundError(
            f"Masks file '{masks_filepath.resolve()}' doesn't exist!"
        )

    with open(masks_filepath, "rb") as masks_file:
        masks: MasksType = pickle.load(masks_file)
    return masks


def load_annotations(annotations_filepath: Path):
    """Load annotations from a .npy file as a Pandas DataFrame"""
    print(f"Loading annotations file: '{annotations_filepath}'")
    with open(annotations_filepath, "rb") as annotations_file:
        annotations = np.load(annotations_file, allow_pickle=True)

    annotations_df = pd.DataFrame(list(annotations))
    annotations_df = annotations_df.sort_values(by=["Frame"]).reset_index(drop=True)

    return annotations_df


def load_categories(
    annotations_df: pd.DataFrame,
) -> datumaro.components.dataset_base.CategoriesInfo:
    """Load Datumaro categories data"""
    label_categories = LabelCategories()
    for class_name in annotations_df.ObjType.unique():
        label_categories.add(class_name)

    return {AnnotationType.label: label_categories}


def load_errors_df(filepath: Path, observation_id: str):
    """ """
    errors_df = pd.read_csv(filepath)
    video_errors_df = errors_df[errors_df.obsID == observation_id]
    video_errors_df = video_errors_df.astype(
        {"mistaken_frame_start": "int32", "mistaken_frame_end": "int32"}
    )
    return video_errors_df


def extract_error_frames(video_errors_df) -> list[int]:
    """
    Extract all unique frame indices that fall within mistaken frame ranges.

    Args:
        video_errors_df (pd.DataFrame): DataFrame with columns 'mistaken_frame_start' and 'mistaken_frame_end'.

    Returns:
        list[int]: Sorted list of unique frame indices containing errors.
    """
    all_error_frames = []

    for _, row in video_errors_df.iterrows():
        start_frame = int(row["mistaken_frame_start"])
        end_frame = int(row["mistaken_frame_end"])
        error_frames_in_row = list(range(start_frame, end_frame + 1))
        all_error_frames.extend(error_frames_in_row)

    unique_error_frames = sorted(set(all_error_frames))

    return unique_error_frames


def get_label_id(chunked_df, col_class_name, col_instance_id, obj_id, label_categories):
    """Get the label id corresponding to `obj_id`.
    Each object has a ground truth label (class name) and each label has a label id.
    """
    for col in [col_instance_id, col_class_name]:
        if col not in chunked_df.columns:
            raise KeyError(
                f"Column '{col}' not found in DataFrame. "
                f"Available columns: {list(chunked_df.columns)}"
            )

    matching_rows = chunked_df.loc[
        chunked_df[col_instance_id] == str(obj_id), col_class_name
    ]

    if matching_rows.empty:
        raise ValueError(
            f"Object ID '{obj_id}' not found in DataFrame. "
            f"Available IDs: {chunked_df[col_instance_id].unique().tolist()}"
        )
    label_str = matching_rows.iloc[0]

    if pd.isna(label_str):
        raise ValueError(f"Object ID '{obj_id}' has null/NaN class label")

    # Find label in categories
    label_result = label_categories[AnnotationType.label].find(label_str)

    if not label_result or label_result[0] is None:
        raise ValueError(
            f"Label '{label_str}' not found in label categories. "
            f"Available labels: {[cat.name for cat in label_categories[AnnotationType.label]]}"
        )

    label_id = label_result[0]
    return label_id


def _get_frame_filename(
    extracted_frame_idx: int, filename_num_zeros: int, extension: str = ".jpg"
) -> str:
    """
    Some extracted frames have a variable length zero-padded length.
    For the focal follow data is 4, for the stationary data is 5.
    """
    return f"{extracted_frame_idx + 1:0{filename_num_zeros}d}{extension}"


def exp_smooth(measurement: float, prev_value: float, beta: float = 0.8) -> float:
    """ """
    return beta * measurement + (1.0 - beta) * prev_value


def detrend_by_differencing(X):
    """Detrend a timeseries by differencing"""
    diff = list()
    for i in range(1, len(X)):
        value = X[i] - X[i - 1]
        diff.append(value)
    return diff
