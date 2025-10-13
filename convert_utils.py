# Standard Library imports
import pickle
from pathlib import Path

# External imports
import numpy as np
import pandas as pd
import torch
import cv2
import datumaro.components.media
import datumaro.components.dataset
import datumaro.components.dataset_base
from datumaro.components.annotation import AnnotationType, Bbox, LabelCategories

# Type aliases for the masks file
FrameIndex = int
ObjectIndex = int
MasksType = dict[FrameIndex, dict[ObjectIndex, torch.Tensor]]


def build_datumaro_dataset(
    masks: MasksType,
    all_error_frames: list[int],
    chunked_df: pd.DataFrame,
    label_categories: datumaro.components.dataset_base.CategoriesInfo,
    images_path: Path,
    col_class_name="ObjType",
    col_instance_id="ObjID",
    filename_num_zeros=4,
    area_threshold=100,
) -> datumaro.components.dataset.Dataset:
    """
    Build a Datumaro dataset from instance segmentation masks and associated metadata.

    This function processes frame-by-frame instance masks, converts them to bounding boxes,
    and creates a Datumaro dataset suitable for object detection tasks. Frames with errors
    are skipped, and empty masks are tracked for diagnostic purposes.

    Parameters
    ----------
    masks : MasksType
        Dictionary mapping extracted frame indices (0-indexed) to frame-level masks.
        Each frame contains a dict of object IDs to sparse tensor masks.
    all_error_frames : list[int]
        List of frame indices (0-indexed) that contain errors in the CSV and should be excluded from the dataset.
    chunked_df : pd.DataFrame
        DataFrame containing the enter and exit frame values for each object instance
    label_categories : datumaro.components.dataset_base.CategoriesInfo
        Datumaro categories information containing label definitions.
    images_path : Path
        Path to the directory containing extracted frame images.
    col_class_name : str, optional
        Name of the column containing object class labels, by default "ObjType".
    col_instance_id : str, optional
        Name of the column containing object instance IDs, by default "ObjID".
    filename_num_zeros : int, optional
        Number of leading zeros for image filenames (e.g., 4 for "0001.jpg"), by default 4.
        Use 4 for focal follow data, 5 for stationary data.
    area_threshold: float, optional
        Minimum bounding box area required for a blob to be considered.

    Returns
    -------
    datumaro.components.dataset.Dataset
        A Datumaro Dataset object containing DatasetItems with bounding box annotations
        for each valid frame.

    Notes
    -----
    - Frame indices in masks are 0-indexed (extracted frame index).
    - Image filenames are 1-indexed and zero-padded (e.g., frame 0 → "0001.jpg").
    - Bounding boxes below the area threshold (defined in config) are filtered out.
    - All generated DatasetItems are assigned to the "train" subset.
    """

    dataset_items: list[datumaro.components.dataset_base.DatasetItem] = []
    count_empty_instance_masks = 0
    count_frames_with_errors = 0
    for extracted_frame_idx in masks.keys():
        if extracted_frame_idx in all_error_frames:
            print(f"Frame {extracted_frame_idx} has errors in the CSV. Skipping.")
            count_frames_with_errors += 1
            continue

        # Load available masks in this frame
        frame_masks = masks[extracted_frame_idx]

        bounding_boxes = []
        for obj_id in frame_masks.keys():
            # Load mask corresponding to one object (can be multiple blobs!)
            sparse_tensor = frame_masks[obj_id]

            if is_empty_sparse_tensor(sparse_tensor):
                count_empty_instance_masks += 1

            # Convert sparse tensor into a dense numpy array
            dense_tensor = sparse_tensor.to_dense()
            object_mask = torch_to_cv2(dense_tensor, is_mask=True)

            # Convert segmentation masks to bounding boxes
            boxes = get_bboxes_from_mask(object_mask)
            boxes = filter_bboxes(boxes, area_threshold=area_threshold)

            # Associate label with bounding boxes
            label_str = chunked_df.loc[
                chunked_df[col_instance_id] == str(obj_id), col_class_name
            ].iloc[0]
            label_id = label_categories[AnnotationType.label].find(label_str)[0]

            for bbox in boxes:
                x1, y1, w, h = bbox
                bounding_boxes.append(Bbox(x1, y1, w, h, label=label_id))

        if len(bounding_boxes) == 0:
            print(f"Frame {extracted_frame_idx} doesn't have bounding boxes!")

        # The number of zeros is important. For the focal follow data is 4, for the stationary data is 5
        file_id = str(extracted_frame_idx + 1).zfill(filename_num_zeros)
        media = datumaro.components.media.Image.from_file(
            str(images_path / f"{file_id}.jpg")
        )

        dataset_items.append(
            datumaro.components.dataset_base.DatasetItem(
                id=file_id,
                subset="train",
                media=media,
                annotations=bounding_boxes,
                attributes={"frame": extracted_frame_idx},
            )
        )

    dataset = datumaro.components.dataset.Dataset.from_iterable(
        dataset_items, categories=label_categories
    )

    print(f"Count of empty instance masks: {count_empty_instance_masks}")
    print(f"Count of frames with errors: {count_frames_with_errors}")

    return dataset


def torch_to_cv2(image: torch.Tensor, is_mask=False) -> np.ndarray:
    """Convert a PyTorch image tensor to an OpenCV (Numpy) image."""
    image = image.detach()

    if is_mask:
        if image.ndim == 3:
            image = image.squeeze()
        image = image.to(torch.uint8)

    else:
        if image.ndim == 4:
            image = image.squeeze()
        image = image.permute(1, 2, 0)

    return image.cpu().numpy()


def is_empty_sparse_tensor(sparse_tensor: torch.Tensor):
    """ """
    return sparse_tensor.values().numel() == 0


def get_bboxes_from_mask(mask) -> list[tuple[int, ...]]:
    """ """
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    bboxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        bboxes.append((x, y, w, h))

    return bboxes


def filter_bboxes(
    bboxes: list[tuple[int, ...]], area_threshold: float = 100
) -> list[tuple[int, ...]]:
    """ """
    bounding_boxes = []
    for bbox in bboxes:
        x, y, w, h = bbox
        area = w * h
        if area > area_threshold:
            bounding_boxes.append((x, y, w, h))
        else:
            print(
                f"Filtering out box due to area threshold. Box area: {area:.1f}, threshold: {area_threshold}"
            )
    return bounding_boxes


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

    Using `click_type_name` column values of 3 and 4, obtain the enter and exit frame values for each `instance_id`.
    """

    # For each instance_id get frame where the object enters the scene
    enter_frame = df[df[click_type_name] == 3][[instance_id, class_name, frame_name]]
    enter_frame = enter_frame.sort_values(by=[instance_id, frame_name], ascending=True)

    # For each instance_id get frame where the object exits the scene
    exit_frame = df[df[click_type_name] == 4][[instance_id, class_name, frame_name]]
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


def load_masks(masks_filepath: Path):
    """ """
    print(f"Loading masks file: '{masks_filepath}'")
    with open(masks_filepath, "rb") as masks_file:
        masks: MasksType = pickle.load(masks_file)
    return masks


def load_annotations(annotations_filepath: Path):
    """ """
    print(f"Loading annotations file: '{annotations_filepath}'")
    with open(annotations_filepath, "rb") as annotations_file:
        annotations = np.load(annotations_file, allow_pickle=True)

    annotations_df = pd.DataFrame(list(annotations))
    annotations_df = annotations_df.sort_values(by=["Frame"]).reset_index(drop=True)

    return annotations_df


def load_categories(
    annotations_df: pd.DataFrame,
) -> datumaro.components.dataset_base.CategoriesInfo:
    """ """
    label_categories = LabelCategories()
    for class_name in annotations_df.ObjType.unique():
        label_categories.add(class_name)

    return {AnnotationType.label: label_categories}


def load_errors_df(filepath: Path, video_id: str):
    """ """
    errors_df = pd.read_csv(filepath)
    video_errors_df = errors_df[errors_df.obsID == video_id]
    video_errors_df = video_errors_df.astype(
        {"mistaken_frame_start": "int32", "mistaken_frame_end": "int32"}
    )
    return video_errors_df


def extract_error_frames(video_errors_df):
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
