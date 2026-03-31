# Standard Library imports
import pickle
from enum import IntEnum
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from collections import defaultdict

# External imports
import cv2
import datumaro.components.dataset_base
from datumaro.components.annotation import AnnotationType, LabelCategories
import numpy as np
import pandas as pd
import torch
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from thefuzz import fuzz

# Local imports
from blob import BlobInfo
from configuration import ParsedObservationID, ManualObservationID

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
        print(
            f"enter_frame: {enter_frame.shape}, {enter_frame[instance_id].values}\n",
            enter_frame,
        )
        print(
            f"exit_frame: {exit_frame.shape}, {exit_frame[instance_id].values}\n",
            exit_frame,
        )
        raise RuntimeError(
            f"An {instance_id} does not have both an enter and exit point!"
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
    print("Masks loaded correctly!")
    return masks


def load_annotations(annotations_filepath: Path):
    """Load annotations from a .npy file as a Pandas DataFrame"""
    print(f"Loading annotations file: '{annotations_filepath}'")
    with open(annotations_filepath, "rb") as annotations_file:
        annotations = np.load(annotations_file, allow_pickle=True)

    annotations_df = pd.DataFrame(list(annotations))

    # Ensure columns have a correct type
    annotations_df["ObjID"] = annotations_df["ObjID"].astype(str)
    annotations_df["ObjType"] = annotations_df["ObjType"].astype(str)
    annotations_df["ClickType"] = annotations_df["ClickType"].astype(int)
    annotations_df["Frame"] = annotations_df["Frame"].astype(int)

    def _parse_location(raw_location):
        """Convert Location from ndarray (as stored in .npy) to list[float]"""
        if isinstance(raw_location, (np.ndarray, list, tuple)):
            return [float(x) for x in raw_location]
        elif isinstance(raw_location, str):
            # Fallback, parse string representation, e.g. "[2503.934, 1143.529]"
            return [float(x) for x in raw_location.strip("[] ").split(",")]
        else:
            return None

    annotations_df["Location"] = annotations_df["Location"].apply(_parse_location)
    annotations_df = annotations_df.sort_values(by=["Frame"]).reset_index(drop=True)

    # Clean trailing whitespaces from ObjType and ObjID columns
    for col in ["ObjType", "ObjID"]:
        if col in annotations_df.columns and annotations_df[col].dtype == "object":
            annotations_df[col] = annotations_df[col].str.strip()

    print("Annotations loaded correctly!")
    return annotations_df


def load_categories(
    annotations_df: pd.DataFrame,
) -> datumaro.components.dataset_base.CategoriesInfo:
    """Load Datumaro categories data"""
    label_categories = LabelCategories(attributes={"ObjID"})
    for class_name in annotations_df.ObjType.unique():
        label_categories.add(class_name)

    return {AnnotationType.label: label_categories}


def load_errors_df(
    filepath: Path, observation_id: ParsedObservationID | ManualObservationID
) -> Optional[pd.DataFrame]:
    """
    Load and extract the error entries associated with a specific observation.

     This function reads a CSV errors file, filters it to retain only the rows
     that correspond to the given `observation_id`, and returns the resulting
     DataFrame.

     Parameters
     ----------
     filepath : Path
         Path to the CSV file containing all recorded errors.
     observation_id : ParsedObservationID | ManualObservationID
         Observation identifier used to locate the relevant subset of
         rows in the errors DataFrame.

     Returns
     -------
     pandas.DataFrame or None
        A DataFrame containing only the error rows associated with the given
        observation, or ``None`` if no matching entries are found.

     Side Effects
     ------------
     Prints diagnostic messages showing the file being read and the extracted
     subset of errors for the observation.
    """
    print(f"Reading errors from file '{filepath.absolute()}'")
    errors_df = pd.read_csv(filepath)
    observations_errors = find_obsId_in_errors_file(observation_id, errors_df)

    if isinstance(observations_errors, pd.DataFrame):
        observations_errors = observations_errors.astype(
            {"mistaken_frame_start": "int32", "mistaken_frame_end": "int32"}
        )
        print(
            f"Errors DataFrame for observation '{observation_id.to_str()}':\n",
            observations_errors,
        )
        return observations_errors


def extract_error_frames(
    video_errors_df: Optional[pd.DataFrame],
    include_end: bool = True,
    error_type: list | str | None = None,
) -> list[int]:
    """
    Extract all unique frame indices that fall within mistaken frame ranges.

    Args:
        video_errors_df (pd.DataFrame): DataFrame with columns 'mistaken_frame_start' and 'mistaken_frame_end'.

    Returns:
        list[int]: Sorted list of unique frame indices containing errors.
        include_end: Whether to include or not the frame at mistaken_frame_end
    """
    if video_errors_df is None:
        return []

    # If an error_type of interest is provided, keep rows with such error_type
    if error_type is not None:
        error_type = list(error_type)
        video_errors_df = video_errors_df.loc[
            video_errors_df.error_type.isin(error_type)
        ]

    all_error_frames = []

    for _, row in video_errors_df.iterrows():
        start_frame = int(row["mistaken_frame_start"])
        end_frame = int(row["mistaken_frame_end"]) + (1 if include_end else 0)
        error_frames_in_row = list(range(start_frame, end_frame))
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
            f"ObjID '{obj_id}' not found in annotations file. "
            f"Available ObjIDs in annotations file: {chunked_df[col_instance_id].unique().tolist()}"
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


def kpss_test(timeseries, significance_level=0.05):
    """
    Modified from:
    https://www.statsmodels.org/dev/examples/notebooks/generated/stationarity_detrending_adf_kpss.html
    """
    kpsstest = kpss(timeseries, regression="c", nlags="auto")
    kpss_output = pd.Series(
        kpsstest[0:3], index=["Test Statistic", "p-value", "Lags Used"]
    )

    if kpss_output["p-value"] < significance_level:
        return "non-stationary"
    else:
        return "stationary"


def adf_test(timeseries, significance_level=0.05):
    """
    Modified from:
    https://www.statsmodels.org/dev/examples/notebooks/generated/stationarity_detrending_adf_kpss.html
    """
    dftest = adfuller(timeseries, autolag="AIC")
    dfoutput = pd.Series(
        dftest[0:4],
        index=[
            "Test Statistic",
            "p-value",
            "#Lags Used",
            "Number of Observations Used",
        ],
    )

    if dfoutput["p-value"] > significance_level:
        return "non-stationary"
    else:
        return "stationary"


@dataclass
class ObservationIdSimilarity:
    """
    Stores similarity comparison results between tested observation IDs and existent observation IDs.
    """

    # tested_obs_id_str : (available_obs_id_str, score)
    comparisons: dict[str, list[tuple[str, int]]]

    def __str__(self):
        lines = []
        for tested, sims in self.comparisons.items():
            lines.append(f"Tested format: '{tested}'")

            # Sort highest to lowest similarity
            sims_sorted = sorted(sims, key=lambda x: x[1], reverse=True)

            # Show top 5
            for obs_id, score in sims_sorted[:5]:
                lines.append(f"  - Similar to '{obs_id}' (score: {score})")

            lines.append("")  # blank line between groups

        return "\n".join(lines)


def find_obsId_in_errors_file(
    obs_id_object: ParsedObservationID | ManualObservationID, errors_df: pd.DataFrame
) -> pd.DataFrame | ObservationIdSimilarity:
    """
    Try multiple string formats of an observation ID to find a match in the
    errors CSV. If a match is found, return the matching rows. If no exact
    match is found, return a list of similarity scores between the generated
    obsID strings and all available obsIDs in the file.

    For ManualObservationID: Uses the exact errors_obs_id without any format variations.
    For ParsedObservationID: Tries multiple date formats and variations.
    """
    # To store the similarity score of each available obsID in the CSV
    # and return it if no match was found
    comparisons: dict[str, list[tuple[str, int]]] = defaultdict(list)

    # Handle ManualObservationID - use exact string provided
    if isinstance(obs_id_object, ManualObservationID):
        obs_id_str = obs_id_object.errors_obs_id
        matching_rows = errors_df[errors_df.obsID == obs_id_str]
        if not matching_rows.empty:
            return matching_rows

        # If no exact match, compute similarities for error reporting
        for available_obs_id in errors_df.obsID.unique():
            score = fuzz.ratio(obs_id_str, available_obs_id)
            comparisons[obs_id_str].append((available_obs_id, score))

        return ObservationIdSimilarity(comparisons)

    # Handle ParsedObservationID - try multiple format variations
    for date_format in ["%m%d%Y", "%m-%d-%Y", "%m%d%y"]:
        for has_token in (True, False):
            obs_id_str = obs_id_object.to_str(
                has_observer=False,
                has_monopod_token=has_token,
                output_date_format=date_format,
            )

            # Check exact match
            matching_rows = errors_df[errors_df.obsID == obs_id_str]
            if not matching_rows.empty:
                return matching_rows

            # Collect similarity info for this version of the observation id
            for available_obs_id in errors_df.obsID.unique():
                score = fuzz.ratio(obs_id_str, available_obs_id)
                comparisons[obs_id_str].append((available_obs_id, score))

    return ObservationIdSimilarity(comparisons)


def find_existing_file(
    base_path: Path, obs_id_object: ParsedObservationID | ManualObservationID, suffix: str
) -> Optional[Path]:
    """
    Find a file with the given suffix for the observation ID.

    For ManualObservationID: Uses specific obs_id based on suffix type.
    For ParsedObservationID: Tries multiple format variations.
    """
    checked_files = []

    # Handle ManualObservationID - use exact string provided
    if isinstance(obs_id_object, ManualObservationID):
        # Determine which obs_id to use based on the suffix
        if suffix == "_annotations.npy":
            filename = obs_id_object.annotations_filename
        elif suffix == "_masks.pkl":
            filename = obs_id_object.masks_filename
        else:
            raise Exception("Unexpected suffix")

        candidate = base_path / filename
        checked_files.append(str(candidate))

        if candidate.exists() and candidate.is_file():
            return candidate

        print("Target file wasn't found, the following files were checked:")
        for s in checked_files:
            print(f" - '{s}'")
        return None

    # Handle ParsedObservationID - try multiple format variations
    for date_format in ["%m%d%Y", "%m-%d-%Y", "%m%d%y"]:
        for has_token in (True, False):
            obs_id_str = obs_id_object.to_str(
                has_monopod_token=has_token,
                output_date_format=date_format,
            )
            candidate = base_path / f"{obs_id_str}{suffix}"
            checked_files.append(str(candidate))

            if candidate.exists():
                if candidate.is_file():
                    return candidate

    print("Target file wasn't found, the following files were checked:")
    for s in checked_files:
        print(f" - '{s}'")
    return None


def find_annot(base_path: Path, obs_id_object: ParsedObservationID | ManualObservationID):
    """Find annotations file for the given observation ID."""
    return find_existing_file(base_path, obs_id_object, "_annotations.npy")


def find_masks(base_path: Path, obs_id_object: ParsedObservationID | ManualObservationID):
    """Find masks file for the given observation ID."""
    return find_existing_file(base_path, obs_id_object, "_masks.pkl")


def next_run_dir(root: Path, prefix: str = "run_") -> Path:
    """
    Create and return the next 'run_N' directory under `root`.
    If no run_N directories exist, creates run_1.
    """
    root.mkdir(parents=True, exist_ok=True)

    run_numbers = []
    for item in root.iterdir():
        if not item.is_dir():
            continue

        if not item.name.startswith(prefix):
            continue

        suffix = item.name.replace(prefix, "")
        if suffix.isdigit():
            run_numbers.append(int(suffix))

    next_number = 1
    if run_numbers:
        next_number = max(run_numbers) + 1

    run_dir = root / f"run_{next_number}"
    run_dir.mkdir(parents=True, exist_ok=False)

    return run_dir
