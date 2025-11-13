# Standard Library imports
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

# Naming convention for the ObservationID of stationary data.
#
# - On the SAM2_Errors, this is specified as:
#     <monopod>_<date-recorded>_<site>_<direction>_<A/B>_<Left/Right>_<videoname>
#
# - For the masks and results videos, this is specified as:
#     <observer>_<date>_<site>_<direction>_<A/B>_<Left/Right>_<videoname>_<colorcorrection>
#
#   NOTE: I've seen variations for the Observation ID on the masks files, like:
#     <observer>_monopod_<date><...>
#     <videoname>
#     <videoname>_synced
#
# - Date recorded may be in the format of MM-DD-YYYY or MMDDYY
#
# - If no color correction was used, it will be blank in the observationID for the masks/results.
#
# For some videos, color correction methods (LACC or retinex) was used on the frames.
# For the SAM2_Errors.csv, that will be specified in the Enhancement column of the SAM2 errors. If a color
# correction was used, the corresponding folder of frames will be video_name_LACC or video_name_ret
#
# Examples:
# JGL_monopod_05302024_site5_east_B_Right_GX030843_masks.pkl
# JGL_monopod_05302024_site5_east_B_Right_GX030843_annotations.npy

DATA_ROOT_PATH = Path(<FILL_ME>)


# Constants for default values
class AnomalyDefaults:
    SPIKE_CHANGE_THRESHOLD = 2.0
    DISPLACEMENT_THRESHOLD_PX = 300
    ZSCORE_WINDOW = 10
    AREA_MAX_THRESHOLD = 250000


@dataclass(frozen=True)
class ParsedObservationID:
    """
    - observer:
    - date: MM-DD-YYYY format
    - site: e.g. "site5"
    - direction: "east"/"west"/"north"/"south"
    - ab: "A" or "B"
    - side: "Left" or "Right"
    - videoname: e.g. "GX030843"
    """

    observer: str
    date: str
    site: str
    direction: str
    ab: str
    side: str
    videoname: str

    accepted_date_format = "%m-%d-%Y"

    def __post_init__(self):
        """ """

        # Validate date format
        try:
            datetime.strptime(self.date, self.accepted_date_format)
        except ValueError:
            raise ValueError(
                f"Invalid date format for {self.date!r}. Expected MM-DD-YYYY."
            )

        # Validate A/B values
        if self.ab.lower() not in {"a", "b"}:
            raise ValueError(
                f"Invalid value for ab: {self.ab!r}. Must be 'a' or 'b' (case-insensitive)."
            )

        # Validate direction
        if self.direction.lower() not in {"east", "west", "north", "south"}:
            raise ValueError(
                f"Invalid direction: {self.direction!r}. Must be 'east', 'west', 'north' or 'south' (case-insensitive)."
            )

        # Validate side
        if self.side.lower() not in {"left", "right"}:
            raise ValueError(
                f"Invalid side: {self.direction!r}. Must be 'left' or 'right' (case-insensitive)."
            )

    def to_str(
        self,
        has_observer=True,
        has_monopod_token=False,
        output_date_format: Literal["%m%d%Y", "%m-%d-%Y", "%m%d%y"] = "%m-%d-%Y",
    ) -> str:
        """
        Reconstruct the 'masks/results' style ID:
        <observer>_<date>_<site>_<direction>_<A/B>_<Left/Right>_<videoname>_<colorcorrection?>
        (date uses YYYYMMDD if we can; otherwise date_raw)
        """
        parts = []
        if has_observer:
            parts.append(self.observer)
        if has_monopod_token:
            parts.append("monopod")
        parts.append(
            datetime.strptime(self.date, self.accepted_date_format).strftime(
                output_date_format
            )
        )
        parts.append(self.site)
        parts.append(self.direction.lower())
        parts.append(self.ab.upper())
        parts.append(self.side.capitalize())
        parts.append(self.videoname.upper())
        return "_".join(parts)


class Config:
    # Path to CSV errors file
    errors_csv_filepath = DATA_ROOT_PATH / "SAM2_errors.csv"

    # Path towards directory containing all the annotation files
    annot_path = DATA_ROOT_PATH / "location_annotations"

    # Path towards directory containing all the masks files
    masks_path = DATA_ROOT_PATH / "SAM2_masks"

    # Path towards output exported CVAT and YOLO datasets
    output_path = DATA_ROOT_PATH / "exports"

    # Minimum bounding box area (in pixels) required for a blob to be considered valid.
    # Blobs with a smaller area are ignored as likely noise or irrelevant detections.
    area_threshold = 200  # px

    # Minimum acceptable dimension (either width or height, in pixels) for a blob.
    # Blobs smaller than this in width or height are discarded.
    min_size_threshold = 20  # px

    # Use 4 for focal follow data, 5 for stationary data.
    number_of_zeros = 5

    # Video frame from where to start the extraction
    start_frame: int = 0

    # Number of frames to extract from each video
    max_frames: int | None = None

    # Length of the Fish Tracker window tracking blob metrics
    window_size: int = 10

    # Masked fish crop classifier confidence threshold
    classifier_conf: float = 0.25

    obsId_to_folder_map: dict[ParsedObservationID, Path] = {
        ParsedObservationID(
            observer="MLM",
            date="04-27-2024",
            site="site1",
            direction="east",
            ab="B",
            side="Right",
            videoname="GX040093",
        ): DATA_ROOT_PATH / "frames" / "GX040093",
    }

    anomaly_rules = [
        {
            "type": "spike",
            "metric": "area",
            "threshold": AnomalyDefaults.SPIKE_CHANGE_THRESHOLD,
            "description": "Detects sudden increases in area",
        },
        {
            "type": "spike",
            "metric": "solidity",
            "threshold": AnomalyDefaults.SPIKE_CHANGE_THRESHOLD,
            "description": "Detects sudden changes in solidity",
        },
        {
            "type": "spike",
            "metric": "compactness",
            "threshold": AnomalyDefaults.SPIKE_CHANGE_THRESHOLD,
            "description": "Detects sudden changes in compactness",
        },
        {
            "type": "absolute",
            "metric": "area",
            "max_value": AnomalyDefaults.AREA_MAX_THRESHOLD,
            "description": "Flags areas exceeding maximum threshold",
        },
        {
            "type": "displacement",
            "threshold": AnomalyDefaults.DISPLACEMENT_THRESHOLD_PX,
            "unit": "pixels",
            "description": "Detects excessive positional displacement",
        },
        {
            "type": "zscore",
            "metric": "area",
            "window": AnomalyDefaults.ZSCORE_WINDOW,
            "description": "Statistical outlier detection using z-score",
        },
    ]


class ClassifierConfig:
    # Path towards raw classifier fish crops
    data_path = Config.output_path / "extracted_crops"

    # Path towards YOLO classification dataset
    yolo_dataset_path = (
        Config.output_path / "extracted_crops" / "yolo_classification_dataset"
    )

    # Path to classifier weights for detecting correctly masked fish
    model_weights_path = Path(<FILL_ME>)
