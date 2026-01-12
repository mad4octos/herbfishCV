# Standard Library imports
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

"""
Note: The classes ParsedObservationID and ManualObservationID are here because they are needed for the configuration 
and I prefer to keep the configuration file self-contained to avoid potential circular dependencies.
"""


DATA_ROOT_PATH = Path(<FILL_ME>)


# Constants for default values
class AnomalyDefaults:
    SPIKE_CHANGE_THRESHOLD = 2.0
    DISPLACEMENT_THRESHOLD_PX = 300
    ZSCORE_WINDOW = 10
    AREA_MAX_THRESHOLD = 250000


@dataclass(frozen=True)
class ManualObservationID:
    """
    Manual observation ID specification for pipelines where observation IDs
    are already known and don't need parsing/formatting.

    This allows you to specify exact observation ID strings as they appear in:
    - The errors CSV file (errors_obs_id)
    - The masks file (masks_filename )
    - The annotations file (annotations_filename)

    Parameters
    ----------
    errors_obs_id : str
        The exact observation ID string as it appears in the errors CSV file
    masks_filename : str
        The base filename for the masks file (e.g., "video_name_masks.pkl")
    annotations_filename : str
        The base filename for the annotations file (e.g., "video_name_annotations.npy")
    display_name : str, optional
        A friendly name for display/logging purposes. If not provided, uses errors_obs_id

    Example
    -------
    ManualObservationID(
        errors_obs_id="JM_060724_152_playa_largu_scuba_TPScv_L",
        masks_filename="CR_JM_060724_152_playa_largu_scuba_TPScv_L_masks.pkl",
        annotations_filename="CR_JM_060724_152_playa_largu_scuba_TPScv_L_annotations.npy",
    )
    """

    errors_obs_id: str
    masks_filename: str
    annotations_filename: str
    display_name: str | None = None

    def __post_init__(self):
        """Set display_name to errors_obs_id if not provided."""
        if self.display_name is None:
            object.__setattr__(self, "display_name", self.errors_obs_id)

    def to_str(self) -> str:
        """Return display name for consistency with ParsedObservationID interface."""
        return (
            self.display_name if self.display_name is not None else self.errors_obs_id
        )


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
        <observer>_<date>_<site>_<direction>_<A|B>_<Left|Right>_<videoname>_<colorcorrection?>
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

    obsId_to_folder_map: dict[ParsedObservationID | ManualObservationID, Path] = {
        # ParsedObservationID(
        #     observer="MLM",
        #     date="04-27-2024",
        #     site="site1",
        #     direction="east",
        #     ab="B",
        #     side="Right",
        #     videoname="GX040093",
        # ): DATA_ROOT_PATH / "frames" / "GX040093",

        # ManualObservationID(
        #     errors_obs_id="JM_060724_152_playa_largu_scuba_TPScv_L",
        #     masks_filename ="CR_JM_060724_152_playa_largu_scuba_TPScv_L_mask.pkl",
        #     annotations_filename ="CR_JM_060724_152_playa_largu_scuba_TPScv_L_annotations.npy",
        # ): Path("/path/to/images/MH_JM_060724_152_L"),
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

    # Masked fish crop classifier confidence threshold
    classifier_conf: float = 0.25
