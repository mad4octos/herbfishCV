# Standard Library imports
import pickle
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Protocol, TypedDict

# External imports
import cv2
import datumaro.components.dataset
import datumaro.components.dataset_base
import datumaro.components.media
from datumaro.components.annotation import AnnotationType, LabelCategories
from datumaro.components.annotation import Bbox as DatumaroBbox
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch

# Local imports
from common import cv2_imshow, sparse_mask_tensor_to_dense_numpy, is_empty_sparse_tensor
from plot_utils import draw_mask_overlay


# Type aliases for the masks file
FrameIndex = int
ObjectIndex = int
MasksType = dict[FrameIndex, dict[ObjectIndex, torch.Tensor]]

ENTER_FRAME_MARKER = 3
EXIT_FRAME_MARKER = 4


class AnomalyDict(TypedDict):
    type: str
    value: float


@dataclass
class BlobInfo:
    blob_num: int
    labeled_mask: np.ndarray  # labeled mask where pixel == blob_num is this blob
    x: int
    y: int
    w: int
    h: int
    area: int

    @property
    def bbox_xywh(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)

    @property
    def bbox_xyxy(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.x + self.w, self.y + self.h)

    def crop_from_image(self, image: np.ndarray) -> np.ndarray:
        """Return the axis-aligned crop of the input image covering the blob bbox."""
        return image[self.y : self.y + self.h, self.x : self.x + self.w]

    def masked_crop_from_image(self, image: np.ndarray, mask_background=True):
        """Mask and crop an image.

        This function can either remove the background (default) or highlight the foreground (the blob)."""

        image = np.copy(image)
        if mask_background:
            image[self.labeled_mask != self.blob_num] = 0
        else:
            image = draw_mask_overlay(
                image, self.labeled_mask, self.blob_num, color=(0, 0, 255)
            )
        return self.crop_from_image(image)


class FishAnomalyRule(Protocol):
    """
    Protocol defining the interface for fish anomaly detection.

    An anomaly rule is a callable that decides whether an anomaly has occurred within the data tracked by a fish tracker
    """

    def __call__(self, fish_tracker: "FishTracker") -> Optional[AnomalyDict]: ...


@dataclass
class AreaChangeAnomaly:
    """Check for sudden area changes in the fish mask"""

    area_change_thresh: float = 0.3

    def __call__(self, fish_tracker: "FishTracker") -> Optional[AnomalyDict]:
        """Calculate percentage change in area between last two frames"""
        curr_area = fish_tracker.areas[-1]
        prev_area = fish_tracker.areas[-2]

        if prev_area > 0:
            area_change = (curr_area - prev_area) / prev_area
        else:
            area_change = 0

        if abs(area_change) > self.area_change_thresh:
            return {"type": "area_change", "value": area_change}

        return None


@dataclass
class LargeDisplacementAnomaly:
    """Check for sudden large movements in fish centroid position"""

    displacement_thresh: int = 200  # pixels

    def __call__(self, fish_tracker: "FishTracker") -> Optional[AnomalyDict]:
        """
        Detect anomalously large displacements between consecutive frames.
        """
        if len(fish_tracker.centroids_x) < 2 or len(fish_tracker.centroids_y) < 2:
            return None

        dx = fish_tracker.centroids_x[-1] - fish_tracker.centroids_x[-2]
        dy = fish_tracker.centroids_y[-1] - fish_tracker.centroids_y[-2]
        displacement = np.sqrt(dx**2 + dy**2)

        if displacement > self.displacement_thresh:
            return {"type": "large_displacement", "value": float(displacement)}


class BlobRule(Protocol):
    """
    Protocol defining the interface for blob filtering rules.

    A blob rule is a callable that decides whether a given blob should be kept or discarded, typically to remove noise
    or irrelevant detections.
    """

    def __call__(self, blob: BlobInfo) -> bool: ...

    def explain(self, blob: BlobInfo) -> str: ...


@dataclass
class MinAreaRule:
    min_area: float

    def __call__(self, blob: BlobInfo) -> bool:
        return blob.area >= self.min_area

    def explain(self, blob: BlobInfo) -> str:
        return f"area {blob.area} < {self.min_area}"


@dataclass
class MinSizeRule:
    min_size: float

    def __call__(self, blob: BlobInfo) -> bool:
        return blob.w >= self.min_size and blob.h >= self.min_size

    def explain(self, blob: BlobInfo) -> str:
        return f"size {blob.w}x{blob.h} < {self.min_size}"


@dataclass
class BlobMetrics:
    """Container for blob property measurements at a single point in time."""

    area: int
    centroid_x: float
    centroid_y: float
    bbox_w: int
    bbox_h: int
    extent: float  # area / bbox area


class FishTracker:
    def __init__(self, obj_id, window_size=20):
        self.id = obj_id
        self.window_size = window_size

        # Number of cycles elapsed since the last update.
        # - This counter is incremented during each call to predict().
        # - It resets to 0 whenever update() is called.
        # - The tracker is removed after certain number of cycles without updates
        self.cycles_since_update = 0

        # Bounded deque: discards items from the opposite end when full.
        # Time series of blob properties
        self.areas: deque = deque(maxlen=window_size)
        self.centroids_x: deque = deque(maxlen=window_size)
        self.centroids_y: deque = deque(maxlen=window_size)
        self.bbox_widths: deque = deque(maxlen=window_size)
        self.bbox_heights: deque = deque(maxlen=window_size)
        self.extents: deque = deque(maxlen=window_size)
        self.anomaly_checks: Iterable[FishAnomalyRule] = [
            AreaChangeAnomaly(),
            LargeDisplacementAnomaly(),
        ]

    def predict(self) -> dict:
        """
        Analyze tracked properties and detect anomalies.
        """
        self.cycles_since_update += 1

        results = {
            "fish_id": self.id,
            "cycles_since_update": self.cycles_since_update,
            "anomalies": [],
        }

        if len(self.areas) < 2:
            return results

        for annomaly_check in self.anomaly_checks:
            if (annomaly := annomaly_check(self)) is not None:
                results["anomalies"].append(annomaly)

        return results

    def update(self, blobs: list[BlobInfo]):
        """ """
        self.cycles_since_update = 0

        if not blobs:
            print("No blobs received... isn't that weird?")
            return

        # Track only the dominant blob (largest by area)
        dominant_blob = max(blobs, key=lambda b: b.area)
        metrics = self._extract_metrics(dominant_blob)

        self.areas.append(metrics.area)
        self.centroids_x.append(metrics.centroid_x)
        self.centroids_y.append(metrics.centroid_y)
        self.bbox_widths.append(metrics.bbox_w)
        self.bbox_heights.append(metrics.bbox_h)
        self.extents.append(metrics.extent)

    @staticmethod
    def _extract_metrics(blob) -> BlobMetrics:
        """Extract metrics from a BlobInfo object."""
        # TODO: the centroid is only valid for the stationary case
        # Calculate centroid from the labeled mask
        y_coords, x_coords = np.where(blob.labeled_mask == blob.blob_num)
        centroid_x = (
            float(np.mean(x_coords)) if len(x_coords) > 0 else blob.x + blob.w / 2
        )
        centroid_y = (
            float(np.mean(y_coords)) if len(y_coords) > 0 else blob.y + blob.h / 2
        )

        # Extent: area / bbox area
        # Extent - The proportion of the pixels in the bounding box that are also in the region.
        # Computed as the Area divided by the area of the bounding box.
        bbox_area = blob.w * blob.h
        extent = blob.area / bbox_area if bbox_area > 0 else 0.0

        return BlobMetrics(
            area=blob.area,
            centroid_x=centroid_x,
            centroid_y=centroid_y,
            bbox_w=blob.w,
            bbox_h=blob.h,
            extent=extent,
        )

    def get_summary(self) -> dict:
        """Get current summary of tracked properties."""
        if not self.areas:
            return {"status": "no_data"}

        return {
            "fish_id": self.id,
            "current_area": self.areas[-1],
            "area_mean": float(np.mean(list(self.areas))),
            "area_variance": float(np.var(list(self.areas)))
            if len(self.areas) > 1
            else 0.0,
            "num_frames_tracked": len(self.areas),
        }

    def __str__(self) -> str:
        summary = self.get_summary()
        return (
            f"FishTracker(id={self.id}, cycles_since_update={self.cycles_since_update}, "
            f"window_size={self.window_size}, frames_tracked={summary['num_frames_tracked']})"
        )

    def __repr__(self) -> str:
        return self.__str__()


class FishBlobsStateTracker:
    """
    Main tracker that manages multiple FishTracker instances.

    Responsibilities:
    - Create and update individual fish trackers
    - Run predictions on all trackers
    - Remove inactive trackers after a threshold of cycles without updates
    """

    def __init__(self, max_cycles_without_update=3, window_size=20):
        """ """
        self.fishes: dict[int, FishTracker] = {}
        self.max_cycles_without_update = max_cycles_without_update
        self.wearing_state_window_size = window_size

    def update(self, obj_id, data: list[tuple[BlobInfo, float, str, int]]):
        """
        TODO: I don't need all the extra info in data, only BlobInfo
        Update a fish's status or create a new fish tracker if not exists.
        """
        blobs = [d[0] for d in data]
        if not blobs:
            return

        if obj_id not in self.fishes:
            self.fishes[obj_id] = FishTracker(
                obj_id, window_size=self.wearing_state_window_size
            )

        self.fishes[obj_id].update(blobs)

    def predict_all(self) -> dict[int, dict]:
        """
        Run predictions on all active trackers.

        Returns:
            Dictionary mapping fish IDs to their prediction results
        """
        results = {}
        for obj_id in self.fishes.keys():
            results[obj_id] = self.fishes[obj_id].predict()
        return results

    def filter_dead_trackers(self):
        """ """
        removed_ids = []
        alive_trackers = {}

        for obj_id, tracker in self.fishes.items():
            if tracker.cycles_since_update < self.max_cycles_without_update:
                alive_trackers[obj_id] = tracker
            else:
                removed_ids.append(obj_id)
                print(
                    f"Removing inactive tracker: fish_id={obj_id}, "
                    f"cycles_without_update={tracker.cycles_since_update}"
                )

        self.fishes = alive_trackers
        return removed_ids

    def get_summary(self) -> dict:
        """
        Get a summary of all active trackers.

        Returns:
            Dictionary with tracker statistics
        """
        return {
            "num_active_trackers": len(self.fishes),
            "fish_ids": list(self.fishes.keys()),
            "max_cycles_without_update": self.max_cycles_without_update,
        }

    def __str__(self) -> str:
        summary = self.get_summary()
        return (
            f"FishBlobsStateTracker(active={summary['num_active_trackers']}, "
            f"fish_ids={summary['fish_ids']})"
        )

    def __repr__(self) -> str:
        return self.__str__()


class DatumaroDatasetBuilder:
    """
    Build a Datumaro dataset from instance segmentation masks and associated metadata.

    This class processes frame-by-frame instance masks, converts them to bounding boxes,
    and creates a Datumaro dataset suitable for object detection tasks. Frames with errors
    are skipped, and empty masks are tracked for diagnostic purposes.
    """

    def __init__(
        self,
        masks: dict,
        all_error_frames: list[int],
        chunked_df: pd.DataFrame,
        label_categories: datumaro.components.dataset_base.CategoriesInfo,
        images_path: Path,
        col_class_name: str = "ObjType",
        col_instance_id: str = "ObjID",
        filename_num_zeros: int = 8,
        area_threshold: float = 100,
        min_size_threshold: float = 20,
        verbose: bool = False,
        classifier=None,
        target_class=None,
        classifier_conf: float = 0.5,
        video_writer: Optional[cv2.VideoWriter] = None,
        tracker=None,
        background_subtractor=None,
        debug=False,
        max_frames=None,
        start_frame=0,
        blob_rules: Optional[Iterable[BlobRule]] = None,
    ):
        """ """
        self.masks = masks
        self.all_error_frames = all_error_frames
        self.chunked_df = chunked_df
        self.label_categories = label_categories
        self.images_path = Path(images_path)
        self.col_class_name = col_class_name
        self.col_instance_id = col_instance_id
        self.filename_num_zeros = filename_num_zeros
        self.area_threshold = area_threshold
        self.min_size_threshold = min_size_threshold
        self.verbose = verbose
        self.classifier = classifier
        self.classifier_conf = classifier_conf
        self.video_writer = video_writer
        self.tracker = FishBlobsStateTracker()
        self.background_subtractor = background_subtractor
        self.dataset_items: list[datumaro.components.dataset_base.DatasetItem] = []
        self.count_empty_instance_masks = 0
        self.count_frames_with_errors = 0
        self.debug_viz = debug
        self.target_class = target_class
        self.max_frames = max_frames
        self.start_frame = start_frame

        if blob_rules is None:
            self.blob_rules: list[BlobRule] = [
                MinAreaRule(self.area_threshold),
                MinSizeRule(self.min_size_threshold),
            ]
        else:
            self.blob_rules = list(blob_rules)

        if tracker is None:
            print("No tracker will be used")

    def build(self) -> datumaro.components.dataset.Dataset:
        """
        Build and return the Datumaro dataset.

        Returns
        -------
        datumaro.components.dataset.Dataset
            A Datumaro Dataset object containing DatasetItems with bounding box annotations.
        """
        for extracted_frame_idx, frame_masks in tqdm(self.masks.items()):
            if extracted_frame_idx < self.start_frame:
                continue

            if (self.max_frames is not None) and (
                extracted_frame_idx >= (self.start_frame + self.max_frames)
            ):
                print("Braking because max number of frames has been reached.")
                break
            self._process_frame(extracted_frame_idx, frame_masks)

        dataset = datumaro.components.dataset.Dataset.from_iterable(
            self.dataset_items, categories=self.label_categories
        )

        if self.video_writer is not None:
            self.video_writer.release()
            print("Finished writing video")

        self._print_statistics()
        return dataset

    def _process_frame(self, extracted_frame_idx: int, frame_masks: dict) -> None:
        """
        Process a single frame and its associated masks.

        Parameters
        ----------
        extracted_frame_idx : int
            The extracted frame index (0-indexed).
        frame_masks : dict
            Dictionary mapping object IDs to sparse tensor masks.
        """
        if self.verbose:
            print(f"Frame {extracted_frame_idx}")

        if extracted_frame_idx in self.all_error_frames:
            print(f"Frame {extracted_frame_idx} has errors in the CSV. Skipping.")
            self.count_frames_with_errors += 1
            return

        filename = _get_frame_filename(extracted_frame_idx, self.filename_num_zeros)
        image_filepath = self.images_path / filename
        input_image = cv2.imread(str(image_filepath), cv2.IMREAD_COLOR)
        if input_image is None:
            raise FileNotFoundError("Frame file doesn't exist!")

        # TODO: move the tracker code out of _extract_datumaro_bounding_boxes
        bounding_boxes = self._extract_datumaro_bounding_boxes(input_image, frame_masks)

        results = self.tracker.predict_all()
        self.tracker.filter_dead_trackers()
        print("results: ", results)

        if self.debug_viz:
            cv2_imshow(input_image)

        if self.video_writer is not None:
            self.video_writer.write(input_image)

        if len(bounding_boxes) == 0 and self.verbose:
            print(f"Frame {extracted_frame_idx} doesn't have bounding boxes!")

        self.dataset_items.append(
            datumaro.components.dataset_base.DatasetItem(
                id=filename.split(".")[0],
                subset="train",
                media=datumaro.components.media.Image.from_file(str(image_filepath)),
                annotations=bounding_boxes,
                attributes={"frame": extracted_frame_idx},
            )
        )

    def _extract_datumaro_bounding_boxes(
        self, input_image: np.ndarray, frame_masks: dict
    ) -> list[DatumaroBbox]:
        """
        Extract bounding boxes from frame masks.
        """
        # copy for deepsoert, because the input_image will be drawn and the embeddings will be picked from this one
        original_image = input_image.copy()
        bounding_boxes: list[DatumaroBbox] = []
        all_filtered_boxes = []
        for obj_id, sparse_object_mask in frame_masks.items():
            if is_empty_sparse_tensor(sparse_object_mask):
                self.count_empty_instance_masks += 1
                continue

            dense_object_mask = sparse_mask_tensor_to_dense_numpy(sparse_object_mask)

            # Get blobs and filter by its properties
            filtered_blobs = self._get_filtered_blobs(dense_object_mask)

            # Generate image crops based on blobs' data
            blob_patches = self._get_blob_patches(
                original_image, filtered_blobs, do_mask=True
            )

            if self.debug_viz:
                input_image = draw_mask_overlay(
                    input_image,
                    dense_object_mask,
                    class_id=obj_id,
                    color=(255, 0, 0),
                    alpha=0.5,
                    binary_mask=True,
                )

            filtered_boxes = self._get_filtered_bounding_boxes_with_trackid(
                filtered_blobs, blob_patches, obj_id
            )

            all_filtered_boxes.extend(filtered_boxes)
            self.tracker.update(obj_id, filtered_boxes)

            # Create a list of Datumaro Bounding Boxes
            datumaro_boxes = self.create_datumaro_boxes(filtered_boxes, obj_id)
            bounding_boxes.extend(datumaro_boxes)

        return bounding_boxes

    def _get_filtered_blobs(self, dense_object_mask: np.ndarray) -> list[BlobInfo]:
        """Get filtered blobs from dense mask using the configured rules."""
        valid_blobs: list[BlobInfo] = []
        for blob in iterate_blobs_from_mask(dense_object_mask):
            if self.verbose:
                x1, y1, x2, y2 = blob.bbox_xyxy
                print(
                    f"Found blob {blob.blob_num} area={blob.area} bbox=({x1},{y1},{x2},{y2})"
                )

            for rule in self.blob_rules:
                if not rule(blob):
                    if self.verbose:
                        print(f"  skipping blob {blob.blob_num}: {rule.explain(blob)}")
                    break
            else:
                valid_blobs.append(blob)

        return valid_blobs

    @staticmethod
    def _get_blob_patches(input_image: np.ndarray, blobs: list[BlobInfo], do_mask=True):
        """Return image patches, based on the blobs information"""
        if do_mask:
            return [
                blob.masked_crop_from_image(input_image, mask_background=False)
                for blob in blobs
            ]
        else:
            return [blob.crop_from_image(input_image) for blob in blobs]

    def _get_filtered_bounding_boxes(
        self, blobs, patches
    ) -> list[tuple[int, int, int, int]]:
        """Get classified bounding boxes from blobs."""

        if self.classifier is None:
            # TODO: fallback behavior
            raise ValueError("Currently, a classifier is required")

        boxes: list[tuple[int, int, int, int]] = []
        for blob, masked_patch in zip(blobs, patches):
            results = self.classifier(
                masked_patch, verbose=False, conf=self.classifier_conf
            )[0]
            classes = results.names
            prediction = classes[results.probs.top1]

            if prediction == self.target_class:
                boxes.append(blob.bbox_xyxy)
                if self.debug_viz:
                    print(f"Prediction: {prediction}")
                    cv2_imshow(masked_patch)

        return boxes

    def _get_filtered_bounding_boxes_with_trackid(
        self, blobs: list[BlobInfo], patches: list[np.ndarray], obj_id: int
    ) -> list[tuple[BlobInfo, float, str, int]]:
        """Get classified bounding boxes from blobs."""

        if self.classifier is None:
            # TODO: fallback behavior
            raise ValueError("Currently, a classifier is required")

        boxes: list[tuple[BlobInfo, float, str, int]] = []
        for blob, masked_patch in zip(blobs, patches):
            results = self.classifier(
                masked_patch, verbose=False, conf=self.classifier_conf
            )[0]
            classes = results.names
            prediction = classes[results.probs.top1]
            conf = results.probs.top1conf.item()

            print(f"Prediction: {prediction}")
            if prediction == self.target_class:
                print("Got fish! Adding to boxes")
                boxes.append((blob, conf, prediction, obj_id))
                if self.debug_viz:
                    cv2_imshow(masked_patch)

        return boxes

    def create_datumaro_boxes(self, bounding_boxes: list, obj_id) -> list[DatumaroBbox]:
        """Add labels to bounding boxes based on object ID and DataFrame."""

        label_id = get_label_id(
            self.chunked_df,
            self.col_class_name,
            self.col_instance_id,
            obj_id,
            self.label_categories,
        )

        output = []
        for blob, conf, prediction, obj_id in bounding_boxes:
            x, y, w, h = blob.bbox_xywh
            output.append(DatumaroBbox(x, y, w, h, label=label_id))
        return output

    def _print_statistics(self) -> None:
        """Print processing statistics."""
        print(f"Count of empty instance masks: {self.count_empty_instance_masks}")
        print(f"Count of frames with errors: {self.count_frames_with_errors}")


def iterate_blobs_from_mask(object_mask: np.ndarray):
    """ """

    num_blobs, labeled_mask, stats, _ = cv2.connectedComponentsWithStats(
        object_mask, connectivity=8, ltype=cv2.CV_32S
    )

    # Iterate labels skipping background (label 0)
    for blob_num in range(1, num_blobs):
        x = int(stats[blob_num, cv2.CC_STAT_LEFT])
        y = int(stats[blob_num, cv2.CC_STAT_TOP])
        w = int(stats[blob_num, cv2.CC_STAT_WIDTH])
        h = int(stats[blob_num, cv2.CC_STAT_HEIGHT])
        area = int(stats[blob_num, cv2.CC_STAT_AREA])

        yield BlobInfo(
            blob_num=blob_num, labeled_mask=labeled_mask, x=x, y=y, w=w, h=h, area=area
        )


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
    enter_frame = df[df[click_type_name] == ENTER_FRAME_MARKER][
        [instance_id, class_name, frame_name]
    ]
    enter_frame = enter_frame.sort_values(by=[instance_id, frame_name], ascending=True)

    # For each instance_id get frame where the object exits the scene
    exit_frame = df[df[click_type_name] == EXIT_FRAME_MARKER][
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


def load_masks(masks_filepath: Path):
    """ """
    print(f"Loading masks file: '{masks_filepath}'")
    if not masks_filepath.exists():
        raise FileNotFoundError("Masks file doesn't exist!")

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
