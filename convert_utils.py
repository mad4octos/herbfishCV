# Standard Library imports
import pickle
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, TypedDict
import logging
from logging import FileHandler
from datetime import datetime
from enum import IntEnum

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
from anomaly_rules import FishAnomalyRule, AnomalyDict
from blob_filter_rules import BlobRule


# Type aliases for the masks file
FrameIndex = int
ObjectIndex = int
MasksType = dict[FrameIndex, dict[ObjectIndex, torch.Tensor]]


class ResultsDict(TypedDict):
    cycles_since_update: int
    anomalies: list[AnomalyDict]


class ClickType(IntEnum):
    ENTER = 3
    EXIT = 4


@dataclass
class BlobMetrics:
    """Container for blob property measurements at a single point in time."""

    # TODO: I really need to replace this. It's just a duplicated version of BlobInfo without labeled_mask

    frame_idx: int
    obj_id: int
    area: int
    centroid_x: float
    centroid_y: float
    bbox_w: int
    bbox_h: int
    extent: float
    solidity: float
    compactness: float


@dataclass
class BlobInfo:
    frame_idx: int
    obj_id: int
    blob_num: int
    area: int
    labeled_mask: np.ndarray  # labeled mask where pixel == blob_num is this blob
    x: int
    y: int
    w: int
    h: int
    centroid_x: float
    centroid_y: float

    # Attributes computed later
    extent: Optional[float] = None
    solidity: Optional[float] = None
    compactness: Optional[float] = None
    predicted_class: Optional[str] = None

    @property
    def bbox_xywh(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)

    @property
    def bbox_xyxy(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.x + self.w, self.y + self.h)

    def crop_from_image(self, image: np.ndarray) -> np.ndarray:
        """Return the axis-aligned crop of the input image covering the blob bbox."""
        return image[self.y : self.y + self.h, self.x : self.x + self.w]

    def mask_and_crop_blob(self, image: np.ndarray, remove_background=True):
        """Mask and crop an image.

        This function can either remove the background (default) or highlight the foreground (the blob)."""

        image = np.copy(image)
        if remove_background:
            # Preserve only the foreground object, make everything else black
            image[self.labeled_mask != self.blob_num] = 0
        else:
            image = draw_mask_overlay(
                image, self.labeled_mask, self.blob_num, color=(0, 0, 255)
            )
        return self.crop_from_image(image)

    def compute_extent(self):
        """ """
        bbox_area = self.w * self.h
        self.extent = round(self.area / bbox_area if bbox_area > 0 else 0.0, 2)

    def compute_solidity(self):
        """ """

        mask = (self.labeled_mask == self.blob_num).astype(np.uint8) * 255

        # Find outer contours and build a single hull
        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            raise Exception("Problem finding Contours")

        hulls = []
        for i in range(len(contours)):
            hull = cv2.convexHull(contours[i], False)
            hull_area = cv2.contourArea(hull)
            hulls.append(hull)

        # For debugging
        # drawing = np.zeros((mask.shape[0], mask.shape[1], 3), np.uint8)
        # for i in range(len(contours)):
        #     color_contours = (0, 255, 0)  # green - color for contours
        #     color = (255, 0, 0)  # blue - color for convex hull
        #     cv2.drawContours(drawing, contours, i, color_contours, 1, 8, hierarchy)
        #     cv2.drawContours(drawing, hulls, i, color, 1, 8)

        self.solidity = round(self.area / hull_area, 2)

    def compute_compactness(self, approx_eps_frac: float = 0.003):
        """
        approx_eps_frac: contour smoothing to reduce pixel noise
        """
        m = (self.labeled_mask == self.blob_num).astype(np.uint8)

        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt = max(cnts, key=cv2.contourArea)
        A = cv2.contourArea(cnt)

        # Smooth the contour so perimeter isn't blown up by pixel jaggies
        P_raw = cv2.arcLength(cnt, True)
        eps = approx_eps_frac * P_raw
        cnt_smooth = cv2.approxPolyDP(cnt, eps, True)
        P = cv2.arcLength(cnt_smooth, True)

        normalized_compactness = float((P**2) / (4 * np.pi * A + 1e-9))
        self.compactness = round(normalized_compactness, 2)


class FishTracker:
    def __init__(
        self,
        obj_id,
        anomaly_rules: Iterable[FishAnomalyRule],
        logger,
        window_size=10,
    ):
        self.id = obj_id
        self.window_size = window_size

        # Number of cycles elapsed since the last update.
        # - This counter is incremented during each call to predict().
        # - It resets to 0 whenever update() is called.
        # - The tracker is removed after certain number of cycles without updates
        self.cycles_since_update = 0

        # Bounded deque: discards items from the opposite end when full.
        # Time series of blob properties
        self.metrics: deque[BlobMetrics] = deque(maxlen=window_size)

        self.anomaly_rules = anomaly_rules

        self.logger = logger

    def log_metrics(self):
        """Pretty-print metrics deque to the logger."""
        if not self.metrics:
            self.logger.info("No metrics available.")
            return

        lines = ["BlobMetrics list (most recent last):"]
        for m in self.metrics:
            lines.append(str(m))

        self.logger.info("\n".join(lines))

    def predict(self) -> ResultsDict:
        """Check for anomalies in the tracked fish properties."""
        self.cycles_since_update += 1

        results: ResultsDict = {
            "cycles_since_update": self.cycles_since_update,
            "anomalies": [],
        }

        if len(self.metrics) < 2:
            return results

        self.log_metrics()
        for anomaly_check in self.anomaly_rules:
            if (anomaly := anomaly_check(self)) is not None:
                # Remove the latest metric, because it was anomalous.
                del self.metrics[-1]

                results["anomalies"].append(anomaly)
                self.logger.info(anomaly_check.explain(anomaly))

            elif anomaly is None:
                # No anomaly found... but that doesn't guarantee there are no anomalies in this blob.
                # It may just be that there was not enough data to check for anomalies!
                # E.g., for the spike anomaly detector (prev vs current frame),
                # it may be that the previous and current frames are not continuous.
                # That may be due to the classifier rejecting one frame, thus creating a discontinuity between
                # frames. i.e. this anomaly check only acts on continuous non-classifier-rejected frames.
                # The frame could still potentially be anomalous!
                pass

        return results

    def update(self, blob: BlobInfo):
        """ """
        self.cycles_since_update = 0
        metrics = self._extract_metrics(blob)
        self.metrics.append(metrics)

    def get_summary(self) -> dict:
        """Get current summary of tracked properties."""
        # TODO: Summary of other properties
        areas = [m.area for m in self.metrics]
        return {
            "fish_id": self.id,
            "current_area": areas[-1],
            "area_mean": np.mean(areas),
            "area_variance": np.var(areas) if len(areas) > 1 else 0.0,
            "num_frames_tracked": len(areas),
        }

    def __str__(self) -> str:
        summary = self.get_summary()
        return (
            f"FishTracker(id={self.id}, cycles_since_update={self.cycles_since_update}, "
            f"window_size={self.window_size}, frames_tracked={summary['num_frames_tracked']})"
        )

    def __repr__(self) -> str:
        return self.__str__()

    @staticmethod
    def _extract_metrics(blob: BlobInfo) -> BlobMetrics:
        """Extract metrics from a BlobInfo object."""

        # TODO: I want just the same as blobinfo but without the whole mask
        return BlobMetrics(
            frame_idx=blob.frame_idx,
            obj_id=blob.obj_id,
            area=blob.area,
            centroid_x=blob.centroid_x,
            centroid_y=blob.centroid_y,
            bbox_w=blob.w,
            bbox_h=blob.h,
            extent=blob.extent,
            solidity=blob.solidity,
            compactness=blob.compactness,
        )


class FishTrackerManager:
    """
    Main tracker that manages multiple FishTracker instances.

    Responsibilities:
    - Create and update individual fish trackers
    - Run predictions on all trackers
    - Remove inactive trackers after a threshold of cycles without updates
    """

    def __init__(
        self,
        anomaly_rules: Iterable[FishAnomalyRule],
        logger,
        max_cycles_without_update=3,
        window_size=10,
    ):
        """ """
        self.trackers: dict[int, FishTracker] = {}
        self.max_cycles_without_update = max_cycles_without_update
        self.window_size = window_size
        self.anomaly_rules = anomaly_rules
        self.logger = logger

    def update(self, blob: BlobInfo):
        """Update a fish's status or create a new fish tracker if not exists."""

        if blob.obj_id not in self.trackers:
            self.trackers[blob.obj_id] = FishTracker(
                blob.obj_id,
                window_size=self.window_size,
                anomaly_rules=self.anomaly_rules,
                logger=self.logger,
            )

        self.trackers[blob.obj_id].update(blob)

    def predict(self, obj_id) -> ResultsDict:
        """ """
        return self.trackers[obj_id].predict()

    def filter_dead_trackers(self):
        """ """
        removed_ids = []
        alive_trackers = {}

        for obj_id, tracker in self.trackers.items():
            if tracker.cycles_since_update < self.max_cycles_without_update:
                alive_trackers[obj_id] = tracker
            else:
                removed_ids.append(obj_id)
                print(
                    f"Removing inactive tracker: fish_id={obj_id}, "
                    f"cycles_without_update={tracker.cycles_since_update}"
                )

        self.trackers = alive_trackers
        return removed_ids

    def get_summary(self) -> dict:
        """
        Get a summary of all active trackers.

        Returns:
            Dictionary with tracker statistics
        """
        return {
            "num_active_trackers": len(self.trackers),
            "fish_ids": list(self.trackers.keys()),
            "max_cycles_without_update": self.max_cycles_without_update,
        }

    def __str__(self) -> str:
        summary = self.get_summary()
        return (
            f"FishTrackerManager(active={summary['num_active_trackers']}, "
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
        classifier,
        blob_rules: Iterable[BlobRule],
        anomaly_rules: Iterable[FishAnomalyRule],
        classifier_conf: float = 0.5,
        col_class_name: str = "ObjType",
        col_instance_id: str = "ObjID",
        filename_num_zeros: int = 8,
        target_class=None,
        start_frame=0,
        max_frames=None,
        verbose: bool = False,
        notebook_debug=False,
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
        self.verbose = verbose
        self.classifier = classifier
        self.classifier_conf = classifier_conf
        self.video_writer = cv2.VideoWriter(
            filename="tracker.mp4",
            fourcc=cv2.VideoWriter_fourcc(*"mp4v"),  # *"MPEG", "MJPG", "mp4v", "FMP4"
            fps=2,
            frameSize=(3840, 2160),
            isColor=True,
        )
        self.dataset_items: list[datumaro.components.dataset_base.DatasetItem] = []
        self.count_empty_instance_masks = 0
        self.count_frames_with_errors = 0
        self.notebook_debug = notebook_debug
        self.target_class = target_class
        self.max_frames = max_frames
        self.start_frame = start_frame
        self.blob_rules = blob_rules
        self.anomaly_rules = anomaly_rules
        self.setup_logging(log_to_console=verbose)
        self.tracker_manager = FishTrackerManager(
            self.anomaly_rules, logger=self.logger
        )

    def setup_logging(self, log_to_console=True, level: int = logging.INFO):
        """
        Configure logging with both console and file handlers.

        Parameters
        ----------
        log_file : str
            Path to the log file. Defaults to "dataset_builder.log"
        level : int
            Logging level. Defaults to logging.INFO
        """

        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(level)

        # Remove existing handlers to allow reconfiguration in notebooks
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # Create formatter
        formatter = logging.Formatter(
            "%(levelname)s: %(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

        if log_to_console:
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # TODO: add video name
        log_file = f"builder_{timestamp}.log"

        # File handler
        file_handler = FileHandler(filename=log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        self.logger = logger

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
                print("Breaking because max number of frames has been reached.")
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
        self.logger.info(f"Processing frame {extracted_frame_idx}...")

        if extracted_frame_idx in self.all_error_frames:
            print(f"Frame {extracted_frame_idx} has errors in the CSV. Skipping.")
            self.count_frames_with_errors += 1
            return

        filename = _get_frame_filename(extracted_frame_idx, self.filename_num_zeros)
        image_filepath = self.images_path / filename
        input_image = cv2.imread(str(image_filepath), cv2.IMREAD_COLOR)
        if input_image is None:
            raise FileNotFoundError("Frame file doesn't exist!")

        # Write the frame index on the top left corner of the frame
        input_image = cv2.putText(
            input_image,
            f"Frame {extracted_frame_idx}",
            (30, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
        )

        # FIXME: too much functionality inside here
        blobs = self._get_blobs(input_image, frame_masks, extracted_frame_idx)

        bounding_boxes = self.create_datumaro_boxes(blobs)

        self.tracker_manager.filter_dead_trackers()

        self.dataset_items.append(
            datumaro.components.dataset_base.DatasetItem(
                id=filename.split(".")[0],
                subset="train",
                media=datumaro.components.media.Image.from_file(str(image_filepath)),
                annotations=bounding_boxes,
                attributes={"frame": extracted_frame_idx},
            )
        )

        if self.notebook_debug:
            cv2_imshow(input_image)

        if self.video_writer is not None:
            self.video_writer.write(input_image)

    def _get_blobs(
        self, input_image: np.ndarray, frame_masks: dict, extracted_frame_idx: int
    ) -> list[BlobInfo]:
        """
        Extract bounding boxes from frame masks.
        """
        original_image = input_image.copy()

        all_blobs = []
        for obj_id, sparse_object_mask in frame_masks.items():
            if is_empty_sparse_tensor(sparse_object_mask):
                self.count_empty_instance_masks += 1
                continue

            # Binary masks
            dense_object_mask = sparse_mask_tensor_to_dense_numpy(sparse_object_mask)

            # Filter blob by basic featurs like area and size, to remove small blobs
            # TODO: maybe return blobs first and then filter
            filtered_blobs = self._get_filtered_blobs(
                dense_object_mask, obj_id, extracted_frame_idx
            )

            # Generate image crops based on blobs data
            blob_patches = self._get_blob_patches(
                original_image, filtered_blobs, do_mask=True
            )

            # Filter blobs with a classifier, only correctly masked fish will be preserved
            classified_blobs = self._classify_blobs(filtered_blobs, blob_patches)

            if classified_blobs:
                # Preserve the largest blob
                dominant_blob = max(classified_blobs, key=lambda b: b.area)

                # Compute other properties
                dominant_blob.compute_solidity()
                dominant_blob.compute_extent()
                dominant_blob.compute_compactness()

                self.tracker_manager.update(dominant_blob)
                results = self.tracker_manager.predict(obj_id)

                # NOTE:
                # - a white label with an ID indicates this blob has not been rejected
                # - a red label with a red rectangle indicates the blob has been rejected by the anomaly detector
                # - no label indicates the blob was not processed by the anomaly detector and will be included in the
                #   output dataset

                if results["anomalies"]:
                    ####################################################################################################
                    # Draw a red rectangle and information regarding why a mask was rejected
                    ####################################################################################################
                    anomalies = ",".join(
                        [f"{a['type']}({a['value']})" for a in results["anomalies"]]
                    )

                    input_image = cv2.putText(
                        input_image,
                        f"ID: {dominant_blob.obj_id} ({anomalies})",
                        (dominant_blob.x, dominant_blob.y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 0, 255),
                        2,
                    )

                    x, y, w, h = map(int, dominant_blob.bbox_xywh)
                    cv2.rectangle(
                        img=input_image,
                        pt1=(x, y),
                        pt2=(x + w, y + h),
                        color=(0, 0, 255),
                        thickness=5,
                    )

                else:
                    ####################################################################################################
                    # Draw a white label with the Object ID, indicating that this blob has not been rejected
                    ####################################################################################################

                    input_image = cv2.putText(
                        input_image,
                        f"ID: {dominant_blob.obj_id}",
                        (dominant_blob.x, dominant_blob.y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (255, 255, 255),
                        2,
                    )
                    all_blobs.append(dominant_blob)

            input_image = draw_mask_overlay(
                input_image,
                dense_object_mask,
                class_id=obj_id,
                color=None,
                alpha=0.5,
                binary_mask=True,
            )

        return all_blobs

    def _get_filtered_blobs(
        self, dense_object_mask: np.ndarray, obj_id: int, extracted_frame_idx: int
    ) -> list[BlobInfo]:
        """Get filtered blobs from dense mask using the configured rules."""
        valid_blobs = []
        for blob in get_blobs_from_mask(dense_object_mask, obj_id, extracted_frame_idx):
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
                blob.mask_and_crop_blob(input_image, remove_background=False)
                for blob in blobs
            ]
        else:
            return [blob.crop_from_image(input_image) for blob in blobs]

    def _classify_blobs(
        self, blobs: list[BlobInfo], patches: list[np.ndarray]
    ) -> list[BlobInfo]:
        """Get classified bounding boxes from blobs."""

        filtered_blobs = []
        for blob, masked_patch in zip(blobs, patches):
            results = self.classifier(
                masked_patch, verbose=False, conf=self.classifier_conf
            )[0]
            classes = results.names
            prediction = classes[results.probs.top1]
            # conf = results.probs.top1conf.item()

            if prediction == self.target_class:
                blob.predicted_class = prediction
                filtered_blobs.append(blob)
                if self.notebook_debug:
                    cv2_imshow(masked_patch)

        return filtered_blobs

    def create_datumaro_boxes(self, blobs: list[BlobInfo]) -> list[DatumaroBbox]:
        """Add labels to bounding boxes based on object ID and DataFrame."""

        output = []
        for blob in blobs:
            label_id = get_label_id(
                self.chunked_df,
                self.col_class_name,
                self.col_instance_id,
                blob.obj_id,
                self.label_categories,
            )
            x, y, w, h = blob.bbox_xywh
            output.append(DatumaroBbox(x, y, w, h, label=label_id))
        return output

    def _print_statistics(self) -> None:
        """Print processing statistics."""
        print(f"Count of empty instance masks: {self.count_empty_instance_masks}")
        print(f"Count of frames with errors: {self.count_frames_with_errors}")


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

        blobs.append(
            BlobInfo(
                frame_idx=extracted_frame_idx,
                blob_num=blob_num,
                obj_id=obj_id,
                labeled_mask=labeled_mask,
                x=x,
                y=y,
                w=w,
                h=h,
                area=area,
                centroid_x=float(round(cx, 1)),
                centroid_y=float(round(cy, 1)),
            )
        )

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
        raise FileNotFoundError(f"Masks file '{masks_filepath}' doesn't exist!")

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


def load_errors_df(filepath: Path, video_id: str):
    """ """
    errors_df = pd.read_csv(filepath)
    video_errors_df = errors_df[errors_df.obsID == video_id]
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
