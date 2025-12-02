# Standard Library imports
import logging
from datetime import datetime
from logging import FileHandler
from pathlib import Path
from typing import Iterable, Literal

# External imports
import cv2
import datumaro.components.dataset
import datumaro.components.dataset_base
import datumaro.components.media
from datumaro.components.annotation import Bbox as DatumaroBbox
import numpy as np
import pandas as pd
from tqdm import tqdm

# Local imports
from anomaly_rules import FishAnomalyRule
from blob_filter_rules import BlobRule
from common import cv2_imshow, is_empty_sparse_tensor, sparse_mask_tensor_to_dense_numpy
from convert_utils import _get_frame_filename, get_blobs_from_mask, get_label_id
from plot_utils import draw_mask_overlay
from tracker import FishTrackerManager
from blob import BlobInfo


class DatumaroDatasetBuilder:
    """
    Build a Datumaro dataset from instance segmentation masks and associated metadata.

    This class processes frame-by-frame instance masks, converts them to bounding boxes,
    and creates a Datumaro dataset suitable for object detection tasks. Frames with errors
    are skipped, and empty masks are tracked for diagnostic purposes.
    """

    def __init__(
        self,
        obs_id: str,  # this is basically the name of the run
        masks: dict,
        error_frames: list[int],
        chunked_df: pd.DataFrame,
        label_categories: datumaro.components.dataset_base.CategoriesInfo,
        images_path: Path,
        export_root_path: Path,
        classifier,
        blob_rules: Iterable[BlobRule],
        window_size,
        anomaly_rules: Iterable[FishAnomalyRule],
        target_class: list[str],
        classifier_conf: float = 0.5,
        col_class_name: str = "ObjType",
        col_instance_id: str = "ObjID",
        filename_num_zeros: int = 8,
        start_frame=0,
        max_frames=None,
        verbose: bool = False,
        notebook_debug=False,
        video_fps: int = 2,
        video_height: int = 2160,
        video_width: int = 3840,
    ):
        """ """
        self.start_time = datetime.now()
        self.obs_id = obs_id
        self.masks = masks
        self.error_frames = error_frames
        self.chunked_df = chunked_df
        self.label_categories = label_categories
        self.images_path = Path(images_path)
        self.export_root_path = export_root_path
        self.col_class_name = col_class_name
        self.col_instance_id = col_instance_id
        self.filename_num_zeros = filename_num_zeros
        self.verbose = verbose
        self.classifier = classifier
        self.classifier_conf = classifier_conf
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
            self.anomaly_rules, logger=self.logger, window_size=window_size
        )
        self.create_video_writer(fps=video_fps, height=video_height, width=video_width)

    def create_video_writer(self, fps: int, height: int, width: int):
        """ """
        self.export_root_path.mkdir(parents=True, exist_ok=True)

        filepath = (
            self.export_root_path
            / f"{self.obs_id}_debug-exported-on-{self.start_time.strftime('%Y%m%d_%H%M%S')}.mp4"
        )
        try:
            self.video_writer = cv2.VideoWriter(
                filename=str(filepath),
                # *"MPEG", "MJPG", "mp4v", "FMP4"
                fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
                fps=fps,
                frameSize=(width, height),
                isColor=True,
            )
            self.logger.info(f"Video writer initialized. Output file: '{filepath}'")
        except Exception as e:
            self.logger.exception("Problem during video writer initialization")

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
        # Create export dir if doesn't exist
        self.export_root_path.mkdir(parents=True, exist_ok=True)

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

        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        log_file = self.export_root_path / f"{self.obs_id}-exported-on-{timestamp}.log"

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
        if not self.error_frames:
            self.logger.warning(
                "No CSV frame errors will be used because none were found!"
            )
        else:
            self.logger.info(f"Frames with errors (from CSV): {self.error_frames}")

        for extracted_frame_idx, frame_masks in tqdm(self.masks.items()):
            if extracted_frame_idx < self.start_frame:
                continue

            if (self.max_frames is not None) and (
                extracted_frame_idx >= (self.start_frame + self.max_frames)
            ):
                self.logger.info(
                    f"Breaking because max number of frames has been reached."
                    f"Started at frame {self.start_frame}, stopped after {self.max_frames} frames."
                )
                break

            try:
                self._process_frame(extracted_frame_idx, frame_masks)
            except FileNotFoundError:
                self.video_writer.release()
                self.logger.exception("Stopping due to exception.")
                raise

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

        if extracted_frame_idx in self.error_frames:
            self.logger.warning(
                f"Frame {extracted_frame_idx} has associated errors in the CSV. Skipping."
            )
            self.count_frames_with_errors += 1
            return

        filename = _get_frame_filename(extracted_frame_idx, self.filename_num_zeros)
        image_filepath = self.images_path / filename
        input_image = cv2.imread(str(image_filepath), cv2.IMREAD_COLOR)
        if input_image is None:
            error_message = f"File '{image_filepath}' doesn't exist!"
            raise FileNotFoundError(error_message)

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

        if self.notebook_debug or (self.video_writer is not None):
            # Write frame index on the top left corner of the frame
            input_image = cv2.putText(
                input_image,
                f"Frame {extracted_frame_idx}",
                (30, 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.9,
                color=(255, 255, 255),
                thickness=2,
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
                # - a white box and label with an ID indicate this blob has not been rejected
                # - a red label with a red rectangle indicates the blob has been rejected by the anomaly detector
                # - no label indicates the blob was not processed by the anomaly detector and will be included in the
                #   output dataset

                if results["anomalies"]:
                    # Draw a red rectangle and information regarding why a mask was rejected
                    anomalies = ",".join(
                        [f"{a['type']}({a['value']})" for a in results["anomalies"]]
                    )
                    self.draw_bbox_and_id(
                        input_image, dominant_blob, "red", extra_text=f"({anomalies}"
                    )

                else:
                    # Draw a green box and label with the Object ID indicating that this blob has not been rejected
                    self.draw_bbox_and_id(input_image, dominant_blob, "white")
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

    def draw_bbox_and_id(
        self,
        image: np.ndarray,
        blob: BlobInfo,
        color_name: Literal["red", "white"],
        extra_text: str = "",
    ):
        """Draw a bounding box and the blob ID."""
        match color_name:
            case "white":
                color = (255, 255, 255)
            case "red":
                color = (0, 0, 255)
            case _:
                raise ValueError(f"Unknown color name: {color_name}")

        x, y, w, h = map(int, blob.bbox_xywh)
        cv2.putText(
            image,
            f"ID: {blob.obj_id} {extra_text}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2,
        )
        cv2.rectangle(
            img=image, pt1=(x, y), pt2=(x + w, y + h), color=color, thickness=4
        )

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

            if prediction in self.target_class:
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
            output.append(
                DatumaroBbox(
                    x, y, w, h, label=label_id, attributes={"ObjID": blob.obj_id}
                )
            )
        return output

    def _print_statistics(self) -> None:
        """Print processing statistics."""
        print(f"Count of empty instance masks: {self.count_empty_instance_masks}")
        print(f"Count of frames with errors: {self.count_frames_with_errors}")
