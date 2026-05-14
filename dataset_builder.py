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
import datumaro.util.mask_tools as mask_tools
import numpy as np
import pandas as pd
from datumaro.components.annotation import Annotation, RleMask
from tqdm import tqdm
from ultralytics import YOLO

# Local imports
from anomaly_rules import FishAnomalyRule
from blob import BlobInfo
from blob_filter_rules import BlobRule
from common import cv2_imshow, is_empty_sparse_tensor, sparse_mask_tensor_to_dense_numpy
from convert_utils import (
    MasksType,
    _get_frame_filename,
    get_blobs_from_mask,
    get_label_id,
)
from plot_utils import draw_mask_overlay
from tracker import FishTrackerManager


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
        masks: MasksType,
        error_frames: list[int],
        chunked_df: pd.DataFrame,
        annotations_df: pd.DataFrame,
        label_categories: datumaro.components.dataset_base.CategoriesInfo,
        export_root_path: Path,
        images_path: Path,
        col_class_name: str = "ObjType",
        col_instance_id: str = "ObjID",
        filename_num_zeros: int = 5,
        start_frame: int = 0,
        max_frames: int | None = None,
        extracted_fps: int | None = None,
        final_fps: int | None = None,
        original_fps: float | None = None,
        sam2_start: int | None = None,
        no_auto: bool = False,
        classifier: YOLO | None = None,
        bg_mode: Literal["gray", "overlay"] | None = None,
        correct_class: str | None = None,
        incorrect_class: str | None = None,
        incorrect_cls_conf_thresh: float | None = None,
        blob_rules: Iterable[BlobRule] | None = None,
        anomaly_rules: Iterable[FishAnomalyRule] | None = None,
        window_size: int | None = None,
        create_video: bool = False,
        video_fps: int | None = None,
        video_height: int | None = None,
        video_width: int | None = None,
        subset: str = "train",
        verbose: bool = False,
        notebook_debug: bool = False,
    ):
        """
        Parameters
        ----------
        obs_id : str
            Observation identifier.
        masks : MasksType
            Nested dictionary mapping frame indices to per-object sparse tensor
            SAM2 masks. Structure: `{frame_idx: {obj_id: sparse_tensor, ...}, ...}`.
        error_frames : list[int]
            Frame indices (0-indexed) that had errors in the CSV file.
            These frames are still exported as dataset items but with empty
            mask annotations instead of real ones. Ground-truth click
            location attributes are preserved only when blobs are detected
            for the frame.
        chunked_df : pd.DataFrame
            DataFrame containing per-object metadata for the current observation
            chunk. Must include columns for class name and instance ID (see
            *col_class_name* and *col_instance_id*). Used to look up the label
            for each detected object.
        annotations_df : pd.DataFrame
            DataFrame with ground-truth point annotations (click locations).
            Expected columns: `ObjID`, `ClickType`, `Frame`, and
            `Location`. Only rows where `ClickType == 1` are considered
            when looking up the nearest ground-truth position for each
            exported mask.
        label_categories : datumaro.components.dataset_base.CategoriesInfo
            Datumaro label category mapping that defines the set of classes
            for the output dataset (e.g. fish species).
        export_root_path : Path
            Root directory where all outputs are written: the debug video and
            log files. The Datumaro dataset is returned by `build` and
            must be saved by the caller.
        images_path : Path
            Directory containing the extracted video frames as image files
            (e.g. JPEGs). Frame filenames are expected to be zero-padded
            integers (see *filename_num_zeros*).
        col_class_name : str, optional
            Column name in *chunked_df* that holds the object class/type
            string. Default is `ObjType`.
        col_instance_id : str, optional
            Column name in *chunked_df* that holds the object instance ID.
            Default is `ObjID`.
        filename_num_zeros : int, optional
            Number of zero-padded digits in the extracted frame filenames.
            For example, `5` expects filenames like `00042.jpg`.
            Default is `5`.
        start_frame : int, optional
            First frame index to process (inclusive). All frames before this
            index are skipped. Default is `0`.
        max_frames : int or None, optional
            Maximum number of frames to process starting from *start_frame*.
            `None` means process all available frames. Default is `None`.
        extracted_fps : int or None, optional
            Frame rate at which frames were extracted from the original video.
            Used together with *final_fps* to compute the subsampling step,
            and with *original_fps* / *sam2_start* for frame-number mapping.
            Default is `None` (no subsampling or mapping).
        final_fps : int or None, optional
            Desired output frame rate. When both *extracted_fps* and
            *final_fps* are provided, only every `extracted_fps // final_fps`
            frame is processed. Default is `None`.
        original_fps : float or None, optional
            Frame rate of the original source video. Used to map extracted
            frame indices back to the original video frame space (needed for
            ground-truth annotation lookup). Default is `None`.
        sam2_start : int or None, optional
            Starting frame offset in the original video where SAM2 mask
            propagation began. Used together with *original_fps* and
            *extracted_fps* for the frame-number mapping formula:
            `original_frame = extracted_frame * (original_fps / extracted_fps) + sam2_start`.
            Default is `None`.
        no_auto : bool, optional
            If True, skip all automatic mask cleaning (blob filtering,
            YOLO classification, and anomaly detection). Only the largest
            blob per object is kept. Default is `False`.
        classifier : YOLO or None, optional
            A YOLO classification model instance (e.g. loaded via
            `ultralytics.YOLO`). Called on cropped/masked blob patches to
            verify that a detected blob belongs to the *correct_class*.
            When provided and *no_auto* is False, *correct_class*,
            *incorrect_class*, *incorrect_cls_conf_thresh*, and *bg_mode*
            must also be set. Ignored when *no_auto* is True.
            Default is `None`.
        bg_mode : {"gray", "overlay"} or None, optional
            Background style used when generating blob image patches for the
            classifier. `gray` fills the background with a neutral grey;
            `overlay` keeps the original image behind the masked blob.
            Required when *classifier* is not `None`. Default is `None`.
        correct_class : str or None, optional
            Class name the classifier must predict for a blob to be accepted
            (e.g. `fish`). Blobs predicted as *incorrect_class* are
            discarded. Required when *classifier* is not `None`.
            Default is `None`.
        incorrect_class : str or None, optional
            Class name used to label blobs that the classifier rejects.
            Required when *classifier* is not `None`. Default is `None`.
        incorrect_cls_conf_thresh : float or None, optional
            Confidence threshold for the *incorrect_class* prediction. When
            the classifier's confidence for the incorrect class meets or
            exceeds this value, the blob is discarded. Required when
            *classifier* is not `None`. Default is `None`.
        blob_rules : Iterable[BlobRule] or None, optional
            Sequence of blob filtering rules applied **before** classification.
            Each rule is a callable that receives a `BlobInfo` and returns
            `True` to keep or `False` to discard the blob based on
            geometric properties (area, size, shape, etc.).
            Required when *no_auto* is False. Default is `None`.
        anomaly_rules : Iterable[FishAnomalyRule] or None, optional
            Sequence of anomaly detection rules applied **after** tracking.
            These evaluate temporal behavior (e.g. sudden changes in area or
            shape over consecutive frames) to flag suspicious blobs.
            Required when *no_auto* is False. Default is `None`.
        window_size : int or None, optional
            Sliding window size passed to `FishTrackerManager`. Controls how
            many recent observations the anomaly detector considers when
            evaluating temporal metrics. Required when *no_auto* is False.
            Default is `None`.
        create_video : bool, optional
            If True, initialize a video writer and write a debug MP4 to
            *export_root_path* showing bounding boxes and mask overlays for
            every non-error processed frame. When True, *video_fps*, *video_height*,
            and *video_width* must be specified. Default is `False`.
        video_fps : int or None, optional
            Frame rate for the debug output video. Required when
            *create_video* is True. Default is `None`.
        video_height : int or None, optional
            Height in pixels of the debug output video. Required when
            *create_video* is True. Default is `None`.
        video_width : int or None, optional
            Width in pixels of the debug output video. Required when
            *create_video* is True. Default is `None`.
        subset : str, optional
            Datumaro dataset split name assigned to every exported item
            (e.g. `train`, `val`). Default is `train`.
        verbose : bool, optional
            If True, enable console logging and print per-blob filter
            explanations. Default is `False`.
        notebook_debug : bool, optional
            If True, display images inline using `cv2_imshow` for debugging
            inside Jupyter notebooks. Default is `False`.
        """
        self.start_time = datetime.now()
        self.obs_id = obs_id
        self.masks = masks
        self.error_frames = error_frames

        # Frame subsampling: if both FPS values are provided, compute the step
        if (extracted_fps is not None) and (final_fps is not None):
            self.frame_step = extracted_fps // final_fps
        else:
            self.frame_step = None

        # Original-video frame mapping: used to convert extracted frame numbers
        # back to the original video frame space (as seen in .npy annotations).
        # Formula: original_frame_number = extracted_frame_number * (original_fps / extracted_fps) + sam2_start
        self.original_fps = original_fps
        self.extracted_fps = extracted_fps
        self.sam2_start = sam2_start

        self.chunked_df = chunked_df
        self.annotations_df = annotations_df
        self.label_categories = label_categories
        self.images_path = Path(images_path)
        self.export_root_path = export_root_path
        self.col_class_name = col_class_name
        self.col_instance_id = col_instance_id
        self.filename_num_zeros = filename_num_zeros
        self.verbose = verbose
        self.classifier = classifier
        self.incorrect_cls_conf_thresh = incorrect_cls_conf_thresh
        self.dataset_items: list[datumaro.components.dataset_base.DatasetItem] = []
        self.count_empty_instance_masks = 0
        self.count_frames_with_errors = 0
        self.notebook_debug = notebook_debug
        self.no_auto = no_auto
        self.subset = subset
        self.correct_class = correct_class
        self.incorrect_class = incorrect_class
        self.max_frames = max_frames
        self.start_frame = start_frame
        self.blob_rules = blob_rules
        self.setup_logging(log_to_console=verbose)
        if self.no_auto:
            self.tracker_manager = None
        else:
            if blob_rules is None:
                raise ValueError("blob_rules is required when no_auto is False")
            if anomaly_rules is None:
                raise ValueError("anomaly_rules is required when no_auto is False")
            if window_size is None:
                raise ValueError("window_size is required when no_auto is False")
            self.tracker_manager = FishTrackerManager(
                anomaly_rules, logger=self.logger, window_size=window_size
            )
            if self.classifier is not None:
                if correct_class is None:
                    raise ValueError(
                        "correct_class is required when classifier is not None"
                    )
                if incorrect_class is None:
                    raise ValueError(
                        "incorrect_class is required when classifier is not None"
                    )
                if incorrect_cls_conf_thresh is None:
                    raise ValueError(
                        "incorrect_cls_conf_thresh is required when classifier is not None"
                    )
                if bg_mode is None:
                    raise ValueError(
                        "`bg_mode` must be specified in the configuration when the classifier is used"
                    )
                self.class_to_index = {
                    cls: idx for idx, cls in self.classifier.names.items()
                }
        self.bg_mode = bg_mode
        self.video_height = video_height
        self.video_width = video_width
        self.video_writer = None
        if create_video:
            assert video_height is not None, (
                "video_height must be specified when `create_video=True`"
            )
            assert video_width is not None, (
                "video_width must be specified when `create_video=True`"
            )
            assert video_fps is not None, (
                "video_fps must be specified when `create_video=True`"
            )
            self.create_video_writer(
                fps=video_fps, height=video_height, width=video_width
            )

    def extracted_to_original_frame(self, extracted_frame_idx: int) -> int | None:
        """
        Map an extracted frame index back to the original video frame number.

        The .npy annotation file stores frame numbers in the original video's
        FPS space.  This method inverts the extraction formula:

            extracted_frame_number = (original_frame_number - sam2_start) / (original_fps / extracted_fps)

        to recover:

            original_frame_number = extracted_frame_number * (original_fps / extracted_fps) + sam2_start

        Returns None when the mapping parameters (--original-fps, --sam2-start)
        were not provided.
        """
        if (
            self.original_fps is None
            or self.sam2_start is None
            or self.extracted_fps is None
        ):
            return None
        return int(
            extracted_frame_idx * (self.original_fps / self.extracted_fps)
            + self.sam2_start
        )

    def original_to_extracted_frame(self, original_frame_idx: int) -> int | None:
        """
        Map an original video frame number to the extracted frame index.

        Applies the extraction formula:

            extracted_frame_number = (original_frame_number - sam2_start) / (original_fps / extracted_fps)

        Returns None when the mapping parameters (--original-fps, --sam2-start)
        were not provided.
        """
        if (
            self.original_fps is None
            or self.sam2_start is None
            or self.extracted_fps is None
        ):
            return None
        return int(
            (original_frame_idx - self.sam2_start)
            / (self.original_fps / self.extracted_fps)
        )

    def get_closest_gt_location(
        self, extracted_frame_idx: int, obj_id: int
    ) -> tuple[list[float], str, int, int] | None:
        """
        Look up the closest ground-truth location for a given object at an extracted frame.

        Convert the extracted frame index (0-indexed) to the original video frame space,
        then finds the annotation row for this ObjID whose Frame is closest.

        Return the [x, y] location list, or None when the mapping parameters
        were not provided or no annotation exists for the given ObjID.
        """
        original_frame = self.extracted_to_original_frame(extracted_frame_idx)
        if original_frame is None:
            self.logger.warning(
                f"Couldn't map extracted frame index {extracted_frame_idx} to original frame. "
                f"No ground truth location will be available."
                f"Probable cause: missing parameters original_fps, sam2_start, or extracted_fps."
            )
            return None

        obj_rows = self.annotations_df[
            (self.annotations_df["ObjID"] == str(obj_id))
            & (self.annotations_df["ClickType"] == 1)
        ]
        if obj_rows.empty:
            return None

        # Find the row with the closest Frame value to the computed original frame.
        # (obj_rows["Frame"] - original_frame) gives the signed difference for each row,
        # .abs() makes them all positive distances, and .idxmin() returns the DataFrame
        # index of the row with the smallest distance.
        closest_idx = (obj_rows["Frame"] - original_frame).abs().idxmin()
        gt_location: list[float] = obj_rows.loc[closest_idx, "Location"]

        gt_obj_id: str = obj_rows.loc[closest_idx, "ObjID"]

        # Original frame number space
        gt_frame_original: int = obj_rows.loc[closest_idx, "Frame"]

        # Extracted frame number
        gt_frame_extracted = self.original_to_extracted_frame(gt_frame_original)

        return gt_location, gt_obj_id, gt_frame_extracted, gt_frame_original

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
        except Exception:
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
        if self.no_auto:
            self.logger.info(
                "--no-auto: skipping automatic mask cleaning (blob filters, classifier, anomaly detection)."
            )

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
            except (FileNotFoundError, IOError):
                if self.video_writer is not None:
                    self.video_writer.release()
                self.logger.exception(
                    f"Stopping: failed to load image for frame {extracted_frame_idx}."
                )
                raise

        dataset = datumaro.components.dataset.Dataset.from_iterable(
            self.dataset_items, categories=self.label_categories
        )

        if self.video_writer is not None:
            self.video_writer.release()
            print("Finished writing video")

        self._print_statistics()
        return dataset

    def _load_frame_image(self, extracted_frame_idx: int) -> tuple:
        """Load a frame image by index"""
        filename = _get_frame_filename(extracted_frame_idx, self.filename_num_zeros)
        image_filepath = self.images_path / filename
        if not image_filepath.exists():
            # look for any file that ends with the expected filename
            matches = list(self.images_path.glob(f"*{filename}"))
            if not matches:
                raise FileNotFoundError(
                    f"File '{image_filepath}' doesn't exist, and no prefixed"
                    f" variants matching '*{filename}' were found!"
                )
            image_filepath = matches[0]  # take the first match
        input_image = cv2.imread(str(image_filepath), cv2.IMREAD_COLOR)
        if input_image is None:
            raise IOError(
                f"File '{image_filepath}' exists but cannot be read as an image."
            )
        return filename, image_filepath, input_image

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

        # Subsample frames based on extracted_fps / final_fps ratio
        if (self.frame_step is not None) and (
            (extracted_frame_idx % self.frame_step) != 0
        ):
            self.logger.info(
                f"Skipping frame {extracted_frame_idx} (keeping every {self.frame_step}th frame)"
            )
            return

        self.logger.info(f"Processing frame {extracted_frame_idx}...")
        filename, image_filepath, input_image = self._load_frame_image(
            extracted_frame_idx
        )

        blobs = self._get_blobs(input_image, frame_masks, extracted_frame_idx)
        media = datumaro.components.media.Image.from_file(str(image_filepath))

        # For error frames, keep the click location by saving an annotation with an empty mask
        # rather than skipping the frame entirely.
        is_error = extracted_frame_idx in self.error_frames
        annotations = (
            self.create_empty_datumaro_annotations(blobs)
            if is_error
            else self.create_datumaro_annotations(blobs)
        )
        self.dataset_items.append(
            datumaro.components.dataset_base.DatasetItem(
                id=filename.split(".")[0],
                subset=self.subset,
                media=media,
                annotations=annotations,
                attributes={"frame": extracted_frame_idx},
            )
        )
        if is_error:
            self.logger.warning(
                f"Frame {extracted_frame_idx} has associated errors in the CSV. Skipping."
            )
            self.count_frames_with_errors += 1
            return

        if self.tracker_manager is not None:
            self.tracker_manager.filter_dead_trackers()
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
        Extract blobs from frame masks, optionally applying automatic cleaning.

        When `self.no_auto` is False (default), each mask goes through blob
        filtering (area/size rules), YOLO classification, and anomaly detection
        before being accepted. When `self.no_auto` is True, all non-empty
        masks are accepted as-is, keeping only the largest blob per object.
        """
        original_image = input_image.copy()

        all_blobs = []
        for obj_id, sparse_object_mask in frame_masks.items():
            if is_empty_sparse_tensor(sparse_object_mask):
                self.count_empty_instance_masks += 1
                continue

            # Binary masks
            dense_object_mask = sparse_mask_tensor_to_dense_numpy(sparse_object_mask)

            if self.no_auto:
                # Skip automatic cleaning: no blob filtering, classification, or anomaly detection.
                # Just extract all blobs and keep the largest one per object.
                raw_blobs = list(
                    get_blobs_from_mask(dense_object_mask, obj_id, extracted_frame_idx)
                )
                if raw_blobs:
                    dominant_blob = max(raw_blobs, key=lambda b: b.area)
                    if self.notebook_debug or (self.video_writer is not None):
                        self.draw_bbox_and_id(input_image, dominant_blob, "white")
                    all_blobs.append(dominant_blob)
            else:
                # Filter blob by basic featurs like area and size, to remove small blobs
                # TODO: maybe return blobs first and then filter
                filtered_blobs = self._get_filtered_blobs(
                    dense_object_mask, obj_id, extracted_frame_idx
                )

                # Generate image crops based on blobs data
                blob_patches = self._get_blob_patches(
                    original_image, filtered_blobs, bg_mode=self.bg_mode
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
                    # dominant_blob.compute_convexity_defects()
                    # dominant_blob.save_crop_and_mask()

                    self.tracker_manager.update(dominant_blob)
                    results = self.tracker_manager.predict(obj_id)

                    # NOTE:
                    # - a white box and label with an ID indicate this blob has not been rejected
                    # - a red label with a red rectangle indicates the blob has been rejected by the anomaly detector
                    # - no label indicates the blob was not processed by the anomaly detector and will be included in the
                    #   output dataset

                    if results["anomalies"]:

                        if self.notebook_debug or (self.video_writer is not None):
                            # Draw a red rectangle and information regarding why a mask was rejected
                            anomalies = ",".join(
                                [f"{a['type']}({a['value']})" for a in results["anomalies"]]
                            )
                            self.draw_bbox_and_id(
                                input_image,
                                dominant_blob,
                                "red",
                                extra_text=f"({anomalies}",
                            )

                    else:
                        if self.notebook_debug or (self.video_writer is not None):
                            # Draw a green box and label with the Object ID indicating that this blob has not been rejected
                            self.draw_bbox_and_id(input_image, dominant_blob, "white")
                        all_blobs.append(dominant_blob)

            if self.notebook_debug or (self.video_writer is not None):
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
        assert self.blob_rules is not None
        valid_blobs = []
        for blob in get_blobs_from_mask(dense_object_mask, obj_id, extracted_frame_idx):
            for rule in self.blob_rules:
                if not rule(blob):
                    self.logger.info(
                        f"  skipping blob {blob.blob_num}: {rule.explain(blob)}"
                    )
                    break
            else:
                valid_blobs.append(blob)

        return valid_blobs

    @staticmethod
    def _get_blob_patches(
        input_image: np.ndarray,
        blobs: list[BlobInfo],
        bg_mode: Literal["gray", "overlay"] | None = None,
    ):
        """Return image patches, based on the blobs information"""
        return [blob.mask_and_crop_blob(input_image, bg_mode) for blob in blobs]

    def _classify_blobs(
        self, blobs: list[BlobInfo], patches: list[np.ndarray]
    ) -> list[BlobInfo]:
        """Get classified bounding boxes from blobs."""
        assert self.classifier is not None
        assert (
            self.correct_class is not None
            and self.incorrect_class is not None
            and self.incorrect_cls_conf_thresh is not None
        )

        filtered_blobs = []
        for blob, masked_patch in zip(blobs, patches):
            results = self.classifier(masked_patch, verbose=False)[0]
            incorrect_class_index = self.class_to_index[self.incorrect_class]
            incorrect_class_pred_conf = results.probs.data[incorrect_class_index].item()
            pred_class = (
                self.incorrect_class
                if incorrect_class_pred_conf >= self.incorrect_cls_conf_thresh
                else self.correct_class
            )

            if pred_class == self.correct_class:
                blob.predicted_class = pred_class
                filtered_blobs.append(blob)
                if self.notebook_debug:
                    cv2_imshow(masked_patch)
            else:
                self.logger.info(
                    f"  skipping blob {blob.blob_num}: classified as {pred_class}"
                )

        return filtered_blobs

    def create_datumaro_annotations(self, blobs: list[BlobInfo]) -> list[Annotation]:
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

            dense_mask = blob.get_blob_mask()
            compressed_rle = mask_tools.mask_to_rle(dense_mask)
            uncompressed_rle = mask_tools.to_uncompressed_rle(
                compressed_rle, width=blob.w, height=blob.h
            )
            attributes: dict[str, int | list[float] | str] = {"ObjID": blob.obj_id}
            gt = self.get_closest_gt_location(blob.frame_idx, blob.obj_id)
            if gt is not None:
                gt_location, gt_obj_id, gt_frame_extracted, gt_frame_original = gt
                attributes["gt_location"] = gt_location
                attributes["gt_obj_id"] = gt_obj_id
                attributes["gt_frame_extracted"] = gt_frame_extracted
                attributes["gt_frame_original"] = gt_frame_original

            output.append(
                RleMask(
                    rle=uncompressed_rle,
                    label=label_id,
                    attributes=attributes,
                )
            )

        return output

    def create_empty_datumaro_annotations(
        self, blobs: list[BlobInfo]
    ) -> list[Annotation]:
        """Like create_datumaro_annotations but with an empty (all-zero) mask.

        When `blobs` is empty, produces a single empty-mask annotation using
        the provided frame_idx, obj_id, w, h so gt_ attributes are still captured.
        """

        output = []
        for blob in blobs:
            label_id = get_label_id(
                self.chunked_df,
                self.col_class_name,
                self.col_instance_id,
                blob.obj_id,
                self.label_categories,
            )

            uncompressed_rle = {"size": [2160, 3840], "counts": b""}
            attributes: dict[str, int | list[float] | str] = {"ObjID": blob.obj_id}
            gt = self.get_closest_gt_location(blob.frame_idx, blob.obj_id)
            if gt is not None:
                gt_location, gt_obj_id, gt_frame_extracted, gt_frame_original = gt
                attributes["gt_location"] = gt_location
                attributes["gt_obj_id"] = gt_obj_id
                attributes["gt_frame_extracted"] = gt_frame_extracted
                attributes["gt_frame_original"] = gt_frame_original
            output.append(
                RleMask(rle=uncompressed_rle, label=label_id, attributes=attributes)
            )

        return output

    def _print_statistics(self) -> None:
        """Print processing statistics."""
        print(f"Count of empty instance masks: {self.count_empty_instance_masks}")
        print(f"Count of frames with errors: {self.count_frames_with_errors}")
