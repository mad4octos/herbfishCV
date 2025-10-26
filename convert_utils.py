# Standard Library imports
import sys
import pickle
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


# External imports
import numpy as np
import pandas as pd
import torch
import cv2
import datumaro.components.media
import datumaro.components.dataset
import datumaro.components.dataset_base
from datumaro.components.annotation import AnnotationType, LabelCategories
from datumaro.components.annotation import Bbox as DatumaroBbox
from tqdm import tqdm
from IPython import display
from PIL import Image

ENTER_FRAME_MARKER = 3
EXIT_FRAME_MARKER = 4

sys.path.append(
    r"C:\Users\Manuel\Desktop\Documentos\1.PROJECTS\COMPUTER VISION\TRACKING\SORT_TRACKER"
)
# Local imports
from sort import draw_tracking_box

# Type aliases for the masks file
FrameIndex = int
ObjectIndex = int
MasksType = dict[FrameIndex, dict[ObjectIndex, torch.Tensor]]


def draw_mask_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    class_id: int,
    alpha: float = 0.15,
    color=None,
    binary_mask=False,
):
    """
    Draw a semi-transparent colored overlay on regions of an image where the mask equals a given class ID.
    """
    if color is None:
        color = color_from_index_bgr(class_id)
    overlay = np.copy(image)

    if binary_mask:
        fg = mask != 0
    else:
        fg = mask == class_id

    overlay[fg] = color
    weighted_image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, dst=image)

    return weighted_image


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


def sparse_mask_tensor_to_dense_numpy(sparse_tensor):
    """ """
    dense_tensor = sparse_tensor.to_dense()
    return torch_to_cv2(dense_tensor, is_mask=True)


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
        self.tracker = tracker
        self.background_subtractor = background_subtractor
        self.dataset_items: list[datumaro.components.dataset_base.DatasetItem] = []
        self.count_empty_instance_masks = 0
        self.count_frames_with_errors = 0
        self.debug_viz = debug
        self.target_class = target_class
        self.max_frames = max_frames
        self.start_frame = start_frame

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

        if self.background_subtractor is not None:
            foreground_mask = self.background_subtractor.apply(input_image)
            foreground_mask = threshold(foreground_mask)
            # foreground_mask = morph_opening(foreground_mask, iterations=4)
            # foreground_mask = morph_closing(foreground_mask, iterations=10)
            cv2_imshow(foreground_mask)

        bounding_boxes = self._extract_datumaro_bounding_boxes(input_image, frame_masks)
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

        Parameters
        ----------
        frame_masks : dict
            Dictionary mapping object IDs to sparse tensor masks.
        input_image : np.ndarray
            The original input image.
        masked_image : np.ndarray
            Copy of the input image for visualization.

        Returns
        -------
        list
            List of bounding boxes with labels.
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

            # Get blobs and filter by area, height and width
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
                    alpha=0.55,
                    binary_mask=True,
                )
            ############################################################################################################
            # FOR SORT
            ############################################################################################################
            # Classify crops as fish/other
            # filtered_boxes = self._get_filtered_bounding_boxes(
            #     filtered_blobs, blob_patches
            # )
            # Show boxes offish in green
            # if self.debug_viz:
            #     for box in filtered_boxes:
            #         x1, y1, x2, y2 = map(int, box)
            #         cv2.rectangle(
            #             img=input_image,
            #             pt1=(x1, y1),
            #             pt2=(x2, y2),
            #             color=(0, 255, 0),
            #             thickness=5,
            #         )

            ############################################################################################################
            # FOR DEEPSORT
            ############################################################################################################

            filtered_boxes = self._get_filtered_bounding_boxes_deepsort(
                filtered_blobs, blob_patches
            )

            all_filtered_boxes.extend(filtered_boxes)

            # # Show boxes of fish in green
            # if self.debug_viz:
            #     for box, conf, label in filtered_boxes:
            #         x, y, w, h = map(int, box)
            #         cv2.rectangle(
            #             img=input_image,
            #             pt1=(x, y),
            #             pt2=(x + w, y + h),
            #             color=(0, 255, 0),
            #             thickness=5,
            #         )
            ############################################################################################################

        # Track fish
        if self.tracker is not None:
            # For SORT
            # updated_results, predictions = self.tracker.update_tracks(filtered_boxes)
            # for observation in updated_results:
            #     draw_tracking_box(input_image, observation)
            #     # TODO: extract filtered kalman boxes from updated_results and make them the new filtered_boxes
            # # Show Kalman boxes in blue
            # if self.debug_viz:
            #     for updated_result in updated_results:
            #         x1, y1, x2, y2 = map(int, updated_result["updated_bbox"])
            #         cv2.rectangle(
            #             img=input_image,
            #             pt1=(x1, y1),
            #             pt2=(x2, y2),
            #             color=(255, 0, 0),
            #             thickness=2,
            #         )

            #     for updated_result in predictions:
            #         x1, y1, x2, y2 = map(int, updated_result)
            #         cv2.rectangle(
            #             img=input_image,
            #             pt1=(x1, y1),
            #             pt2=(x2, y2),
            #             color=(0, 255, 255),
            #             thickness=2,
            #         )

            # For DEEPSORT
            tracks = self.tracker.update_tracks(
                all_filtered_boxes, frame=original_image
            )

            for track in tracks:
                x1, y1, x2, y2 = map(int, track.to_tlbr())
                color = (0, 255, 0) if track.is_confirmed() else (0, 255, 255)

                cv2.rectangle(input_image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    input_image,
                    f"ID: {track.track_id}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    color,
                    2,
                )

            # Create a list of Datumaro Bounding Boxes
            # datumaro_boxes = self.create_datumaro_boxes(filtered_boxes, obj_id)
            # bounding_boxes.extend(datumaro_boxes)

        return bounding_boxes

    def _get_filtered_blobs(self, dense_object_mask: np.ndarray) -> list[BlobInfo]:
        """Get filtered blobs from dense mask.
        The blobs are filtered based on a minimal area, height and width."""

        valid_blobs: list[BlobInfo] = []
        for blob in iterate_blobs_from_mask(dense_object_mask):
            if self.verbose:
                x1, y1, x2, y2 = blob.bbox_xyxy
                print(
                    f"Found blob {blob.blob_num} area={blob.area} bbox=({x1},{y1},{x2},{y2})"
                )

            if blob.area < self.area_threshold:
                if self.verbose:
                    print(
                        f"  skipping blob {blob.blob_num}: area {blob.area} < {self.area_threshold}"
                    )
                continue
            if blob.w < self.min_size_threshold or blob.h < self.min_size_threshold:
                if self.verbose:
                    print(
                        f"  skipping blob {blob.blob_num}: size {blob.w}x{blob.h} < {self.min_size_threshold}"
                    )
                continue

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

    def _get_filtered_bounding_boxes_deepsort(
        self, blobs, patches
    ) -> list[tuple[tuple[int, int, int, int], float, str]]:
        """Get classified bounding boxes from blobs."""

        if self.classifier is None:
            # TODO: fallback behavior
            raise ValueError("Currently, a classifier is required")

        boxes: list[tuple[tuple[int, int, int, int], float, str]] = []
        for blob, masked_patch in zip(blobs, patches):
            results = self.classifier(
                masked_patch, verbose=False, conf=self.classifier_conf
            )[0]
            classes = results.names
            prediction = classes[results.probs.top1]
            conf = results.probs.top1conf.item()

            if prediction == self.target_class:
                boxes.append((blob.bbox_xywh, conf, prediction))
                if self.debug_viz:
                    print(f"Prediction: {prediction}")
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

        return [
            DatumaroBbox(x, y, w, h, label=label_id) for x, y, w, h in bounding_boxes
        ]

    def _print_statistics(self) -> None:
        """Print processing statistics."""
        print(f"Count of empty instance masks: {self.count_empty_instance_masks}")
        print(f"Count of frames with errors: {self.count_frames_with_errors}")


def _get_frame_filename(
    extracted_frame_idx: int, filename_num_zeros: int, extension: str = ".jpg"
) -> str:
    """
    Some extracted frames have a variable length zero-padded length.
    For the focal follow data is 4, for the stationary data is 5.
    """
    return f"{extracted_frame_idx + 1:0{filename_num_zeros}d}{extension}"


def morph_opening(image, iterations=3):
    """
    Opening: erode then dilate
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=iterations)
    return opened


def morph_closing(image, iterations=5):
    """
    Closing: dilate then erode
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    return closed


def threshold(image):
    """ """
    ret, threshold = cv2.threshold(image.copy(), 120, 255, cv2.THRESH_BINARY)
    return threshold


def cv2_imshow(a):
    """A replacement for cv2.imshow() for use in Jupyter notebooks.

    Args:
      a: np.ndarray. shape (N, M) or (N, M, 1) is an NxM grayscale image. For
        example, a shape of (N, M, 3) is an NxM BGR color image, and a shape of
        (N, M, 4) is an NxM BGRA color image.
    """
    a = a.clip(0, 255).astype("uint8")
    # cv2 stores colors as BGR; convert to RGB
    if a.ndim == 3:
        if a.shape[2] == 4:
            a = cv2.cvtColor(a, cv2.COLOR_BGRA2RGBA)
        else:
            a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
    display.display(Image.fromarray(a))


GOLDEN_RATIO_CONJUGATE = 0.618033988749895  # 1/φ per Ankerl
DEFAULT_S, DEFAULT_V = 0.5, 0.95  # suggested by the article


def hsv_to_rgb_ankerl(h: float, s: float, v: float) -> tuple[int, int, int]:
    """
    HSV in [0,1) -> RGB in 0..255.

    Modified from:
    https://martin.ankerl.com/2009/12/09/how-to-create-random-colors-programmatically/
    """
    h_i = int(h * 6)
    f = h * 6 - h_i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)

    if h_i == 0:
        r, g, b = v, t, p
    elif h_i == 1:
        r, g, b = q, v, p
    elif h_i == 2:
        r, g, b = p, v, t
    elif h_i == 3:
        r, g, b = p, q, v
    elif h_i == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q

    return int(r * 256) % 256, int(g * 256) % 256, int(b * 256) % 256


def color_from_index(
    idx: int, h0: float = 0.0, s: float = DEFAULT_S, v: float = DEFAULT_V
) -> tuple[int, int, int]:
    """
    Deterministic color for any non-negative integer.
    Consecutive indices are well-separated by stepping hue with 1/φ.
    """
    # distribute hues by adding the golden-ratio conjugate and wrapping
    h = (h0 + idx * GOLDEN_RATIO_CONJUGATE) % 1.0
    return hsv_to_rgb_ankerl(h, s, v)


def color_from_index_bgr(idx: int, **kwargs) -> tuple[int, int, int]:
    """
    Same as color_from_index but returns BGR (handy for OpenCV).
    """
    r, g, b = color_from_index(idx, **kwargs)
    return (b, g, r)
