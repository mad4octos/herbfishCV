from dataclasses import dataclass
from typing import Optional
import torch
import cv2
import numpy as np

# Local imports
from common import sparse_mask_tensor_to_dense_numpy, dense_mask_numpy_to_sparse_tensor
from plot_utils import draw_mask_overlay


@dataclass
class BlobInfo:
    """Container for blob-level metadata and geometry"""

    frame_idx: int
    obj_id: int
    blob_num: int
    area: int
    x: int
    y: int
    w: int
    h: int
    centroid_x: float
    centroid_y: float

    # labeled_mask is expected to be set with store_mask
    # labeled mask where pixel == blob_num is this blob
    labeled_mask: Optional[torch.Tensor] = None

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

    def mask_and_crop_blob(
        self, image: np.ndarray, remove_background=True, overlay_alpha=0.2
    ):
        """Mask and crop an image.

        This function can either remove the background (default) or highlight the foreground (the blob)."""

        image = np.copy(image)
        if remove_background:
            # Preserve only the foreground object, make everything else black
            image[self.get_dense_mask() != self.blob_num] = 0
        else:
            image = draw_mask_overlay(
                image,
                self.get_dense_mask(),
                self.blob_num,
                color=(0, 0, 255),
                alpha=overlay_alpha,
            )
        return self.crop_from_image(image)

    def compute_extent(self):
        """ """
        bbox_area = self.w * self.h
        self.extent = round(self.area / bbox_area if bbox_area > 0 else 0.0, 2)

    def compute_solidity(self):
        """ """

        mask = (self.get_dense_mask() == self.blob_num).astype(np.uint8) * 255

        # Find outer contours and build a single hull
        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            raise Exception("Problem finding Contours")

        # TODO: assert there is only one contour
        hulls = []
        for i in range(len(contours)):
            hull = cv2.convexHull(contours[i], False)
            hull_area = cv2.contourArea(hull)
            hulls.append(hull)

        self.solidity = round(self.area / hull_area, 2)

    def compute_compactness(self, approx_eps_frac: float = 0.003):
        """
        approx_eps_frac: contour smoothing to reduce pixel noise
        """
        m = (self.get_dense_mask() == self.blob_num).astype(np.uint8)

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

    def get_blob_mask(self):
        """ """
        mask = (self.get_dense_mask() == self.blob_num).astype(np.uint8) * 255
        return mask

    def store_mask(self, array: np.ndarray):
        """Store labeled mask as a sparse tensor"""
        self.labeled_mask = dense_mask_numpy_to_sparse_tensor(array)

    def get_dense_mask(self):
        """Return the sparse labeled mask as a dense array"""
        return sparse_mask_tensor_to_dense_numpy(self.labeled_mask)
