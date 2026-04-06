# External mports
import cv2
import numpy as np
import numpy.typing as npt
import torch
from IPython import display
from PIL import Image


def cv2_imshow(a):
    """A replacement for cv2.imshow() for use in Jupyter notebooks.

    From: Google Colab imported libraries.

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


def torch_to_cv2(image: torch.Tensor, is_mask=False) -> npt.NDArray[np.uint8]:
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


def sparse_mask_tensor_to_dense_numpy(
    sparse_tensor: torch.Tensor,
) -> npt.NDArray[np.uint8]:
    """ """
    dense_tensor = sparse_tensor.to_dense()
    return torch_to_cv2(dense_tensor, is_mask=True)


def dense_mask_numpy_to_sparse_tensor(dense_numpy: np.ndarray):
    """ """
    return torch.tensor(dense_numpy).to_sparse()


def is_empty_sparse_tensor(sparse_tensor: torch.Tensor):
    """ """
    return sparse_tensor.values().numel() == 0


def calc_sparse_memory_consumption(sparse_tensor: torch.Tensor):
    """
    The memory consumption of a sparse COO tensor is at least...
    https://docs.pytorch.org/docs/stable/sparse.html#sparse-coo-tensors
    """
    # ndim is the dimensionality of the tensor and nse is the number of specified elements
    ndim, nse = sparse_tensor.indices().shape

    # `1` for uint8
    element_size = 1  # size of element type in bytes
    return (ndim * 8 + element_size) * nse


def get_memory_consumption(dense_tensor: torch.Tensor):
    """ """
    h, w = dense_tensor.shape
    # `1` for uint8
    element_size = 1  # size of element type in bytes
    return h * w * element_size
