# External impors
import cv2
import numpy as np
import numpy.typing as npt
import torch
from PIL import Image
from sklearn.metrics import f1_score
from ultralytics.data import ClassificationDataset
from ultralytics.models.yolo.classify import ClassificationValidator
from ultralytics.models.yolo.classify.train import ClassificationTrainer
from ultralytics.utils.metrics import ClassifyMetrics

# Local imports
from plot_utils import draw_mask_overlay

YOLO_GRAY = (114, 114, 114)  # default YOLO letterbox fill value (BGR)


class RGBAClassificationDataset(ClassificationDataset):
    def __init__(self, *args, bg_mode: str = "gray", **kwargs):
        """
        Args:
            bg_mode (str): How to handle the background (alpha == 0) region.
                "gray"    : replace background with YOLO gray (114, 114, 114). Default.
                "overlay" : draw a semi-transparent red overlay on the foreground.
        """
        super().__init__(*args, **kwargs)
        if bg_mode not in ("gray", "overlay"):
            raise ValueError(f"bg_mode must be 'gray' or 'overlay', got '{bg_mode}'")
        self.bg_mode = bg_mode
        self.train_mode = "train" in self.prefix
        self.probabilities = self._compute_probabilities()

    def _count_samples_per_class(
        self, class_indices: npt.NDArray[np.int64]
    ) -> npt.NDArray[np.float32]:
        """
        Count the number of samples belonging to each class, returning an array of
        length `len(self.base.classes)`. Classes with zero samples are reported as
        1 so the result can safely be used as a divisor downstream.

        Args:
            class_indices: Class index for each sample.

        Returns:
            Sample count per class, with zeros replaced by 1.
        """
        counts_per_class = np.bincount(
            class_indices, minlength=len(self.base.classes)
        ).astype(np.float32)
        counts_per_class = np.where(counts_per_class == 0, 1, counts_per_class)
        return counts_per_class

    def _compute_probabilities(self) -> npt.NDArray[np.float32]:
        """
        Compute a per-sample sampling probability inversely proportional to class frequency.

        Each sample is assigned weight 1/count(class), so within every class the
        individual weights sum to exactly 1.0 regardless of class size. Normalizing
        over all samples therefore gives each class an equal share of the probability
        mass, producing a uniform distribution across classes and causing rare classes
        to be oversampled relative to their frequency in the dataset.

        Returns:
            Float array of length len(self.samples) with per-sample probabilities
            summing to 1.
        """
        class_indices = np.array([s[1] for s in self.samples], dtype=np.int64)
        counts_per_class = self._count_samples_per_class(class_indices)
        inv_freq_per_class = 1.0 / counts_per_class
        inv_freq_per_sample = inv_freq_per_class[class_indices]
        return inv_freq_per_sample / inv_freq_per_sample.sum()

    def __getitem__(self, i: int) -> dict:
        """
        Modified from original YOLO code.
        Return subset of data and targets corresponding to given indices.

        Args:
            i (int): Index of the sample to retrieve.

        Returns:
            (dict): Dictionary containing the image and its class index.
        """

        if self.train_mode:
            i = np.random.choice(len(self.samples), p=self.probabilities)

        # filename, index, filename.with_suffix('.npy'), image
        f, j, fn, im = self.samples[i]
        if self.cache_ram:
            # Warning: two separate if statements required here, do not combine this with previous line
            if im is None:
                im = self.samples[i][3] = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        elif self.cache_disk:
            if not fn.exists():  # load npy
                np.save(
                    fn.as_posix(),
                    cv2.imread(f, cv2.IMREAD_UNCHANGED),
                    allow_pickle=False,
                )
            im = np.load(fn)
        else:
            im = cv2.imread(f, cv2.IMREAD_UNCHANGED)  # BGRA

        rgb = im[:, :, :3]
        mask = im[:, :, 3]

        if self.bg_mode == "gray":
            rgb = rgb.copy()
            rgb[mask == 0] = YOLO_GRAY
            im = rgb
        else:  # "overlay"
            im = draw_mask_overlay(
                rgb,
                mask,
                class_id=0,  # unused
                color=(0, 0, 255),
                alpha=0.2,
                binary_mask=True,
            )

        # Convert NumPy array to PIL image
        im = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        sample = self.torch_transforms(im)
        return {"img": sample, "cls": j}


class RGBAClassifyMetrics(ClassifyMetrics):
    def __init__(self):
        super().__init__()
        self.f1 = 0.0

    def process(self, targets: torch.Tensor, pred: torch.Tensor):
        """Process target classes and predicted classes to compute metrics.

        Args:
            targets (torch.Tensor): Target classes.
            pred (torch.Tensor): Predicted classes.
        """
        pred, targets = torch.cat(pred), torch.cat(targets)
        y_true = targets.cpu().numpy()
        y_pred = pred[:, 0].cpu().numpy()  # top-1 predictions
        self.top1 = float((y_true == y_pred).mean())
        self.f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    @property
    def fitness(self) -> float:
        """Return macro-F1 as fitness score."""
        return float(self.f1)

    @fitness.setter
    def fitness(self, _value):
        pass  # Ignore assignments from parent class; value is derived from self.f1


class RGBAClassificationValidator(ClassificationValidator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics = RGBAClassifyMetrics()


class RGBClassificationTrainer(ClassificationTrainer):
    bg_mode: str = "gray"

    def build_dataset(self, img_path: str, mode: str = "train", batch=None):
        """Create a ClassificationDataset instance given an image path and mode.

        Args:
            img_path (str): Path to the dataset images.
            mode (str, optional): Dataset mode ('train', 'val', or 'test').
            batch (Any, optional): Batch information (unused in this implementation).

        Returns:
            (ClassificationDataset): Dataset for the specified mode.
        """
        return RGBAClassificationDataset(
            root=img_path, args=self.args, augment=mode == "train", prefix=mode,
            bg_mode=self.bg_mode,
        )

    def get_validator(self):
        self.loss_names = ["loss"]
        return RGBAClassificationValidator(
            self.test_loader, self.save_dir, args=self.args, _callbacks=self.callbacks
        )
