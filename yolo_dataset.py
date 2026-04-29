# Standard Library imports
import copy
import random
from collections import Counter

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
from ultralytics.utils import LOGGER
from ultralytics.utils.metrics import ClassifyMetrics
from ultralytics.utils.torch_utils import is_parallel, torch_distributed_zero_first
from ultralytics.data import build_dataloader

# Local imports
from plot_utils import draw_mask_overlay

YOLO_GRAY = (114, 114, 114)  # default YOLO letterbox fill value (BGR)


class RGBAClassificationDataset(ClassificationDataset):
    def __init__(self, *pos_args, bg_mode: str = "gray", **kwargs):
        """
        Args:
            bg_mode (str): How to handle the background (alpha == 0) region.
                "gray"    : replace background with YOLO gray (114, 114, 114). Default.
                "overlay" : draw a semi-transparent red overlay on the foreground.
        """
        if bg_mode not in ("gray", "overlay"):
            raise ValueError(f"bg_mode must be 'gray' or 'overlay', got '{bg_mode}'")

        namespace_args = kwargs.get("args")
        augment = kwargs.get("augment", False)
        fraction = getattr(namespace_args, "fraction", 1.0)

        # Prevent the parent from slicing all samples uniformly; apply fraction
        # selectively to the majority class only (to reduce class imbalance and speed up training).
        if namespace_args is not None and fraction < 1.0:
            modified_namespace = copy.copy(namespace_args)
            modified_namespace.fraction = 1.0
            kwargs = {**kwargs, "args": modified_namespace}

        super().__init__(*pos_args, **kwargs)

        # augment=True only during training (build_dataset passes augment=mode=="train")
        if augment and fraction < 1.0:
            self._apply_fraction_to_majority(fraction)

        self.bg_mode = bg_mode
        self.train_mode = "train" in self.prefix
        self.probabilities = self._compute_probabilities()

    def _apply_fraction_to_majority(self, fraction: float) -> None:
        """Reduce the majority class to `fraction` of its original size.

        Leaves all minority classes untouched, so the net effect is to shrink
        the class imbalance rather than uniformly sub-sampling every class.

        Args:
            fraction: Proportion of majority-class samples to keep (0 < fraction < 1).
        """
        class_indices = np.array([s[1] for s in self.samples])
        majority_class = int(np.bincount(class_indices).argmax())

        majority_samples = [s for s in self.samples if s[1] == majority_class]
        minority_samples = [s for s in self.samples if s[1] != majority_class]

        n_before = len(majority_samples)
        random.shuffle(majority_samples)
        majority_samples = majority_samples[: round(n_before * fraction)]
        n_after = len(majority_samples)
        self.samples = majority_samples + minority_samples

        self._log_fraction_result(
            fraction, majority_class, n_before, n_after, minority_samples
        )

    def _log_fraction_result(
        self,
        fraction: float,
        majority_class: int,
        n_before: int,
        n_after: int,
        minority_samples: list,
    ) -> None:
        """Log a per-class summary of the fraction sub-sampling applied to the majority class.

        Args:
            fraction: The fraction that was applied.
            majority_class: Class index of the majority class.
            n_before: Number of majority-class samples before sub-sampling.
            n_after: Number of majority-class samples after sub-sampling.
            minority_samples: Remaining (untouched) samples from all non-majority classes.
        """
        class_names = self.base.classes
        minority_counts = Counter(s[1] for s in minority_samples)
        lines = [
            f"{self.prefix}fraction={fraction} applied to majority class only:",
            f"  {class_names[majority_class]}: {n_before} -> {n_after} samples",
            *(
                f"  {class_names[i]}: {n} -> {n} samples (unchanged)"
                for i, n in sorted(minority_counts.items())
            ),
        ]
        LOGGER.info("\n".join(lines))

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
            (RGBAClassificationDataset): Dataset for the specified mode.
        """
        return RGBAClassificationDataset(
            root=img_path,
            args=self.args,
            augment=mode == "train",
            prefix=mode,
            bg_mode=self.bg_mode,
        )

    def get_dataloader(
        self,
        dataset_path: str,
        batch_size: int = 16,
        rank: int = 0,
        mode: str = "train",
    ):
        """The only difference betwen this method and the original ultralytics 8.3.227
        ClassificationTrainer.get_dataloader is that build_dataloader is called with pin_memory=False.
        This fixes OOM erros when running multiple trials with Optuna.
        """
        # init dataset *.cache only once if DDP
        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode)

        loader = build_dataloader(
            dataset,
            batch_size,
            self.args.workers,
            rank=rank,
            drop_last=self.args.compile,
            pin_memory=False,
        )
        # Attach inference transforms
        if mode != "train":
            if is_parallel(self.model):
                self.model.module.transforms = loader.dataset.torch_transforms
            else:
                self.model.transforms = loader.dataset.torch_transforms
        return loader

    def get_validator(self):
        self.loss_names = ["loss"]
        return RGBAClassificationValidator(
            self.test_loader, self.save_dir, args=self.args, _callbacks=self.callbacks
        )
