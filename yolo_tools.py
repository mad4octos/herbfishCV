# Standard library imports
from pathlib import Path
from types import SimpleNamespace

# External imports
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from ultralytics import YOLO

# Local imports
from yolo_dataset import RGBAClassificationDataset

"""
Ultralytics' built-in .val() evaluates classification accuracy using argmax
over the softmax output — equivalent to a fixed 0.5 confidence threshold in
binary classification. It does not expose the full F1-vs-threshold curve, nor
does it return the raw per-sample confidence values needed to sweep it.

To find the threshold that maximises F1 on the validation set,
you need the positive-class probability for every sample, which .val()
discards internally after computing its fixed-threshold metrics. Hence
the manual collection of confidences via model.predict(), followed by the
vectorised threshold sweep in find_best_threshold().
"""


def find_best_threshold(
    positive_class_confs: torch.Tensor,  # (N,)
    targets: torch.Tensor,  # (N,) binary ground truth
    n_thresholds: int = 200,
    plot: bool = True,
) -> tuple[float, float]:
    """
    Sweep confidence thresholds and find the one maximising F1.
    Rejected samples (conf < t) are counted as false negatives.

    Returns:
        best_threshold, best_f1
    """
    thresholds = torch.linspace(0, 1, n_thresholds)  # (T,)

    # Vectorised sweep — no Python loop over thresholds
    # preds[i, j] = 1 if sample i conf >= threshold j
    preds_matrix = positive_class_confs.unsqueeze(1) >= thresholds.unsqueeze(
        0
    )  # (N, T)
    targets_bool = targets.bool().unsqueeze(1)  # (N, 1)

    tp = (preds_matrix & targets_bool).sum(0).float()  # (T,)
    fp = (preds_matrix & ~targets_bool).sum(0).float()
    fn = (~preds_matrix & targets_bool).sum(0).float()

    precision = tp / (tp + fp).clamp(min=1e-9)
    recall = tp / (tp + fn).clamp(min=1e-9)
    f1_scores = 2 * precision * recall / (precision + recall).clamp(min=1e-9)

    best_idx = f1_scores.argmax()
    best_threshold = thresholds[best_idx].item()
    best_f1 = f1_scores[best_idx].item()

    if plot:
        t_np = thresholds.numpy()
        f1_np = f1_scores.numpy()
        pr_np = precision.numpy()
        rc_np = recall.numpy()

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # F1 vs threshold
        ax = axes[0]
        ax.plot(t_np, f1_np, label="F1")
        ax.plot(t_np, pr_np, label="Precision", linestyle="--", alpha=0.7)
        ax.plot(t_np, rc_np, label="Recall", linestyle="--", alpha=0.7)
        ax.axvline(
            best_threshold,
            color="red",
            linestyle=":",
            label=f"Best t={best_threshold:.3f}",
        )
        ax.axhline(best_f1, color="red", linestyle=":", alpha=0.5)
        ax.set_xlabel("Confidence threshold")
        ax.set_ylabel("Score")
        ax.set_title("F1 / Precision / Recall vs Threshold")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Precision-Recall curve
        ax = axes[1]
        ax.plot(rc_np, pr_np)
        ax.scatter(
            recall[best_idx].item(),
            precision[best_idx].item(),
            color="red",
            zorder=5,
            label=f"Best t={best_threshold:.3f}  F1={best_f1:.3f}",
        )
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    return best_threshold, best_f1


def get_targets_and_confs(
    model: YOLO, dataloader, positive_class_name: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run inference over the dataloader and collect top-1 predictions,
    positive-class confidences, and ground truth labels.

    Returns:
        pos_confs:   (N,) confidence of the positive class
        all_targets: (N,) ground truth labels
    """
    class_idx = next(
        i for i, name in model.names.items() if name == positive_class_name
    )

    pos_confs, all_targets = [], []

    for batch in dataloader:
        imgs = batch["img"]
        labels = batch["cls"]

        results = model.predict(imgs, verbose=False)

        for result, label in zip(results, labels):
            pos_confs.append(result.probs.data[class_idx].item())
            all_targets.append(label.item())
    return (
        torch.tensor(all_targets),
        torch.tensor(pos_confs),
    )


def _make_rgba_dataloader(
    data_path: Path, batch_size: int, bg_mode: str, imgsz: int
) -> DataLoader:
    dataset_args = SimpleNamespace(imgsz=imgsz, cache=False, fraction=1.0)
    dataset = RGBAClassificationDataset(
        root=str(data_path),
        args=dataset_args,
        augment=False,
        prefix=data_path.name,
        bg_mode=bg_mode,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def evaluate_and_report(
    model: YOLO,
    data_root: Path,
    batch_size: int,
    imgsz: int,
    incorrect_class: str,
    header: str,
    report_path: Path,
    bg_mode: str = "gray",
) -> tuple[float, float]:
    """
    Find the best confidence threshold on the val split, evaluate on the test
    split, print the classification report, and write a summary to disk.

    Returns:
        best_threshold, best_f1
    """
    val_dataloader = _make_rgba_dataloader(data_root / "val", batch_size, bg_mode, imgsz)
    test_dataloader = _make_rgba_dataloader(data_root / "test", batch_size, bg_mode, imgsz)
    val_targets, val_confs = get_targets_and_confs(
        model, val_dataloader, positive_class_name=incorrect_class
    )
    test_targets, test_confs = get_targets_and_confs(
        model, test_dataloader, positive_class_name=incorrect_class
    )
    best_t, best_f1 = find_best_threshold(val_confs, val_targets)
    print(f"Best threshold: {best_t:.3f}  →  F1 max: {best_f1:.3f}")
    preds = (test_confs >= best_t).to(torch.uint8)
    report = classification_report(
        test_targets, preds, target_names=list(model.names.values()), digits=3
    )
    print(report)

    summary = (
        f"{header}"
        f"Threshold (val, '{incorrect_class}' class): {best_t:.4f}\n"
        f"F1 at threshold (val, '{incorrect_class}' class): {best_f1:.4f}\n\n"
        f"CLASSIFICATION REPORT — TEST SPLIT\n\n"
        f"{report}"
    )
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(summary)
    print(f"\nReport saved to {report_path}")

    return best_t, best_f1
