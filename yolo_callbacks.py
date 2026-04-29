import json
from pathlib import Path
from typing import Literal

# External imports
import numpy as np
import torch
from sklearn.metrics import f1_score
from livelossplot import PlotLosses
from livelossplot.outputs.matplotlib_plot import MatplotlibPlot
from ultralytics.models.yolo.classify.train import ClassificationTrainer
from ultralytics.models.yolo.classify.val import ClassificationValidator
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils.metrics import ap_per_class, smooth
from ultralytics.engine.validator import BaseValidator


class LossPlotCallbacks:
    def __init__(
        self,
        mode: Literal["detection", "classification"],
        figpath: str = "loss_plot.png",
        names: list[str] | None = None,
    ):
        f1_keys = [f"f1/{n}" for n in names] if names else ["f1_max"]

        if mode == "detection":
            groups = {
                "Box Loss": ["train/box_loss", "val/box_loss"],
                "Cls Loss": ["train/cls_loss", "val/cls_loss"],
                "DFL Loss": ["train/dfl_loss", "val/dfl_loss"],
                "F1 per class": f1_keys,
            }
            self.loss_names = ("box_loss", "cls_loss", "dfl_loss")
        elif mode == "classification":
            groups = {
                "Loss": ["train/loss", "val/loss"],
                "F1 per class": f1_keys,
            }
            self.loss_names = ("loss",)
        else:
            raise ValueError(
                f"Unknown mode '{mode}'. Expected 'detection' or 'classification'."
            )

        self.mode = mode

        self.mpl_plot = MatplotlibPlot(figpath=figpath)
        self.live_plot = PlotLosses(groups=groups, outputs=[self.mpl_plot])  # type: ignore[arg-type]
        self.live_logs: dict = {}

        self.f1_max: float = 0.0
        self.max_conf_for_f1: float = 0.0
        self.max_f1_by_class: dict = {}

        self.plot_only = False

    def on_train_start(self, trainer: DetectionTrainer | ClassificationTrainer):
        self.mpl_plot.figpath = str(trainer.save_dir / "loss_plot.png")

    def on_train_epoch_end(self, trainer: DetectionTrainer | ClassificationTrainer):
        """Record the normalised training sub-losses for the current epoch."""
        if isinstance(trainer, DetectionTrainer):
            tloss = trainer.tloss.detach().cpu().numpy()  # shape (3,): [box, cls, dfl]
            for name, value in zip(trainer.loss_names, tloss):
                self.live_logs[f"train/{name}"] = float(value)
        elif isinstance(trainer, ClassificationTrainer):
            if trainer.tloss is not None:
                tloss = float(trainer.tloss.detach().cpu().numpy())
                self.live_logs["train/loss"] = tloss

    def on_val_batch_end(self, validator: BaseValidator):
        """
        Compute the best F1 score and corresponding confidence threshold from accumulated stats.

        This logic is lifted from ultralytics internals and runs here — rather than in
        on_val_end — because validator.metrics.stats is cleared before on_val_end fires.
        As a side-effect, this callback executes once per validation batch; only the last
        call's results are meaningful (they cover all batches seen so far).

        Populates:
            self.f1_max: Highest mean F1 across all classes.
            self.max_conf_for_f1: Confidence threshold that yields self.f1_max.
            self.max_f1_by_class: Per-class dict of {"f1": float, "conf": float}.

        Skipped entirely when self.plot_only is True.
        """
        if self.plot_only:
            return

        if isinstance(validator, DetectionValidator):
            ###############################################
            # Obtain the best conf threshold
            ###############################################
            stats = {
                k: np.concatenate(v, 0) for k, v in validator.metrics.stats.items()
            }
            if not stats:
                print("No stats")
                return stats

            (
                tp,
                fp,
                p,
                r,
                f1,
                ap,
                unique_classes,
                p_curve,
                r_curve,
                f1_curve,
                x,
                prec_values,
            ) = ap_per_class(
                stats["tp"],
                stats["conf"],
                stats["pred_cls"],
                stats["target_cls"],
                plot=False,
                save_dir=Path(),
                names=validator.metrics.names,
                on_plot=None,
                prefix="Box",
            )
            x = np.linspace(0, 1, 1000)

            f1_mean = smooth(f1_curve.mean(0), 0.1)
            best_f1_idx = f1_mean.argmax()

            #################################
            # Store values to print them later
            #################################

            # Overall max f1 and conf
            self.max_conf_for_f1 = x[best_f1_idx]
            self.f1_max = f1_mean.max()

            # Per-class max f1 and conf
            max_f1_idxs = f1_curve.argmax(axis=1)
            self.max_f1_by_class = {}
            for i, max_cls_f1_idx in enumerate(max_f1_idxs):
                class_name = validator.names[i]
                self.max_f1_by_class[class_name] = {
                    "conf": x[max_cls_f1_idx],
                    "f1": f1_curve[i, max_cls_f1_idx].item(),
                }
        elif isinstance(validator, ClassificationValidator):
            # FIXME: this requires access to raw probabilities to be actually f1 max
            ###############################################
            # Compute per-class F1 from accumulated preds
            ###############################################

            if not validator.pred or not validator.targets:
                return

            preds = torch.cat(validator.pred)[:, 0].cpu().numpy()  # top-1 class index
            targets = torch.cat(validator.targets).cpu().numpy()

            per_class_f1: np.ndarray = f1_score(
                targets, preds, average=None, zero_division=0
            )  # type: ignore[assignment]

            self.f1_max = float(per_class_f1.mean())
            # not applicable for classification because don't have access to probabilities
            self.max_conf_for_f1 = float("nan")

            self.max_f1_by_class = {
                validator.names[i]: {"f1": float(f1), "conf": float("nan")}  # type: ignore[index]
                for i, f1 in enumerate(per_class_f1)
            }

    def _print_detection_summary(self) -> None:
        """Print detection loss and F1 summary to stdout, clearing the screen first."""
        val_totals = [
            self.live_logs.get(f"val/{n}", float("nan")) for n in self.loss_names
        ]
        print(
            f"Val loss   — box: {val_totals[0]:.4f}  cls: {val_totals[1]:.4f}  dfl: {val_totals[2]:.4f}"
        )

        train_totals = [
            self.live_logs.get(f"train/{n}", float("nan")) for n in self.loss_names
        ]
        print(
            f"Train loss — box: {train_totals[0]:.4f}  cls: {train_totals[1]:.4f}  dfl: {train_totals[2]:.4f}"
        )
        gaps = [v - t for v, t in zip(val_totals, train_totals)]
        print(
            f"Val gap    — box: {gaps[0]:.4f}  cls: {gaps[1]:.4f}  dfl: {gaps[2]:.4f}"
        )

        print(f"\nMax overall F1={self.f1_max:.3f} at conf={self.max_conf_for_f1:.3f}")
        for class_name, d in self.max_f1_by_class.items():
            print(f"  - {class_name:>10} | F1={d['f1']:.3f} at conf={d['conf']:.3f}")

    def _print_classification_summary(self) -> None:
        """Print classification loss and F1 summary to stdout, clearing the screen first."""
        vloss = self.live_logs["val/loss"]
        print(f"Val loss   — {vloss:.4f}")
        # Note: in classification I don't have access to the probabilities, so this is not max F1 but F1
        print(f"\nF1={self.f1_max:.3f}")
        for class_name, d in self.max_f1_by_class.items():
            print(f"  - {class_name:>10} | F1={d['f1']:.3f}")

    def on_val_end(self, validator: BaseValidator):
        """
        Finalise the epoch: log validation loss, update the live plot, print a summary,
        save the confusion matrix, and write F1/confidence scores to JSON.

        Reads self.f1_max and self.max_f1_by_class set by on_val_batch_end.
        Clears self.live_logs at the end so the next epoch starts clean.
        JSON export and live_logs.clear() are skipped when self.plot_only is True.
        """

        ########################################
        # Compute normalised validation loss
        ########################################
        n_batches = len(validator.dataloader)  # type: ignore[arg-type]
        vloss = validator.loss.detach().cpu().numpy() / n_batches

        if isinstance(validator, DetectionValidator):
            for name, value in zip(self.loss_names, vloss):
                self.live_logs[f"val/{name}"] = float(value)
            self.live_logs["f1_max"] = float(self.f1_max)
            for class_name, d in self.max_f1_by_class.items():
                self.live_logs[f"f1/{class_name}"] = float(d["f1"])

        elif isinstance(validator, ClassificationValidator):
            self.live_logs["val/loss"] = float(vloss)
            self.live_logs["f1_max"] = float(self.f1_max)
            for class_name, d in self.max_f1_by_class.items():
                self.live_logs[f"f1/{class_name}"] = float(d["f1"])

        if not self.plot_only:
            ########################################
            # Export F1 / confidence scores to JSON
            ########################################
            scores = {
                "overall": {
                    "f1": float(self.f1_max),
                    "conf": float(self.max_conf_for_f1),
                },
                "per_class": {
                    k: {"f1": float(v["f1"]), "conf": float(v["conf"])}
                    for k, v in self.max_f1_by_class.items()
                },
            }
            scores_path = validator.save_dir / "f1_conf_scores.json"
            with open(scores_path, "w") as f:
                json.dump(scores, f, indent=2)

        ########################################
        # Save confusion matrix
        ########################################
        validator.confusion_matrix.plot(
            save_dir=str(validator.save_dir),
            normalize=False,
            on_plot=validator.on_plot,
        )

        ########################################
        # Update live plot
        ########################################
        self.live_plot.update(self.live_logs)
        self.live_plot.send()

        if isinstance(validator, DetectionValidator):
            self._print_detection_summary()
        elif isinstance(validator, ClassificationValidator):
            self._print_classification_summary()

        ########################################
        # Reset logs for next epoch
        ########################################
        if not self.plot_only:
            self.live_logs.clear()
