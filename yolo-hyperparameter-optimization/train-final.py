#!/usr/bin/env python3
# train_final.py

# Standard Library imports
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# External imports
import yaml
from ultralytics import YOLO

# Local imports
from yolo_callbacks import LossPlotCallbacks
from yolo_dataset import RGBClassificationTrainer
from yolo_tools import evaluate_and_report


def main():
    parser = argparse.ArgumentParser(
        description="Final training with best hyperparameters"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to dataset root (with train/val/test splits)",
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument("--device", type=str, default="0", help="CUDA device")
    parser.add_argument(
        "--project", type=str, default="./final_model", help="Project directory"
    )
    parser.add_argument(
        "--hyp",
        type=str,
        default="best_hyperparameters.yaml",
        help="Path to hyperparameters YAML",
    )
    parser.add_argument(
        "--incorrect-class",
        type=str,
        default="incorrect",
        help="Name of the positive (incorrect) class used for threshold search",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Fraction of dataset to use (0.0–1.0). Use a small value (e.g. 0.1) for quick smoke-test runs.",
    )
    parser.add_argument(
        "--name", type=str, default="final_train", help="Run name (subfolder under project)"
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of dataloader workers"
    )
    args = parser.parse_args()

    # Load hyperparameters from YAML
    with open(args.hyp, "r") as f:
        hyp = yaml.safe_load(f)

    print("Loaded hyperparameters:")
    for key, value in hyp.items():
        print(f"  {key}: {value}")

    # bg_mode and model are not YOLO train args — remove before passing to model.train()
    bg_mode = hyp.pop("bg_mode")
    RGBClassificationTrainer.bg_mode = bg_mode
    model_name = hyp.pop("model")

    # Initialize the model
    print(f"\nInitializing model: {model_name}")
    model = YOLO(model_name)

    names = sorted(p.name for p in (Path(args.data) / "train").iterdir() if p.is_dir())
    cbs = LossPlotCallbacks(
        figpath=f"{args.project}/{args.name}/loss_plot.png",
        mode="classification",
        names=names,
    )
    model.add_callback("on_train_epoch_end", cbs.on_train_epoch_end)
    model.add_callback("on_val_batch_end", cbs.on_val_batch_end)
    model.add_callback("on_val_end", cbs.on_val_end)

    # Train with best hyperparameters
    print("\nStarting training with best hyperparameters...")
    model.train(
        data=args.data,
        epochs=args.epochs,
        device=args.device,
        project=args.project,
        name=args.name,
        workers=args.workers,
        trainer=RGBClassificationTrainer,
        deterministic=True,
        fraction=args.fraction,
        **hyp,
    )

    print("\nTraining complete!")

    batch_size = hyp.get("batch", 16)
    imgsz = hyp.get("imgsz", 224)
    scale = hyp.get("scale", 0.0)
    hyp_lines = "\n".join(f"  {k}: {v}" for k, v in hyp.items())
    header = (
        f"FINAL TRAINING RESULTS\n"
        f"Model: {model_name}\n"
        f"bg_mode: {bg_mode}\n"
        f"Epochs: {args.epochs}\n"
        f"Hyperparameters file: {args.hyp}\n"
        f"Hyperparameters:\n{hyp_lines}\n\n"
    )
    evaluate_and_report(
        model,
        Path(args.data),
        batch_size,
        imgsz,
        args.incorrect_class,
        header,
        "final_training_results.txt",
        bg_mode=bg_mode,
        scale=scale,
    )


if __name__ == "__main__":
    main()
