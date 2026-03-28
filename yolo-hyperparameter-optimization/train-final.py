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
        "--bg-mode",
        type=str,
        default="overlay",
        choices=["gray", "overlay"],
        help="Background mode for RGBA images passed to RGBClassificationTrainer",
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
    args = parser.parse_args()

    # Load hyperparameters from YAML
    with open(args.hyp, "r") as f:
        hyp = yaml.safe_load(f)

    print("Loaded hyperparameters:")
    for key, value in hyp.items():
        print(f"  {key}: {value}")

    # bg_mode and model are not YOLO train args — remove before passing to model.train()
    bg_mode = hyp.pop("bg_mode", args.bg_mode)
    RGBClassificationTrainer.bg_mode = bg_mode
    model_name = hyp.pop("model")

    # Initialize the model
    print(f"\nInitializing model: {model_name}")
    model = YOLO(model_name)

    # Train with best hyperparameters
    print("\nStarting training with best hyperparameters...")
    model.train(
        data=args.data,
        epochs=args.epochs,
        device=args.device,
        project=args.project,
        trainer=RGBClassificationTrainer,
        deterministic=True,
        fraction=args.fraction,
        **hyp,
    )

    print("\nTraining complete!")

    batch_size = hyp.get("batch", 16)
    imgsz = hyp.get("imgsz", 224)
    hyp_lines = "\n".join(f"  {k}: {v}" for k, v in hyp.items())
    header = (
        f"FINAL TRAINING RESULTS\n"
        f"Model: {model_name}\n"
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
        Path(args.project) / "final_training_results.txt",
        bg_mode=bg_mode,
    )


if __name__ == "__main__":
    main()
