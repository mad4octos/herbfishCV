#!/usr/bin/env python3

# Standard Library imports
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root

# External imports
import optuna
import yaml
from optuna.trial import FrozenTrial
from optuna.trial._trial import Trial
from ultralytics import YOLO

# Local imports
from yolo_dataset import RGBClassificationTrainer
from yolo_tools import evaluate_and_report
from yolo_callbacks import LossPlotCallbacks

# Model architectures to search over
MODEL_CANDIDATES = ["yolo11n-cls.pt", "yolo11s-cls.pt"]

# Fixed augmentation values applied in every trial and final training
FIXED_AUG_PARAMS = {
    "flipud": 0.5,  # ultralytics default is 0.0
    "scale": 0.0,   # ultralytics default is 0.5; avoid downscaling (avg image 111×125 px)
}

# Default hyperparameter ranges - these will be passed directly to the YOLO train method
DEFAULT_HYP_RANGES = {
    "lrf": (0.001, 0.1),  # the fraction of the initial learning rate that the scheduler decays to by the final epoch
    "weight_decay": (0.0001, 0.01),  # Weight decay
    "dropout": (0.0, 0.5),  # Dropout rate
}


def objective(trial: Trial, args):
    """Define the objective function to be optimized."""
    # Sample hyperparameters
    train_args = {}

    # Add the base arguments
    train_args["data"] = args.data
    train_args["epochs"] = args.epochs
    train_args["device"] = args.device
    train_args["project"] = args.project
    train_args["name"] = f"trial_{trial.number}"
    train_args["val"] = True  # Always validate during training
    train_args["deterministic"] = True
    train_args["trainer"] = RGBClassificationTrainer
    train_args["workers"] = args.workers
    train_args["fraction"] = args.fraction

    # The optimizer, learning rate (lr0) and momentum will be auto-selected.
    # The optimizer is chosen between AdamW and SGD based on the number of training iterations.
    train_args["optimizer"] = "auto"

    train_args.update(FIXED_AUG_PARAMS)

    # Ultralytics uses auto_augment=randaugment by default, which includes the following transformations:
    # Identity, ShearX, ShearY, TranslateX, TranslateY, Rotate, Brightness, Color, Contrast, Sharpness, Posterize, 
    # Solarize, AutoContrast, Equalize


    # Add hyperparameters that will be directly passed to train method
    for param_name, param_range in DEFAULT_HYP_RANGES.items():
        if param_name == "lrf":
            # Log-uniform distribution for learning rate
            train_args[param_name] = trial.suggest_float(
                param_name, param_range[0], param_range[1], log=True
            )
        else:
            train_args[param_name] = trial.suggest_float(
                param_name, param_range[0], param_range[1]
            )

    # Add specific parameters you might want to tune
    train_args["batch"] = trial.suggest_categorical(
        "batch", [4, 8, 16, 32, 64, 128, 256]
    )
    train_args["imgsz"] = trial.suggest_categorical("imgsz", [128, 192, 224, 256])

    # Include bg_mode as a categorical hyperparameter in the search
    bg_mode = trial.suggest_categorical("bg_mode", ["gray", "overlay"])
    RGBClassificationTrainer.bg_mode = bg_mode

    try:
        # Sample model architecture as a categorical hyperparameter
        model_name = trial.suggest_categorical("model", MODEL_CANDIDATES)
        model = YOLO(model_name)

        cbs = LossPlotCallbacks(
            figpath=f"{args.project}/trial_{trial.number}/loss_plot.png",
            mode="classification",
            names=["correct", "incorrect"],
        )
        model.add_callback("on_train_epoch_end", cbs.on_train_epoch_end)
        model.add_callback("on_val_batch_end", cbs.on_val_batch_end)
        model.add_callback("on_val_end", cbs.on_val_end)

        print(f"Loading model from: {model_name}")

        # Train the model with the sampled hyperparameters directly passed
        results = model.train(**train_args)

        # RGBClassificationTrainer.fitness returns macro F1
        fitness = results.fitness
        return float(fitness) if fitness is not None else 0.0
    except Exception as e:
        print(f"Training failed with error: {e}")
        return 0.0  # Return a low score so this trial is not selected


def evaluate_best_model(best_trial: FrozenTrial, args):
    """Evaluate the best model on the test split of the training dataset."""
    batch_size = best_trial.params.get("batch", 16)
    imgsz = best_trial.params.get("imgsz", 224)
    bg_mode = best_trial.params.get("bg_mode", "overlay")

    # Path to the best model from the optimization
    best_model_path = (
        Path(args.project) / f"trial_{best_trial.number}" / "weights" / "best.pt"
    )

    # Initialize the model with the best weights
    if best_model_path.exists():
        RGBClassificationTrainer.bg_mode = bg_mode
        model = YOLO(str(best_model_path))

        header = (
            f"BEST TRIAL RESULTS\n"
            f"Trial #: {best_trial.number}\n"
            f"Selection: Highest macro F1 on val split across all Optuna trials\n"
            f"Macro F1 (val) = {best_trial.value:.4f}\n\n"
        )
        return evaluate_and_report(
            model,
            Path(args.data),
            batch_size,
            imgsz,
            args.incorrect_class,
            header,
            Path(args.project) / "best_trial_results.txt",
            bg_mode=bg_mode,
            scale=FIXED_AUG_PARAMS["scale"],
        )
    else:
        print(f"Best model weights not found at {best_model_path}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Bayesian Hyperparameter Optimization for YOLO Classification"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to dataset root directory (with train/val/test subdirectories)",
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs per trial"
    )
    parser.add_argument(
        "--trials", type=int, default=300, help="Number of optimization trials"
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of dataloader workers"
    )
    parser.add_argument("--device", type=str, default="0", help="CUDA device")
    parser.add_argument(
        "--project", type=str, default="runs/bayesian_opt", help="Project directory"
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
        help="Fraction of dataset to use per trial (0.0–1.0). Use a small value (e.g. 0.1) for quick smoke-test runs.",
    )
    args = parser.parse_args()

    print(f"Searching over models: {MODEL_CANDIDATES}")

    project_path = Path(args.project).resolve()
    project_path.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{project_path}/optuna_study.db"

    study = optuna.create_study(
        direction="maximize",
        study_name="yolo_bayesian_optimization",
        sampler=optuna.samplers.TPESampler(seed=42),
        storage=storage,
    )

    print(f"Starting Bayesian Optimization with {args.trials} trials")
    study.optimize(lambda trial: objective(trial, args), n_trials=args.trials)

    # Print optimization results
    print("\nBest trial:")
    best_trial = study.best_trial
    print(f"  Value: {best_trial.value}")
    print("  Params:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    # Save the best hyperparameters
    best_params = best_trial.params
    best_hyp = {
        param: best_params.get(param, DEFAULT_HYP_RANGES[param][0])
        for param in DEFAULT_HYP_RANGES
    }
    best_hyp["batch"] = best_params.get("batch", 16)
    best_hyp["imgsz"] = best_params.get("imgsz", 224)
    best_hyp["bg_mode"] = best_params.get("bg_mode", "overlay")
    best_hyp["model"] = best_params.get("model", MODEL_CANDIDATES[0])
    best_hyp.update(FIXED_AUG_PARAMS)

    with open("best_hyperparameters.yaml", "w") as f:
        yaml.dump(best_hyp, f)

    print("\nBest hyperparameters saved to best_hyperparameters.yaml")

    # Evaluate the best model on the test split of the dataset
    evaluate_best_model(best_trial, args)

    # Generate optimization visualization
    try:
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_html("optimization_history.html")

        fig = optuna.visualization.plot_param_importances(study)
        fig.write_html("param_importances.html")

        fig = optuna.visualization.plot_contour(study)
        fig.write_html("contour_plot.html")

        print("\nOptimization visualizations saved as HTML files.")
    except Exception as e:
        print(f"Error generating visualizations: {e}")


if __name__ == "__main__":
    main()
