# YOLO Hyperparameter Optimization Framework

A comprehensive pipeline for automatically optimizing YOLO model hyperparameters using Bayesian optimization.

## Overview

This framework provides an end-to-end solution for finding optimal hyperparameters for YOLO classification models. It leverages Bayesian optimization (via Optuna) to efficiently search the hyperparameter space and trains a final model with the best parameters, optimizing for macro F1 score on the validation split.

## Components

- `run-opt.sh`: Main entry script that orchestrates the entire optimization pipeline
- `bayesian-opt-yolo.py`: Implements Bayesian optimization to find optimal hyperparameters
- `train-final.py`: Trains the final model using the best discovered hyperparameters from all runs
- `dashboard1.py`: Generates an HTML dashboard from trial results — **not yet updated for this classification task**

## Requirements

- Python 3.8+
- Required Python packages:
  - ultralytics
  - optuna
  - numpy
  - matplotlib
  - opencv-python
  - pyyaml
  - tqdm

## Usage

```bash
./run-opt.sh --data PATH_TO_DATA [options]
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--data` | *(required)* | Path to dataset root directory (with `train/val/test` subdirectories) |
| `--model` | `yolo11n-cls.pt` | Initial YOLO classification model path or hub name |
| `--epochs` | `10` | Number of epochs per optimization trial |
| `--final-epochs` | `30` | Number of epochs for final model training |
| `--trials` | `100` | Number of Optuna optimization trials |
| `--device` | `0` | CUDA device index |
| `--workers` | `10` | Number of dataloader workers |
| `--incorrect-class` | `incorrect` | Name of the positive class used for threshold search |

### Example

```bash
./run-opt.sh \
  --data ../datasets/fish_classification \
  --model yolo11n-cls.pt \
  --epochs 10 \
  --final-epochs 30 \
  --trials 100 \
  --device 0 \
  --incorrect-class incorrect
```

## Optimization Process

1. **Bayesian Optimization**: Systematically explores hyperparameter combinations, maximizing macro F1 on the validation split across all trials.
2. **Final Training**: Trains a model from scratch using the best discovered hyperparameters for an extended number of epochs.

## Hyperparameters Optimized

| Parameter | Type | Range / Options |
|---|---|---|
| `lrf` | log-uniform float | [0.001, 0.1] |
| `weight_decay` | float | [0.0001, 0.01] |
| `dropout` | float | [0.0, 0.5] |
| `batch` | categorical | 4, 8, 16, 32, 64, 128, 256 |
| `imgsz` | categorical | 128, 192, 224, 256 |
| `bg_mode` | categorical | `gray`, `overlay` |

The following training settings are fixed (not tuned): `optimizer=auto` (AdamW or SGD selected automatically based on training iterations), `flipud=0.5`, `scale=0.0`, and RandAugment via ultralytics' default `auto_augment`.

## Output

The optimization process creates a timestamped directory (`yolo_optimization_<TIMESTAMP>/`) containing:

- `best_hyperparameters.yaml`: Best hyperparameter values discovered
- `optimization_results/`: Training results for each Optuna trial, including `best_trial_results.txt` with the evaluation report on the test split
- `final_model/`: Final model trained with the best hyperparameters
- Optuna visualization plots (HTML):
  - `optimization_history.html`
  - `param_importances.html`
  - `contour_plot.html`

## Using the Optimized Model

After optimization, you can use the final model in your Python code:

```python
from ultralytics import YOLO
model = YOLO('/path/to/final_model/best.pt')
results = model.predict('path/to/image.jpg')
```
