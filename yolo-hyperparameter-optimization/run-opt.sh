#!/bin/bash
# run_yolo_optimization.sh

# This script runs the complete pipeline for Bayesian optimization of YOLO hyperparameters

# Check if required tools are installed
command -v python3 >/dev/null 2>&1 || { echo "Python 3 is required but not installed. Aborting."; exit 1; }

# Parse command line arguments
DATA=""
EPOCHS=10
FINAL_EPOCHS=30
TRIALS=100
DEVICE="0"
WORKERS=10
INCORRECT_CLASS="incorrect"
FRACTION=1.0

while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --data)
      DATA="$2"
      shift
      shift
      ;;
    --epochs)
      EPOCHS="$2"
      shift
      shift
      ;;
    --final-epochs)
      FINAL_EPOCHS="$2"
      shift
      shift
      ;;
    --trials)
      TRIALS="$2"
      shift
      shift
      ;;
    --device)
      DEVICE="$2"
      shift
      shift
      ;;
    --workers)
      WORKERS="$2"
      shift
      shift
      ;;
    --incorrect-class)
      INCORRECT_CLASS="$2"
      shift
      shift
      ;;
    --fraction)
      FRACTION="$2"
      shift
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Check if data argument is provided
if [ -z "$DATA" ]; then
  echo "Error: --data argument is required."
  echo "Usage: ./run-opt.sh --data path/to/data [--epochs 10] [--final-epochs 30] [--trials 100] [--device 0] [--workers 10] [--incorrect-class incorrect] [--fraction 1.0]"
  exit 1
fi

# Ensure data path is absolute
if [[ "$DATA" != /* ]] && [[ "$DATA" != ./* ]]; then
  DATA="$(pwd)/$DATA"
  echo "Converted data path to absolute: $DATA"
fi

# Copy data to local SSD (Approximately 300 GB/node)
# https://curc.readthedocs.io/en/latest/compute/filesystems.html#local-scratch-on-alpine-and-blanca
if [[ -n "${SLURM_SCRATCH}" ]]; then
  cp -r "$DATA" "$SLURM_SCRATCH"
  DATA="$SLURM_SCRATCH/$(basename "$DATA")"
  echo "Using local SSD copy: $DATA"
fi

# Create project directories
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
PROJECT_DIR="yolo_optimization_${TIMESTAMP}"
mkdir -p $PROJECT_DIR
cd $PROJECT_DIR

echo "========================================"
echo "YOLO Classification Optimization Pipeline"
echo "========================================"
echo "Data:              $DATA"
echo "Models searched:   yolo11n-cls.pt, yolo11s-cls.pt"
echo "Epochs per trial:  $EPOCHS"
echo "Final epochs:      $FINAL_EPOCHS"
echo "Number of trials:  $TRIALS"
echo "Device:            $DEVICE"
echo "Workers:           $WORKERS"
echo "Background mode:   optimized (searched during Bayesian opt)"
echo "Incorrect class:   $INCORRECT_CLASS"
echo "Fraction:          $FRACTION"
echo "Project directory: $(pwd)"
echo "========================================"

# Step 1: Run Bayesian hyperparameter optimization
echo "[1/2] Running Bayesian hyperparameter optimization..."
python3 ../bayesian-opt-yolo.py \
  --data "$DATA" \
  --epochs $EPOCHS \
  --trials $TRIALS \
  --device $DEVICE \
  --workers $WORKERS \
  --incorrect-class "$INCORRECT_CLASS" \
  --fraction $FRACTION \
  --project ./trials

# Check if optimization completed successfully
if [ ! -f "./best_hyperparameters.yaml" ]; then
  echo "Error: Bayesian optimization failed or did not produce best hyperparameters."
  exit 1
fi
echo "Optimization completed. Best hyperparameters saved to best_hyperparameters.yaml"

# Step 2: Train final model with best hyperparameters
echo "[2/2] Training final model with best hyperparameters..."
python3 ../train-final.py \
  --data "$DATA" \
  --epochs $FINAL_EPOCHS \
  --device "$DEVICE" \
  --incorrect-class "$INCORRECT_CLASS" \
  --fraction $FRACTION \
  --hyp ./best_hyperparameters.yaml \
  --project ./final_model

# Find the final model
FINAL_MODEL=$(find ./final_model -name "best.pt" | sort -n | tail -1)
if [ -z "$FINAL_MODEL" ]; then
  echo "Error: Could not find final model weights."
  exit 1
fi
echo "Final model trained: $FINAL_MODEL"

echo "========================================"
echo "Optimization pipeline completed!"
echo "========================================"
echo "Results summary:"
echo "- Best hyperparameters:        ./best_hyperparameters.yaml"
echo "- Best trial evaluation:       ./best_trial_results.txt"
echo "- Best trial threshold plot:   ./best_trial_results.threshold_plot.png"
echo "- Final model:                 $FINAL_MODEL"
echo "- Final model evaluation:      ./final_training_results.txt"
echo "- Final model threshold plot:  ./final_training_results.threshold_plot.png"
echo "- Optimization history:        ./optimization_history.html"
echo "- Parameter importances:       ./param_importances.html"
echo "- Contour plot:                ./contour_plot.html"
echo "========================================"
echo "To use the optimized model, run:"
echo "from ultralytics import YOLO"
echo "model = YOLO('$FINAL_MODEL')"
echo "========================================"
