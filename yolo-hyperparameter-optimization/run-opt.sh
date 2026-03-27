#!/bin/bash
# run_yolo_optimization.sh

# This script runs the complete pipeline for Bayesian optimization of YOLO hyperparameters

# Check if required tools are installed
command -v python3 >/dev/null 2>&1 || { echo "Python 3 is required but not installed. Aborting."; exit 1; }

# Parse command line arguments
DATA=""
MODEL="yolo11n-cls.pt"
EPOCHS=10
FINAL_EPOCHS=30
TRIALS=100
DEVICE="0"
WORKERS=10
BG_MODE="overlay"
INCORRECT_CLASS="incorrect"

while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --data)
      DATA="$2"
      shift
      shift
      ;;
    --model)
      MODEL="$2"
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
    --bg-mode)
      BG_MODE="$2"
      shift
      shift
      ;;
    --incorrect-class)
      INCORRECT_CLASS="$2"
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
  echo "Usage: ./run-opt.sh --data path/to/data.yaml [--model yolo11n-cls.pt] [--epochs 40] [--final-epochs 100] [--trials 300] [--device 0] [--workers 4] [--bg-mode overlay] [--incorrect-class incorrect]"
  exit 1
fi

# Ensure model path is absolute
if [[ "$MODEL" != /* ]] && [[ "$MODEL" != ./* ]]; then
  if [ -f "$MODEL" ]; then
    MODEL="$(pwd)/$MODEL"
    echo "Converted model path to absolute: $MODEL"
  fi
  # Otherwise, assume it's a model name from the YOLO model hub
fi

# Ensure data path is absolute
if [[ "$DATA" != /* ]] && [[ "$DATA" != ./* ]]; then
  DATA="$(pwd)/$DATA"
  echo "Converted data path to absolute: $DATA"
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
echo "Model:             $MODEL"
echo "Epochs per trial:  $EPOCHS"
echo "Final epochs:      $FINAL_EPOCHS"
echo "Number of trials:  $TRIALS"
echo "Device:            $DEVICE"
echo "Workers:           $WORKERS"
echo "Background mode:   optimized (searched during Bayesian opt)"
echo "Incorrect class:   $INCORRECT_CLASS"
echo "Project directory: $(pwd)"
echo "========================================"

# Step 1: Run Bayesian hyperparameter optimization
echo "[1/2] Running Bayesian hyperparameter optimization..."
python3 ../bayesian-opt-yolo.py \
  --data "$DATA" \
  --model "$MODEL" \
  --epochs $EPOCHS \
  --trials $TRIALS \
  --device $DEVICE \
  --workers $WORKERS \
  --incorrect-class "$INCORRECT_CLASS" \
  --project ./optimization_results

# Check if optimization completed successfully
if [ ! -f "./best_hyperparameters.yaml" ]; then
  echo "Error: Bayesian optimization failed or did not produce best hyperparameters."
  exit 1
fi
echo "Optimization completed. Best hyperparameters saved to best_hyperparameters.yaml"

# Extract the best bg_mode from the saved hyperparameters
BG_MODE_BEST=$(python3 -c "import yaml; d=yaml.safe_load(open('best_hyperparameters.yaml')); print(d.get('bg_mode', 'overlay'))")
echo "Best bg_mode from optimization: $BG_MODE_BEST"

# Step 2: Train final model with best hyperparameters
echo "[2/2] Training final model with best hyperparameters..."
python3 ../train-final.py \
  --data "$DATA" \
  --model "$MODEL" \
  --epochs $FINAL_EPOCHS \
  --device "$DEVICE" \
  --bg-mode "$BG_MODE_BEST" \
  --incorrect-class "$INCORRECT_CLASS" \
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
echo "- Final model:                 $FINAL_MODEL"
echo "- Optimization visualizations: ./optimization_history.html"
echo "========================================"
echo "To use the optimized model, run:"
echo "from ultralytics import YOLO"
echo "model = YOLO('$FINAL_MODEL')"
echo "========================================"
