#!/bin/bash
# Run the SwinCVS Multi-Head + Optimal Labels experiment on RunPod
set -e

export DATASET_DIR="${DATASET_DIR:-/workspace}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export SWINCVS_AUTO=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "SwinCVS Multi-Head + Optimal Labels Experiment"
echo "  Config: config/SwinCVS_config_runpod.yaml"
echo "  Dataset: $DATASET_DIR/endoscapes/"
echo "  Workers: $NUM_WORKERS"
echo "============================================================"

nvidia-smi

python run_multihead_experiment.py \
    --config config/SwinCVS_config_runpod.yaml \
    --dataset_dir "$DATASET_DIR" \
    2>&1 | tee experiment_outputs/run_log_runpod.txt
