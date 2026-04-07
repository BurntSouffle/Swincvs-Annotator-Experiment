#!/bin/bash
# RunPod setup script for SwinCVS Multi-Head + Optimal Labels Experiment
# ======================================================================
#
# Expected RunPod volume layout:
#   /workspace/
#     endoscapes/          <- upload your endoscapes dataset here
#       all/               <- 58,586 images + annotation_coco.json
#       train/             <- annotation_ds_coco.json + images
#       val/
#       test/
#     SwinCVS/             <- this repo (cloned from GitHub)
#
# Usage:
#   1. Clone repo:       cd /workspace && git clone <your-repo-url> SwinCVS
#   2. Upload dataset:   (rsync/scp endoscapes/ to /workspace/endoscapes/)
#   3. Run this script:  cd /workspace/SwinCVS && bash setup_runpod.sh
#   4. Run experiment:   bash run_experiment.sh
# ======================================================================

set -e

echo "============================================================"
echo "SwinCVS RunPod Setup"
echo "============================================================"

# Environment
export DATASET_DIR="/workspace"
export NUM_WORKERS=4
export SWINCVS_AUTO=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# 1. Install dependencies (RunPod images typically have torch pre-installed)
echo ""
echo "[1/4] Installing Python dependencies..."
pip install -r requirements.txt

# 2. Verify dataset exists
echo ""
echo "[2/4] Checking dataset..."
if [ ! -d "/workspace/endoscapes/all" ]; then
    echo "ERROR: /workspace/endoscapes/all/ not found."
    echo "Upload the endoscapes dataset to /workspace/endoscapes/ first."
    exit 1
fi

FRAME_COUNT=$(ls /workspace/endoscapes/all/*.jpg 2>/dev/null | wc -l)
echo "  Found $FRAME_COUNT frames in endoscapes/all/"

for SPLIT in train val test; do
    if [ ! -f "/workspace/endoscapes/$SPLIT/annotation_ds_coco.json" ]; then
        echo "  ERROR: Missing /workspace/endoscapes/$SPLIT/annotation_ds_coco.json"
        exit 1
    fi
    echo "  $SPLIT/annotation_ds_coco.json OK"
done

# 3. Verify/download weights
echo ""
echo "[3/4] Checking weights..."
mkdir -p weights

REQUIRED_WEIGHTS=(
    "swinv2_base_patch4_window12to24_192to384_22kto1k_ft.pth"
    "SwinCVS_E2E_MC_IMNP_sd5_bestMAP.pt"
)

ALL_PRESENT=true
for W in "${REQUIRED_WEIGHTS[@]}"; do
    if [ -f "weights/$W" ]; then
        echo "  $W OK"
    else
        echo "  MISSING: weights/$W"
        ALL_PRESENT=false
    fi
done

if [ "$ALL_PRESENT" = false ]; then
    echo ""
    echo "  Some weights are missing. The script will attempt to download them"
    echo "  from the SharePoint URL when you run the experiment."
    echo "  Alternatively, upload them manually to $SCRIPT_DIR/weights/"
fi

# 4. Generate optimal labels if not present
echo ""
echo "[4/4] Checking optimal labels..."
if [ -f "/workspace/endoscapes/train/annotation_ds_coco_optimal.json" ]; then
    echo "  Optimal labels already exist. Skipping."
else
    echo "  Generating optimal labels..."
    python create_optimal_labels.py
fi

echo ""
echo "============================================================"
echo "Setup complete!"
echo ""
echo "To run the experiment:"
echo "  bash run_experiment.sh"
echo ""
echo "Or manually:"
echo "  export DATASET_DIR=/workspace"
echo "  export NUM_WORKERS=4"
echo "  python run_multihead_experiment.py --config config/SwinCVS_config_runpod.yaml"
echo "============================================================"
