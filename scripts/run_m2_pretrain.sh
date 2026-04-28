#!/bin/bash
# M2: HolterFM pretraining — full execution script for 8×C500
# Step 1: verify model forward/backward
# Step 2: run full pretraining
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

DATA_DIR="${DATA_DIR:-data/DMS}"
VALID_LIST="${VALID_LIST:-valid_records.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/pretrain}"
EPOCHS="${EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LR="${LR:-2e-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
RESUME="${RESUME:-}"

echo "=== M2: HolterFM Pretraining ==="
echo "Data: $DATA_DIR"
echo "Valid list: $VALID_LIST"
echo "Output: $OUTPUT_DIR"
echo "Epochs: $EPOCHS, BS: $BATCH_SIZE, LR: $LR"
echo "GPUs: $(nvidia-smi -L 2>/dev/null | wc -l)"
echo ""

# --- Step 1: Quick sanity check (single GPU, 1 record, forward+backward) ---
echo "=== Step 1: Sanity check ==="
PYTHONPATH="$PROJECT_DIR" python scripts/verify_model.py
echo "Sanity check passed."
echo ""

# --- Step 2: Full pretraining ---
echo "=== Step 2: Full pretraining ==="
mkdir -p "$OUTPUT_DIR"

RESUME_FLAG=""
if [ -n "$RESUME" ]; then
    RESUME_FLAG="--resume $RESUME"
    echo "Resuming from: $RESUME"
fi

PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
torchrun --nproc_per_node=8 --master_port=29500 \
    src/training/pretrain.py \
    --data_dir "$DATA_DIR" \
    --valid_list "$VALID_LIST" \
    --output_dir "$OUTPUT_DIR" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --num_workers "$NUM_WORKERS" \
    $RESUME_FLAG \
    2>&1 | tee "$OUTPUT_DIR/train.log"

echo ""
echo "=== M2 Complete ==="
echo "Checkpoints in: $OUTPUT_DIR"
echo "Best model: $OUTPUT_DIR/holter_fm_best.pt"
