#!/bin/bash
# M2: HolterFM pretraining (R020)
# Run on 8×A100 80GB
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
echo ""

RESUME_FLAG=""
if [ -n "$RESUME" ]; then
    RESUME_FLAG="--resume $RESUME"
    echo "Resuming from: $RESUME"
fi

# Single-node multi-GPU via torchrun
PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --nproc_per_node=8 --master_port=29500 \
    src/training/pretrain.py \
    --data_dir "$DATA_DIR" \
    --valid_list "$VALID_LIST" \
    --output_dir "$OUTPUT_DIR" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --num_workers "$NUM_WORKERS" \
    $RESUME_FLAG
