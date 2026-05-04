#!/bin/bash
# M3: Run all downstream evaluations (R030-R034)
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1
export MACA_PATH="${MACA_PATH:-/opt/maca}"
export PYTHONPATH="$PROJECT_DIR"

CHECKPOINT="${CHECKPOINT:-checkpoints/pretrain/holter_fm_best.pt}"
DATA_DIR="${DATA_DIR:-data/DMS}"
DEVICE="${DEVICE:-cuda}"
OUT_DIR="${OUT_DIR:-results/downstream}"

mkdir -p "$OUT_DIR"

echo "=== M3: Downstream Evaluation ==="
echo "Checkpoint: $CHECKPOINT"
echo ""

# R030: Beat classification — linear probe (3 seeds)
for SEED in 42 123 456; do
    echo "--- R030: Beat classification (linear_probe, seed=$SEED) ---"
    python -m src.evaluation.downstream_eval \
        --task beat_classification --mode linear_probe \
        --checkpoint "$CHECKPOINT" --data_dir "$DATA_DIR" \
        --epochs 20 --lr 1e-3 --seed $SEED --device "$DEVICE" \
        --output "$OUT_DIR/beat_lp_s${SEED}.json"
done

# R031: Beat classification — fine-tune (3 seeds)
for SEED in 42 123 456; do
    echo "--- R031: Beat classification (fine_tune, seed=$SEED) ---"
    python -m src.evaluation.downstream_eval \
        --task beat_classification --mode fine_tune \
        --checkpoint "$CHECKPOINT" --data_dir "$DATA_DIR" \
        --epochs 20 --lr 1e-3 --seed $SEED --device "$DEVICE" \
        --output "$OUT_DIR/beat_ft_s${SEED}.json"
done

# R032: PVC burden regression
echo "--- R032: PVC burden regression ---"
python -m src.evaluation.downstream_eval \
    --task pvc_burden --mode frozen \
    --checkpoint "$CHECKPOINT" --data_dir "$DATA_DIR" \
    --epochs 100 --lr 1e-3 --seed 42 --device "$DEVICE" \
    --output "$OUT_DIR/burden.json"

# R033: Report concept prediction
echo "--- R033: Report concept prediction ---"
python -m src.evaluation.downstream_eval \
    --task report_concepts --mode frozen \
    --checkpoint "$CHECKPOINT" --data_dir "$DATA_DIR" \
    --epochs 30 --lr 5e-4 --seed 42 --device "$DEVICE" \
    --output "$OUT_DIR/concepts.json"

echo ""
echo "=== All downstream evaluations complete ==="
echo "Results in $OUT_DIR/"
