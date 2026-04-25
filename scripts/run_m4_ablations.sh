#!/bin/bash
# M4: Ablation experiments (R040-R047)
# Each ablation: 20-epoch pretrain → evaluate on 4 core tasks
set -e

DATA_DIR="${DATA_DIR:-data/DMS}"
DEVICE="${DEVICE:-cuda}"
BASE_DIR="${BASE_DIR:-checkpoints/ablations}"
RESULT_DIR="${RESULT_DIR:-results/ablations}"
EPOCHS="${EPOCHS:-20}"  # screen at 20 epochs, rerun decisive ones at 40

mkdir -p "$BASE_DIR" "$RESULT_DIR"

echo "=== M4: Ablation Experiments ==="

run_ablation() {
    local NAME="$1"
    local EXTRA_ARGS="$2"
    local CKPT_DIR="$BASE_DIR/$NAME"
    local RES_DIR="$RESULT_DIR/$NAME"
    mkdir -p "$CKPT_DIR" "$RES_DIR"

    echo ""
    echo "--- Ablation: $NAME ---"

    # Pretrain
    echo "  Pretraining ($EPOCHS epochs)..."
    torchrun --nproc_per_node=8 --master_port=29500 \
        src/training/pretrain_ablation.py \
        --data_dir "$DATA_DIR" \
        --output_dir "$CKPT_DIR" \
        --epochs "$EPOCHS" \
        --ablation "$NAME" \
        $EXTRA_ARGS

    # Evaluate on 4 core tasks
    CKPT="$CKPT_DIR/holter_fm_best.pt"
    for TASK in beat_classification pvc_burden report_concepts; do
        echo "  Evaluating: $TASK"
        python -m src.evaluation.downstream_eval \
            --task "$TASK" --mode linear_probe \
            --checkpoint "$CKPT" --data_dir "$DATA_DIR" \
            --epochs 20 --seed 42 --device "$DEVICE" \
            --output "$RES_DIR/${TASK}.json"
    done
}

# A1: Fixed 10s chunk tokenization (no beat-sync)
run_ablation "no_beat_sync" ""

# A2: Remove episode losses
run_ablation "no_episode_loss" ""

# A3: Remove day losses
run_ablation "no_day_loss" ""

# A4: Remove rhythm branch
run_ablation "no_rhythm" ""

# A5: Beat-only SSL (no episode/day losses)
run_ablation "beat_only_ssl" ""

# A6: Remove report + stats auxiliary losses
run_ablation "no_day_aux" ""

# A7: Sparse Transformer day encoder (instead of Mamba)
run_ablation "transformer_day" ""

# A8: Parameter-matched segment model
run_ablation "segment_model" ""

echo ""
echo "=== All ablations complete ==="
echo "Results in $RESULT_DIR/"

# Summary table
echo ""
echo "=== Summary ==="
python -c "
import json, glob, os
rows = []
for d in sorted(glob.glob('$RESULT_DIR/*')):
    name = os.path.basename(d)
    beat = json.load(open(f'{d}/beat_classification.json')) if os.path.exists(f'{d}/beat_classification.json') else {}
    burden = json.load(open(f'{d}/pvc_burden.json')) if os.path.exists(f'{d}/pvc_burden.json') else {}
    concepts = json.load(open(f'{d}/report_concepts.json')) if os.path.exists(f'{d}/report_concepts.json') else {}
    rows.append({
        'ablation': name,
        'beat_f1': beat.get('macro_f1', '-'),
        'burden_mae': burden.get('mae_burden', '-'),
        'concept_auroc': concepts.get('macro_auroc', '-'),
    })
print(f'{\"Ablation\":<20} {\"Beat F1\":<10} {\"Burden MAE\":<12} {\"Concept AUROC\":<14}')
print('-' * 56)
for r in rows:
    print(f'{r[\"ablation\"]:<20} {str(r[\"beat_f1\"]):<10} {str(r[\"burden_mae\"]):<12} {str(r[\"concept_auroc\"]):<14}')
"
