#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/root/project}"
INIT_CKPT="${1:-}"
OUT_DIR="${2:-$PROJECT_DIR/checkpoints/n100_smoke}"
LOG_PATH="${3:-$PROJECT_DIR/train_n100_smoke.log}"

cd "$PROJECT_DIR"
mkdir -p "$OUT_DIR"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CHECK_ARGS=()
INIT_ARGS=()
if [[ -n "$INIT_CKPT" ]]; then
  CHECK_ARGS=(--checkpoint "$INIT_CKPT")
  INIT_ARGS=(--init-checkpoint "$INIT_CKPT")
fi

python scripts/check_n100_training_ready.py \
  --n-nodes 100 \
  --n-pomo 8 \
  --batch-size 1 \
  --problem-type MIXED \
  --pref-mode random \
  "${CHECK_ARGS[@]}"

python scripts/train_variants.py \
  --variant full \
  --epochs 1 \
  --steps-per-epoch 5 \
  --batch-size 1 \
  --grad-accum-steps 1 \
  --n-nodes 100 \
  --n-pomo 8 \
  --problem-type MIXED \
  --scenario full \
  --tightness 0.7 \
  --pref-mode random \
  --lr 1e-5 \
  --entropy-start 0.004 \
  --entropy-end 0.004 \
  --curriculum-start-tightness 0.7 \
  --curriculum-epochs 1 \
  --early-stop-min-epochs 1 \
  --early-stop-patience 0 \
  --ema-decay 0.997 \
  --val-set-size 4 \
  --val-batch-chunk 1 \
  --val-pref-mode random \
  --val-problem-types R,C,RC,MIXED \
  --val-decode-type greedy \
  --save-every 1 \
  "${INIT_ARGS[@]}" \
  --output-dir "$OUT_DIR" \
  > "$LOG_PATH" 2>&1

echo "N100 smoke finished."
echo "Log: $LOG_PATH"
echo "Output: $OUT_DIR"
