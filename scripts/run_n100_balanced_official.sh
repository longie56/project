#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/root/project}"
INIT_CKPT="${1:-}"
OUT_DIR="${2:-$PROJECT_DIR/checkpoints/n100_full_balanced_v1}"
LOG_PATH="${3:-$PROJECT_DIR/train_n100_full_balanced_v1.log}"

cd "$PROJECT_DIR"
mkdir -p "$OUT_DIR"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

INIT_ARGS=()
if [[ -n "$INIT_CKPT" ]]; then
  INIT_ARGS=(--init-checkpoint "$INIT_CKPT")
fi

nohup python scripts/train_variants.py \
  --variant full \
  --epochs 100 \
  --steps-per-epoch 1000 \
  --batch-size 4 \
  --grad-accum-steps 8 \
  --n-nodes 100 \
  --n-pomo 100 \
  --problem-type MIXED \
  --scenario full \
  --tightness 0.7 \
  --pref-mode fixed \
  --pref-alpha 0.5 \
  --pref-beta 0.5 \
  --val-pref-mode fixed \
  --val-pref-alpha 0.5 \
  --val-pref-beta 0.5 \
  --lr 4e-5 \
  --entropy-start 0.012 \
  --entropy-end 0.002 \
  --curriculum-start-tightness 1.0 \
  --curriculum-epochs 35 \
  --scheduler-patience 10 \
  --scheduler-factor 0.5 \
  --early-stop-min-epochs 100 \
  --early-stop-patience 0 \
  --ema-decay 0.997 \
  --val-set-size 512 \
  --val-batch-chunk 1 \
  --val-decode-type greedy \
  --val-sample-rollouts 0 \
  --val-multi-start-rollouts 0 \
  --elite-fraction 0.10 \
  --elite-loss-weight 0.20 \
  --save-every 5 \
  "${INIT_ARGS[@]}" \
  --output-dir "$OUT_DIR" \
  > "$LOG_PATH" 2>&1 &

echo $! > "$OUT_DIR/train.pid"
echo "Started N100 balanced fixed-preference training."
echo "PID: $(cat "$OUT_DIR/train.pid")"
echo "Log: $LOG_PATH"
echo "Output: $OUT_DIR"
