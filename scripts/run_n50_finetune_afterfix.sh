#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/root/project}"
INIT_CKPT="${1:-$PROJECT_DIR/checkpoints/n50_full/full_best_n50.pt}"
OUT_DIR="${2:-$PROJECT_DIR/checkpoints/n50_afterfix_finetune_v1}"
LOG_PATH="${3:-$PROJECT_DIR/train_n50_afterfix_finetune_v1.log}"

cd "$PROJECT_DIR"
mkdir -p "$OUT_DIR"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

nohup python scripts/train_variants.py \
  --variant full \
  --epochs 30 \
  --steps-per-epoch 300 \
  --batch-size 64 \
  --n-nodes 50 \
  --n-pomo 50 \
  --problem-type R \
  --scenario full \
  --tightness 0.7 \
  --lr 3e-5 \
  --entropy-start 0.006 \
  --entropy-end 0.002 \
  --curriculum-start-tightness 0.75 \
  --curriculum-epochs 5 \
  --scheduler-patience 3 \
  --scheduler-factor 0.5 \
  --early-stop-min-epochs 8 \
  --early-stop-patience 6 \
  --ema-decay 0.995 \
  --init-checkpoint "$INIT_CKPT" \
  --val-set-size 128 \
  --val-batch-chunk 4 \
  --val-decode-type greedy \
  --val-sample-rollouts 4 \
  --save-every 5 \
  --output-dir "$OUT_DIR" \
  > "$LOG_PATH" 2>&1 &

echo $! > "$OUT_DIR/train.pid"
echo "Started N50 after-fix finetune."
echo "PID: $(cat "$OUT_DIR/train.pid")"
echo "Log: $LOG_PATH"
echo "Output: $OUT_DIR"
