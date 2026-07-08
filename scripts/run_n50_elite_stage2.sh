#!/usr/bin/env bash
set -euo pipefail

CKPT="${1:-/root/project/checkpoints/n50_full/full_best_n50_elite.pt}"
OUT_DIR="${2:-/root/project/checkpoints/n50_elite_stage2_v1}"
LOG_FILE="${3:-/root/project/train_n50_elite_stage2_v1.log}"

cd "$(dirname "$0")/.."

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

nohup python scripts/train_variants.py \
  --variant full \
  --epochs 35 \
  --steps-per-epoch 400 \
  --batch-size 32 \
  --n-nodes 50 \
  --n-pomo 50 \
  --problem-type R \
  --scenario full \
  --tightness 0.7 \
  --lr 2e-5 \
  --entropy-start 0.004 \
  --entropy-end 0.001 \
  --curriculum-start-tightness 0.7 \
  --curriculum-epochs 1 \
  --scheduler-patience 4 \
  --scheduler-factor 0.5 \
  --early-stop-min-epochs 8 \
  --early-stop-patience 8 \
  --ema-decay 0.997 \
  --init-checkpoint "$CKPT" \
  --val-set-size 256 \
  --val-batch-chunk 2 \
  --val-decode-type greedy \
  --val-sample-rollouts 0 \
  --elite-fraction 0.15 \
  --elite-loss-weight 0.65 \
  --save-every 5 \
  --output-dir "$OUT_DIR" \
  > "$LOG_FILE" 2>&1 &

echo $! > "${LOG_FILE%.log}.pid"
echo "Started N50 elite stage-2 fine-tune"
echo "pid: $(cat "${LOG_FILE%.log}.pid")"
echo "log: $LOG_FILE"
echo "out: $OUT_DIR"
