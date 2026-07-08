#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-/root/project/checkpoints/n50_mixed_5090_v1}"
LOG_FILE="${2:-/root/project/train_n50_mixed_5090_v1.log}"
BATCH_SIZE="${BATCH_SIZE:-72}"

cd "$(dirname "$0")/.."

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

nohup python scripts/train_variants.py \
  --variant full \
  --epochs 90 \
  --steps-per-epoch 360 \
  --batch-size "$BATCH_SIZE" \
  --n-nodes 50 \
  --n-pomo 50 \
  --problem-type MIXED \
  --val-problem-types R,C,RC,MIXED \
  --scenario full \
  --tightness 0.7 \
  --lr 8e-5 \
  --entropy-start 0.025 \
  --entropy-end 0.004 \
  --curriculum-start-tightness 1.0 \
  --curriculum-epochs 60 \
  --scheduler-patience 6 \
  --scheduler-factor 0.5 \
  --min-lr 1e-5 \
  --early-stop-min-epochs 65 \
  --early-stop-patience 10 \
  --ema-decay 0.997 \
  --pref-mode random \
  --pref-random-low 0.1 \
  --pref-random-high 0.9 \
  --val-pref-mode random \
  --val-set-size 192 \
  --val-batch-chunk 16 \
  --val-decode-type greedy \
  --save-every 5 \
  --fast-cuda \
  --amp \
  --amp-dtype bfloat16 \
  --matmul-precision high \
  --output-dir "$OUT_DIR" \
  > "$LOG_FILE" 2>&1 &

echo $! > "${LOG_FILE%.log}.pid"
echo "Started N50 mixed 5090 training"
echo "pid: $(cat "${LOG_FILE%.log}.pid")"
echo "log: $LOG_FILE"
echo "out: $OUT_DIR"
echo "batch_size: $BATCH_SIZE"
