#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/root/project}"
INIT_CKPT="${1:-$PROJECT_DIR/checkpoints/n50_full/full_best_n50_elite.pt}"
OUT_DIR="${2:-$PROJECT_DIR/checkpoints/n75_bridge_for_n100_v1}"
LOG_PATH="${3:-$PROJECT_DIR/train_n75_bridge_for_n100_v1.log}"

cd "$PROJECT_DIR"
mkdir -p "$OUT_DIR"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

nohup python scripts/train_variants.py \
  --variant full \
  --epochs 25 \
  --steps-per-epoch 600 \
  --batch-size 8 \
  --grad-accum-steps 4 \
  --n-nodes 75 \
  --n-pomo 75 \
  --problem-type MIXED \
  --scenario full \
  --tightness 0.7 \
  --pref-mode random \
  --val-pref-mode random \
  --val-problem-types R,C,RC,MIXED \
  --lr 4e-5 \
  --entropy-start 0.010 \
  --entropy-end 0.003 \
  --curriculum-start-tightness 0.85 \
  --curriculum-epochs 10 \
  --scheduler-patience 5 \
  --scheduler-factor 0.5 \
  --early-stop-min-epochs 25 \
  --early-stop-patience 0 \
  --ema-decay 0.997 \
  --init-checkpoint "$INIT_CKPT" \
  --val-set-size 256 \
  --val-batch-chunk 1 \
  --val-decode-type greedy \
  --elite-fraction 0.10 \
  --elite-loss-weight 0.20 \
  --fast-cuda \
  --amp \
  --amp-dtype bfloat16 \
  --matmul-precision high \
  --save-every 5 \
  --output-dir "$OUT_DIR" \
  > "$LOG_PATH" 2>&1 &

echo $! > "$OUT_DIR/train.pid"
echo "Started N75 bridge training for N100."
echo "PID: $(cat "$OUT_DIR/train.pid")"
echo "Log: $LOG_PATH"
echo "Output: $OUT_DIR"
