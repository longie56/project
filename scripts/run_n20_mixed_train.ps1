param(
    [string]$OutputDir = ".\checkpoints\n20_mixed_full_v1",
    [int]$Epochs = 50,
    [int]$StepsPerEpoch = 200,
    [int]$BatchSize = 64,
    [int]$ValSetSize = 300,
    [int]$Seed = 42
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

python .\scripts\train_variants.py `
  --variant full `
  --epochs $Epochs `
  --steps-per-epoch $StepsPerEpoch `
  --batch-size $BatchSize `
  --n-nodes 20 `
  --n-pomo 20 `
  --problem-type MIXED `
  --scenario full `
  --tightness 0.7 `
  --lr 1e-4 `
  --entropy-start 0.02 `
  --entropy-end 0.002 `
  --curriculum-start-tightness 1.0 `
  --curriculum-epochs 20 `
  --scheduler-patience 6 `
  --scheduler-factor 0.5 `
  --early-stop-min-epochs 25 `
  --early-stop-patience 10 `
  --ema-decay 0.995 `
  --val-set-size $ValSetSize `
  --val-batch-chunk 16 `
  --val-decode-type greedy `
  --val-sample-rollouts 0 `
  --val-multi-start-rollouts 0 `
  --save-every 10 `
  --seed $Seed `
  --output-dir $OutputDir
