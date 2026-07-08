import argparse
import json
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    CURRICULUM_EPOCHS,
    CURRICULUM_START_TIGHTNESS,
    EARLY_STOPPING_PATIENCE,
    EARLY_STOPPING_MIN_EPOCHS,
    EMA_DECAY,
    ENTROPY_COEF,
    ENTROPY_COEF_END,
    EPOCHS,
    LR,
    LR_SCHEDULER_FACTOR,
    LR_SCHEDULER_PATIENCE,
    MIN_LR,
    N_NODES,
    N_POMO,
    NORMALIZE_ADVANTAGE,
    STEPS_PER_EPOCH,
    USE_EMA_EVAL,
    VAL_BATCH_CHUNK,
    VAL_SET_SIZE,
)
from model.pomo import PC_EGA_POMO
from model.variants import VARIANTS, get_variant_names
from seed import set_global_seed
from train.trainer import Trainer
from utils.visualization import plot_convergence, plot_training_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=list(VARIANTS.keys()) + ["all"], default="all")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--steps-per-epoch", type=int, default=STEPS_PER_EPOCH)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--n-nodes", type=int, default=N_NODES)
    parser.add_argument("--n-pomo", type=int, default=None)
    parser.add_argument("--problem-type", default="R")
    parser.add_argument("--scenario", choices=["full", "simple"], default="full")
    parser.add_argument("--tightness", type=float, default=0.7)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--scheduler-patience", type=int, default=LR_SCHEDULER_PATIENCE)
    parser.add_argument("--scheduler-factor", type=float, default=LR_SCHEDULER_FACTOR)
    parser.add_argument("--min-lr", type=float, default=MIN_LR)
    parser.add_argument("--entropy-start", type=float, default=ENTROPY_COEF)
    parser.add_argument("--entropy-end", type=float, default=ENTROPY_COEF_END)
    parser.add_argument("--curriculum-start-tightness", type=float, default=CURRICULUM_START_TIGHTNESS)
    parser.add_argument("--curriculum-epochs", type=int, default=CURRICULUM_EPOCHS)
    parser.add_argument("--early-stop-patience", type=int, default=EARLY_STOPPING_PATIENCE)
    parser.add_argument("--early-stop-min-epochs", type=int, default=EARLY_STOPPING_MIN_EPOCHS)
    parser.add_argument("--ema-decay", type=float, default=EMA_DECAY)
    parser.add_argument("--disable-ema-eval", action="store_true")
    parser.add_argument("--disable-adv-normalization", action="store_true")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--val-set-size", type=int, default=VAL_SET_SIZE)
    parser.add_argument("--val-batch-chunk", type=int, default=VAL_BATCH_CHUNK)
    parser.add_argument("--val-decode-type", choices=["greedy", "sample"], default="greedy")
    parser.add_argument("--val-sample-rollouts", type=int, default=0)
    parser.add_argument("--val-multi-start-rollouts", type=int, default=0)
    parser.add_argument("--val-augment-8-fold", action="store_true")
    parser.add_argument("--elite-fraction", type=float, default=0.0)
    parser.add_argument("--elite-loss-weight", type=float, default=0.0)
    parser.add_argument("--output-dir", default=CHECKPOINT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", default="")
    parser.add_argument(
        "--init-checkpoint",
        default="",
        help="Load model weights only before training; optimizer/scheduler/epoch history are reset.",
    )
    parser.add_argument("--bc-teacher", default="", help="Optional filtered teacher dataset for weak BC regularization.")
    parser.add_argument("--bc-batch-size", type=int, default=0)
    parser.add_argument("--bc-weight-start", type=float, default=0.0)
    parser.add_argument("--bc-weight-end", type=float, default=0.0)
    parser.add_argument("--bc-epochs", type=int, default=0)
    parser.add_argument("--pref-mode", choices=["random", "fixed"], default="random")
    parser.add_argument("--pref-alpha", type=float, default=0.5)
    parser.add_argument("--pref-beta", type=float, default=0.5)
    parser.add_argument("--pref-random-low", type=float, default=0.1)
    parser.add_argument("--pref-random-high", type=float, default=0.9)
    parser.add_argument("--val-pref-mode", choices=["random", "fixed"], default=None)
    parser.add_argument("--val-pref-alpha", type=float, default=None)
    parser.add_argument("--val-pref-beta", type=float, default=None)
    parser.add_argument(
        "--val-problem-types",
        default="",
        help="Optional comma-separated validation sets, e.g. R,C,RC,MIXED. If empty, use --problem-type only.",
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="Accumulate gradients across this many mini-batches before optimizer.step().",
    )
    parser.add_argument(
        "--fast-cuda",
        action="store_true",
        help="Enable non-deterministic cuDNN benchmarking and TF32-friendly settings for faster cloud training.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use CUDA autocast during training/validation to reduce memory and improve Tensor Core throughput.",
    )
    parser.add_argument(
        "--amp-dtype",
        choices=["bfloat16", "float16"],
        default="bfloat16",
        help="Autocast dtype. bfloat16 is recommended on RTX 50-series for stability.",
    )
    parser.add_argument(
        "--matmul-precision",
        choices=["highest", "high", "medium"],
        default="high",
        help="torch.set_float32_matmul_precision setting used when available.",
    )
    args = parser.parse_args()
    if args.n_pomo is None:
        args.n_pomo = args.n_nodes

    set_global_seed(args.seed)
    if torch.cuda.is_available():
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision(args.matmul_precision)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if args.fast_cuda:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
            print(
                "[perf] fast CUDA enabled: cudnn.benchmark=True, "
                "cudnn.deterministic=False, TF32 allowed"
            )
        print(
            f"[perf] device={torch.cuda.get_device_name(0)}, "
            f"matmul_precision={args.matmul_precision}"
        )
    os.makedirs(args.output_dir, exist_ok=True)

    summary = {}
    for variant_name in get_variant_names(args.variant):
        model = PC_EGA_POMO(**VARIANTS[variant_name])
        if args.init_checkpoint:
            init_path = args.init_checkpoint.format(variant=variant_name)
            if os.path.exists(init_path):
                try:
                    ckpt = torch.load(init_path, map_location="cpu", weights_only=True)
                except TypeError:
                    ckpt = torch.load(init_path, map_location="cpu")
                state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
                missing, unexpected = model.load_state_dict(state, strict=False)
                print(
                    f"[{variant_name}] initialized weights from {init_path} "
                    f"(missing={len(missing)}, unexpected={len(unexpected)})"
                )
            else:
                raise FileNotFoundError(f"--init-checkpoint not found: {init_path}")
        trainer = Trainer(
            model=model,
            variant_name=variant_name,
            lr=args.lr,
            batch_size=args.batch_size,
            n_nodes=args.n_nodes,
            n_pomo=args.n_pomo,
            steps_per_epoch=args.steps_per_epoch,
            problem_type=args.problem_type,
            scenario=args.scenario,
            tightness=args.tightness,
            save_dir=args.output_dir,
            val_set_size=args.val_set_size,
            val_batch_chunk=args.val_batch_chunk,
            scheduler_patience=args.scheduler_patience,
            scheduler_factor=args.scheduler_factor,
            min_lr=args.min_lr,
            entropy_coef_start=args.entropy_start,
            entropy_coef_end=args.entropy_end,
            curriculum_start_tightness=args.curriculum_start_tightness,
            curriculum_epochs=args.curriculum_epochs,
            early_stopping_patience=args.early_stop_patience,
            early_stopping_min_epochs=args.early_stop_min_epochs,
            ema_decay=args.ema_decay,
            use_ema_eval=(USE_EMA_EVAL and (not args.disable_ema_eval)),
            normalize_advantage=(NORMALIZE_ADVANTAGE and (not args.disable_adv_normalization)),
            val_decode_type=args.val_decode_type,
            val_sample_rollouts=args.val_sample_rollouts,
            val_multi_start_rollouts=args.val_multi_start_rollouts,
            val_augment_8_fold=args.val_augment_8_fold,
            elite_fraction=args.elite_fraction,
            elite_loss_weight=args.elite_loss_weight,
            bc_teacher_path=args.bc_teacher,
            bc_batch_size=args.bc_batch_size,
            bc_weight_start=args.bc_weight_start,
            bc_weight_end=args.bc_weight_end,
            bc_epochs=args.bc_epochs,
            pref_mode=args.pref_mode,
            pref_alpha=args.pref_alpha,
            pref_beta=args.pref_beta,
            val_pref_mode=args.val_pref_mode,
            val_pref_alpha=args.val_pref_alpha,
            val_pref_beta=args.val_pref_beta,
            val_problem_types=args.val_problem_types,
            grad_accum_steps=args.grad_accum_steps,
            pref_random_low=args.pref_random_low,
            pref_random_high=args.pref_random_high,
            amp=args.amp,
            amp_dtype=args.amp_dtype,
        )

        resumed_epoch = 0
        if args.resume:
            resume_path = args.resume.format(variant=variant_name)
            if os.path.exists(resume_path):
                resumed_epoch = trainer.load_checkpoint(resume_path)

        checkpoints = trainer.fit(
            epochs=args.epochs,
            save_every=args.save_every,
            checkpoint_prefix=variant_name,
            start_epoch=resumed_epoch,
        )
        curve_path = os.path.join(args.output_dir, f"{variant_name}_convergence.png")
        metrics_curve_path = os.path.join(args.output_dir, f"{variant_name}_metrics.png")
        plot_convergence(trainer.losses, trainer.losses, save_path=curve_path)
        plot_training_metrics(
            trainer.train_best_costs,
            trainer.train_mean_costs,
            trainer.val_costs,
            save_path=metrics_curve_path,
        )

        summary[variant_name] = {
            "resumed_epoch": resumed_epoch,
            "epochs": args.epochs,
            "final_loss": trainer.losses[-1] if trainer.losses else None,
            "final_train_best_cost": trainer.train_best_costs[-1] if trainer.train_best_costs else None,
            "final_train_mean_cost": trainer.train_mean_costs[-1] if trainer.train_mean_costs else None,
            "final_val_cost": trainer.val_costs[-1] if trainer.val_costs else None,
            "best_val_cost": trainer.best_val_cost if trainer.best_val_cost < float("inf") else None,
            "best_val_score": trainer.best_val_score if trainer.best_val_score < float("inf") else None,
            "best_epoch": trainer.best_epoch,
            "checkpoints": checkpoints,
            "best_checkpoint": os.path.join(args.output_dir, f"{variant_name}_best.pt"),
            "curve_path": curve_path,
            "metrics_curve_path": metrics_curve_path,
            "recommended_checkpoint": os.path.join(args.output_dir, f"{variant_name}_best.pt"),
            "train_knobs": {
                "init_checkpoint": args.init_checkpoint,
                "val_set_size": args.val_set_size,
                "val_batch_chunk": args.val_batch_chunk,
                "val_decode_type": args.val_decode_type,
                "val_sample_rollouts": args.val_sample_rollouts,
                "val_multi_start_rollouts": args.val_multi_start_rollouts,
                "val_augment_8_fold": args.val_augment_8_fold,
                "elite_fraction": args.elite_fraction,
                "elite_loss_weight": args.elite_loss_weight,
                "entropy_start": args.entropy_start,
                "entropy_end": args.entropy_end,
                "scheduler_patience": args.scheduler_patience,
                "scheduler_factor": args.scheduler_factor,
                "min_lr": args.min_lr,
                "curriculum_start_tightness": args.curriculum_start_tightness,
                "curriculum_epochs": args.curriculum_epochs,
                "early_stop_patience": args.early_stop_patience,
                "early_stop_min_epochs": args.early_stop_min_epochs,
                "ema_decay": args.ema_decay,
                "use_ema_eval": (USE_EMA_EVAL and (not args.disable_ema_eval)),
                "normalize_advantage": (NORMALIZE_ADVANTAGE and (not args.disable_adv_normalization)),
                "scenario": args.scenario,
                "n_pomo": args.n_pomo,
                "bc_teacher": args.bc_teacher,
                "bc_batch_size": args.bc_batch_size,
                "bc_weight_start": args.bc_weight_start,
                "bc_weight_end": args.bc_weight_end,
                "bc_epochs": args.bc_epochs,
                "pref_mode": args.pref_mode,
                "pref_alpha": args.pref_alpha,
                "pref_beta": args.pref_beta,
                "pref_random_low": args.pref_random_low,
                "pref_random_high": args.pref_random_high,
                "val_pref_mode": args.val_pref_mode if args.val_pref_mode is not None else args.pref_mode,
                "val_pref_alpha": args.val_pref_alpha if args.val_pref_alpha is not None else args.pref_alpha,
                "val_pref_beta": args.val_pref_beta if args.val_pref_beta is not None else args.pref_beta,
                "val_problem_types": args.val_problem_types or args.problem_type,
                "grad_accum_steps": args.grad_accum_steps,
                "fast_cuda": args.fast_cuda,
                "amp": args.amp,
                "amp_dtype": args.amp_dtype,
                "matmul_precision": args.matmul_precision,
            },
        }

    summary_path = os.path.join(args.output_dir, "training_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
