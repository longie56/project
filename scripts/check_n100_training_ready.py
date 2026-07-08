import argparse
import os
import sys
import time

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.solomon_generator import SolomonGenerator
from model.pomo import PC_EGA_POMO
from model.variants import VARIANTS
from seed import set_global_seed
from train.trainer import Trainer


def _load_checkpoint_if_given(model, checkpoint):
    if not checkpoint:
        return
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    try:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(checkpoint, map_location="cpu")
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"checkpoint loaded: {checkpoint} (missing={len(missing)}, unexpected={len(unexpected)})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--n-pomo", type=int, default=8)
    parser.add_argument("--n-nodes", type=int, default=100)
    parser.add_argument("--problem-type", default="R")
    parser.add_argument("--tightness", type=float, default=0.7)
    parser.add_argument("--scenario", default="full")
    parser.add_argument("--pref-mode", choices=["fixed", "random"], default="fixed")
    parser.add_argument("--pref-alpha", type=float, default=0.5)
    parser.add_argument("--pref-beta", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_global_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"torch={torch.__version__}, cuda={torch.cuda.is_available()}, device={device}")
    if torch.cuda.is_available():
        print(f"gpu={torch.cuda.get_device_name(0)}")
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"gpu_mem_total_gb={total_gb:.2f}")

    model = PC_EGA_POMO(**VARIANTS["full"]).to(device)
    _load_checkpoint_if_given(model, args.checkpoint)
    model.eval()

    generator = SolomonGenerator()
    nodes, depot, visited = generator.generate(
        args.batch_size,
        args.n_nodes,
        args.problem_type,
        args.tightness,
        device,
    )
    if args.pref_mode == "fixed":
        pref = torch.tensor([[args.pref_alpha, args.pref_beta]], device=device).expand(args.batch_size, -1)
        pref = pref / pref.sum(dim=1, keepdim=True).clamp_min(1e-9)
    else:
        pref = torch.rand(args.batch_size, 2, device=device) * 0.8 + 0.1
        pref = pref / pref.sum(dim=1, keepdim=True)

    start = time.perf_counter()
    with torch.no_grad():
        result = model.inference_rollout(
            nodes,
            pref,
            depot,
            visited_mask=visited,
            N_pomo=args.n_pomo,
            decode_type="greedy",
            scenario=args.scenario,
        )
    elapsed = time.perf_counter() - start
    print(
        f"n{args.n_nodes} inference smoke:",
        f"cost={result['best_cost'].mean().item():.4f}",
        f"phys={result['best_phys'].mean().item():.4f}",
        f"pen={result['best_pen'].mean().item():.4f}",
        f"unvisit={result['best_unvisit'].mean().item():.4f}",
        f"feasible={result['feasible'].float().mean().item():.4f}",
        f"time={elapsed:.2f}s",
    )

    trainer = Trainer(
        model=PC_EGA_POMO(**VARIANTS["full"]),
        variant_name="full",
        batch_size=1,
        n_nodes=args.n_nodes,
        n_pomo=min(args.n_pomo, args.n_nodes),
        steps_per_epoch=1,
        problem_type=args.problem_type,
        scenario=args.scenario,
        tightness=args.tightness,
        val_set_size=2,
        val_batch_chunk=1,
        pref_mode=args.pref_mode,
        pref_alpha=args.pref_alpha,
        pref_beta=args.pref_beta,
        use_ema_eval=False,
    )
    if trainer.val_data is None:
        raise RuntimeError("validation set was not built")
    val_pref_preview = {
        problem_type: data["pref"][: min(2, data["pref"].size(0))].detach().cpu().tolist()
        for problem_type, data in trainer.val_data.items()
    }
    print("validation pref smoke:", val_pref_preview)
    print("N100 training readiness smoke passed.")


if __name__ == "__main__":
    main()
