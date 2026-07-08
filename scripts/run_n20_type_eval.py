import argparse
import os
import subprocess
import sys


def _run(cmd):
    print("\n" + " ".join(cmd), flush=True)
    completed = subprocess.run(cmd)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def _csv(value):
    return [item.strip().upper() for item in value.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate one N20 checkpoint separately on R/C/RC synthetic Solomon-style test sets."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--instances", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--problem-types", default="R,C,RC")
    parser.add_argument("--output-dir", default="tables/n20_by_type")
    parser.add_argument("--n-nodes", type=int, default=20)
    parser.add_argument("--n-pomo", type=int, default=20)
    parser.add_argument("--scenario", choices=["full", "simple"], default="full")
    parser.add_argument("--tightness", type=float, default=0.7)
    parser.add_argument("--pref-mode", choices=["fixed", "random"], default="fixed")
    parser.add_argument("--pref-alpha", type=float, default=0.5)
    parser.add_argument("--pref-beta", type=float, default=0.5)
    parser.add_argument("--pref-random-low", type=float, default=0.1)
    parser.add_argument("--pref-random-high", type=float, default=0.9)
    parser.add_argument("--decode-type", choices=["greedy", "sample"], default="greedy")
    parser.add_argument("--sample-rollouts", type=int, default=0)
    parser.add_argument("--multi-start-rollouts", type=int, default=0)
    parser.add_argument("--augment-8-fold", action="store_true")
    parser.add_argument("--run-alns", action="store_true")
    parser.add_argument("--alns-time-limit", type=float, default=5.0)
    parser.add_argument("--total-time-budget", type=float, default=5.0)
    parser.add_argument("--time-budget-safety", type=float, default=0.15)
    parser.add_argument("--alns-max-iter", type=int, default=1200)
    parser.add_argument("--alns-destroy-min", type=int, default=1)
    parser.add_argument("--alns-destroy-max", type=int, default=8)
    parser.add_argument("--alns-max-insert-positions", type=int, default=12)
    parser.add_argument("--alns-swap-neighbors", type=int, default=18)
    parser.add_argument("--alns-restart-rounds", type=int, default=1)
    parser.add_argument("--run-ortools-physical", action="store_true")
    parser.add_argument("--ortools-time-limit", type=float, default=5.0)
    parser.add_argument("--ortools-route-time-limit", type=float, default=3.0)
    parser.add_argument("--run-pyvrp", action="store_true")
    parser.add_argument("--pyvrp-time-limit", type=float, default=10.0)
    parser.add_argument("--skip-aggregate", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    compare_script = os.path.join(os.path.dirname(__file__), "compare_baselines_n20.py")
    table_script = os.path.join(os.path.dirname(__file__), "build_paper_table.py")

    result_paths = []
    for problem_type in _csv(args.problem_types):
        json_path = os.path.join(
            args.output_dir,
            f"n20_{problem_type.lower()}_{args.pref_mode}_seed{args.seed}_{args.instances}.json",
        )
        md_path = os.path.join(
            args.output_dir,
            f"n20_{problem_type.lower()}_{args.pref_mode}_seed{args.seed}_{args.instances}.md",
        )
        cmd = [
            sys.executable,
            compare_script,
            "--instances",
            str(args.instances),
            "--n-nodes",
            str(args.n_nodes),
            "--problem-type",
            problem_type,
            "--scenario",
            args.scenario,
            "--tightness",
            str(args.tightness),
            "--pref-mode",
            args.pref_mode,
            "--pref-alpha",
            str(args.pref_alpha),
            "--pref-beta",
            str(args.pref_beta),
            "--pref-random-low",
            str(args.pref_random_low),
            "--pref-random-high",
            str(args.pref_random_high),
            "--variant",
            "full",
            "--checkpoint",
            args.checkpoint,
            "--decode-type",
            args.decode_type,
            "--n-pomo",
            str(args.n_pomo),
            "--sample-rollouts",
            str(args.sample_rollouts),
            "--multi-start-rollouts",
            str(args.multi_start_rollouts),
            "--run-drl",
            "--seed",
            str(args.seed),
            "--output-json",
            json_path,
        ]
        if args.augment_8_fold:
            cmd.append("--augment-8-fold")
        if args.run_alns:
            cmd.extend(
                [
                    "--run-alns",
                    "--alns-init-source",
                    "auto",
                    "--alns-time-limit",
                    str(args.alns_time_limit),
                    "--total-time-budget",
                    str(args.total_time_budget),
                    "--time-budget-safety",
                    str(args.time_budget_safety),
                    "--alns-max-iter",
                    str(args.alns_max_iter),
                    "--alns-destroy-min",
                    str(args.alns_destroy_min),
                    "--alns-destroy-max",
                    str(args.alns_destroy_max),
                    "--alns-max-insert-positions",
                    str(args.alns_max_insert_positions),
                    "--alns-swap-neighbors",
                    str(args.alns_swap_neighbors),
                    "--alns-restart-rounds",
                    str(args.alns_restart_rounds),
                ]
            )
        if args.run_ortools_physical:
            cmd.extend(
                [
                    "--run-ortools-physical",
                    "--ortools-physical-routing-mode",
                    "serial_order",
                    "--ortools-time-limit",
                    str(args.ortools_time_limit),
                    "--ortools-route-time-limit",
                    str(args.ortools_route_time_limit),
                ]
            )
        if args.run_pyvrp:
            cmd.extend(["--run-pyvrp", "--pyvrp-time-limit", str(args.pyvrp_time_limit)])

        _run(cmd)
        result_paths.append(json_path)

        methods = ["drl"]
        if args.run_alns:
            methods.append("alns")
        if args.run_ortools_physical:
            methods.insert(0, "ortools_physical")
        if args.run_pyvrp:
            methods.insert(0, "pyvrp")
        _run(
            [
                sys.executable,
                table_script,
                "--compare-json",
                json_path,
                "--methods",
                ",".join(methods),
                "--ref-method",
                "best",
                "--title",
                f"N20 {problem_type} test set",
                "--output-md",
                md_path,
            ]
        )

    print("\nGenerated result files:")
    for path in result_paths:
        print(path)

    if not args.skip_aggregate and result_paths:
        aggregate_script = os.path.join(os.path.dirname(__file__), "build_by_type_table.py")
        aggregate_methods = ["drl"]
        if args.run_alns:
            aggregate_methods.append("alns")
        if args.run_ortools_physical:
            aggregate_methods.insert(0, "ortools_physical")
        if args.run_pyvrp:
            aggregate_methods.insert(0, "pyvrp")
        _run(
            [
                sys.executable,
                aggregate_script,
                *result_paths,
                "--methods",
                ",".join(aggregate_methods),
                "--title",
                "N20 by-type test sets",
                "--output-md",
                os.path.join(args.output_dir, "n20_by_type_summary.md"),
                "--output-json",
                os.path.join(args.output_dir, "n20_by_type_summary.json"),
            ]
        )
        _run(
            [
                sys.executable,
                aggregate_script,
                *result_paths,
                "--methods",
                ",".join(aggregate_methods),
                "--title",
                "N20 by-type distribution summary",
                "--include-stats",
                "--output-md",
                os.path.join(args.output_dir, "n20_by_type_stats.md"),
            ]
        )
        _run(
            [
                sys.executable,
                aggregate_script,
                *result_paths,
                "--methods",
                ",".join(aggregate_methods),
                "--title",
                "N20 by-type compact summary",
                "--format",
                "compact",
                "--output-md",
                os.path.join(args.output_dir, "n20_by_type_compact.md"),
            ]
        )


if __name__ == "__main__":
    main()
