"""Read-only paired and physical-instance-cluster report for the frozen cloud run."""
from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ATTEMPT = HERE / "attempt_001"
FIELDS = ("cost_c0", "tardiness_s", "flight_time_s", "total_energy_j")
BOOTSTRAP_SEED = 20261027
BOOTSTRAP_DRAWS = 10000


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired_values(rows: list[dict], arm: str, field: str) -> list[float]:
    values = []
    for row in rows:
        old, new = row[f"old_{arm}"][field], row[f"new_{arm}"][field]
        if old is not None and new is not None and math.isfinite(old) and math.isfinite(new):
            values.append(new - old)
    return values


def cluster_interval(rows: list[dict], arm: str, field: str) -> dict:
    clusters = defaultdict(list)
    for row in rows:
        n, kind, rep, _, _ = row["unique_key"]
        clusters[(n, kind, rep)].append(row)
    cluster_means = []
    for group in clusters.values():
        values = paired_values(group, arm, field)
        if values:
            cluster_means.append(statistics.fmean(values))
    if not cluster_means:
        return {"clusters": 0, "mean_of_cluster_means": None, "ci95": None}
    rng = random.Random(BOOTSTRAP_SEED)
    k = len(cluster_means)
    samples = sorted(statistics.fmean(cluster_means[rng.randrange(k)] for _ in range(k))
                     for _ in range(BOOTSTRAP_DRAWS))
    return {"clusters": k,
            "mean_of_cluster_means": statistics.fmean(cluster_means),
            "ci95": [samples[int(.025 * BOOTSTRAP_DRAWS)],
                     samples[int(.975 * BOOTSTRAP_DRAWS)]]}


def summarize(rows: list[dict]) -> dict:
    result = {"settings": len(rows),
              "physical_instances": len({tuple(r["unique_key"][:3]) for r in rows}),
              "candidate_lanes": sum(r["candidate_count"] for r in rows),
              "cleanup_changed_candidates": sum(r["cleanup_changed_candidates"] for r in rows),
              "cleaned_replay_error_candidates": sum(r["cleaned_replay_error_candidates"] for r in rows),
              "S_reselected": sum(r["S_reselected"] for r in rows),
              "B_reselected": sum(r["B_reselected"] for r in rows),
              "arms": {}}
    for arm in ("S", "B"):
        entry = {"old_complete": sum(r[f"old_{arm}"]["complete"] for r in rows),
                 "new_complete": sum(r[f"new_{arm}"]["complete"] for r in rows),
                 "old_feasible": sum(r[f"old_{arm}"]["feasible"] for r in rows),
                 "new_feasible": sum(r[f"new_{arm}"]["feasible"] for r in rows),
                 "paired_metrics": {}}
        for field in FIELDS:
            values = paired_values(rows, arm, field)
            entry["paired_metrics"][field] = {
                "finite_pairs": len(values),
                "mean_new_minus_old": statistics.fmean(values) if values else None,
                "cluster_bootstrap": cluster_interval(rows, arm, field),
            }
        result["arms"][arm] = entry
    return result


def main() -> None:
    completion = json.loads((ATTEMPT / "completion.json").read_text())
    assert completion["rows"] == 945
    source = ATTEMPT / "selection_rows.jsonl"
    assert sha(source) == completion["selection_rows_sha256"]
    rows = [json.loads(line) for line in source.open()]
    assert len(rows) == 945
    expected = {(n, kind, rep, alpha, seed)
                for n in (20, 50, 100) for kind in ("R", "C", "RC")
                for rep in range(5) for alpha in (.5, .6, .625, .65, .675, .7, .9)
                for seed in (11, 23, 37)}
    assert {tuple(row["unique_key"]) for row in rows} == expected
    report = {"scope": "posthoc_W05-I_development_not_blind_main90",
              "source_sha256": sha(source),
              "analyzer_sha256": sha(Path(__file__)),
              "bootstrap_seed": BOOTSTRAP_SEED,
              "bootstrap_draws": BOOTSTRAP_DRAWS,
              "overall": summarize(rows),
              "by_scale": {str(n): summarize([r for r in rows if r["unique_key"][0] == n])
                           for n in (20, 50, 100)}}
    target = ATTEMPT / "POSTHOC_CLUSTER_REPORT.json"
    with target.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"report": str(target), "sha256": sha(target),
                      "settings": report["overall"]["settings"]}))


if __name__ == "__main__":
    main()
