"""Select S/B from Gu-replayed post-cleanup candidates; no model changes."""
from __future__ import annotations


def representatives(candidates: list[dict]) -> list[int]:
    seen = set()
    chosen = []
    for candidate in candidates:
        actions = candidate["original_actions"]
        first = actions[0] if actions else None
        if first not in seen:
            seen.add(first)
            chosen.append(candidate["lane"])
    return chosen


def rank_s(candidates: list[dict], lanes: list[int]) -> int:
    selected = [candidates[i] for i in lanes]
    complete = [c for c in selected if c["complete"] and not c["rollout_limit_hit"]]
    if complete:
        return min(complete, key=lambda c: (c["cost_c0"], c["lane"]))["lane"]
    fewest = min(c["unserved"] for c in selected)
    eligible = [c for c in selected if c["unserved"] == fewest]
    no_limit = [c for c in eligible if not c["rollout_limit_hit"]]
    if no_limit:
        eligible = no_limit
    return min(eligible, key=lambda c: (c["cost_c0"], c["lane"]))["lane"]


def select_post_cleanup(candidates: list[dict]) -> tuple[int, int, list[int]]:
    reps = representatives(candidates)
    s_lane = rank_s(candidates, reps)
    baseline = candidates[s_lane]
    eligible = [c for c in candidates if c["complete"] and not c["rollout_limit_hit"]
                and c["cost_c0"] <= baseline["cost_c0"] + 1e-6
                and c["tardiness_s"] <= baseline["tardiness_s"] + .05]
    b_lane = (min(eligible, key=lambda c: (c["tardiness_s"], c["cost_c0"],
                                           c["lane"]))["lane"] if eligible else s_lane)
    return s_lane, b_lane, reps
