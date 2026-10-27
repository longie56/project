"""Quality-blind nine-row timing gate for either frozen POMO candidate tier."""
from __future__ import annotations

import math
from collections import Counter
from typing import Mapping, Sequence

TIERS = {"eight_candidate", "same_candidate_full"}
SCALES = (20, 50, 100)


class TimingGateRejected(RuntimeError):
    pass


def classify(record: Mapping[str, object], tier: str) -> str:
    if tier not in TIERS:
        raise TimingGateRejected("unknown POMO candidate tier")
    n = int(record["n"])
    if n not in SCALES or float(record["alpha"]) != .5 or int(record["training_seed"]) != 11:
        raise TimingGateRejected("unexpected scale, preference or seed in POMO pilot")
    expected = 8 if tier == "eight_candidate" else n
    if record["candidate_tier"] != tier or int(record["candidate_count"]) != expected:
        raise TimingGateRejected("POMO pilot tier or candidate count differs from frozen tier")
    if int(record["manifest_prefix_index"]) not in range(3):
        raise TimingGateRejected("POMO pilot item is outside manifest prefix")
    budget = 60.0 if n in (20, 50) else 300.0
    elapsed = float(record["elapsed_s"])
    if float(record["budget_s"]) != budget or not math.isfinite(elapsed) or elapsed < 0:
        raise TimingGateRejected("invalid POMO pilot budget or time")
    if record["status"] != "evaluated" or elapsed > budget or record["budget_qualified"] is not True:
        return "stop"
    return "narrow_margin" if elapsed > budget / 2 else "wide_margin"


def decide(records: Sequence[Mapping[str, object]], tier: str) -> dict[str, object]:
    if len(records) != 9:
        raise TimingGateRejected(f"POMO pilot incomplete: {len(records)}/9")
    if Counter(int(x["n"]) for x in records) != {20: 3, 50: 3, 100: 3}:
        raise TimingGateRejected("POMO pilot scale counts differ from frozen prefix")
    keys = Counter((int(x["n"]), int(x["manifest_prefix_index"])) for x in records)
    if keys != Counter((n, i) for n in SCALES for i in range(3)):
        raise TimingGateRejected("POMO pilot prefix missing or duplicated")
    labels = [classify(x, tier) for x in records]
    return {"decision": ("stop" if "stop" in labels else
                         "narrow_margin" if "narrow_margin" in labels else "wide_margin"),
            "rows": 9, "narrow_margin_rows": labels.count("narrow_margin")}
