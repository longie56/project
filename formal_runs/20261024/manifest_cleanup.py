"""Remove parcels loaded but not delivered before the next depot return.

This is a deterministic action-stream normalization. It does not change the
customer sequence, speed choices, depot waits, returns, or Gu physics. The
caller must replay both streams with the canonical environment.
"""

from __future__ import annotations

from collections.abc import Iterable


def remove_undelivered_loads(actions: Iterable[int], n_nodes: int) -> tuple[list[int], dict]:
    if n_nodes <= 0:
        raise ValueError("n_nodes must be positive")
    source = [int(action) for action in actions]
    keep = [True] * len(source)
    trip_start = 0
    removed = []
    trips = 0
    for end, action in enumerate(source):
        if not 3 * n_nodes <= action < 3 * n_nodes + 2:
            continue
        delivered = {a // 2 for a in source[trip_start:end] if 0 <= a < 2 * n_nodes}
        for index in range(trip_start, end):
            a = source[index]
            if 2 * n_nodes <= a < 3 * n_nodes and a - 2 * n_nodes not in delivered:
                keep[index] = False
                removed.append({"action_index": index, "order": a - 2 * n_nodes,
                                "sortie_index": trips})
        trips += 1
        trip_start = end + 1
    if trip_start != len(source):
        raise ValueError("Action stream has no final depot return")
    cleaned = [action for action, retained in zip(source, keep) if retained]
    return cleaned, {"removed_load_count": len(removed), "removed_loads": removed,
                     "sorties": trips}
