"""Lexicographic POMO selection for complete physical routes at inference."""

from __future__ import annotations

import torch


def greedy_action(logits: torch.Tensor, legal: torch.Tensor, n_nodes: int,
                  strategy: str = "joint") -> torch.Tensor:
    """Decode one legal action, either jointly or by route then speed.

    Hierarchical decoding first maximizes the marginal probability of a
    customer/return group, then chooses its more likely legal speed. Loading
    and depot waiting remain singleton groups. It uses the same joint policy
    logits; only the deterministic inference rule differs.
    """
    n = n_nodes
    if logits.ndim != 2 or logits.shape != legal.shape or logits.size(1) != 3*n+3:
        raise ValueError("Action logit/mask shape mismatch")
    if not bool(legal.any(dim=1).all()):
        raise ValueError("No legal action in greedy decoding")
    masked = logits.masked_fill(~legal, float("-inf"))
    if strategy == "joint":
        return masked.argmax(dim=1)
    if strategy != "hierarchical":
        raise ValueError(f"Unknown greedy strategy: {strategy}")
    flight = masked[:, :2*n].reshape(-1, n, 2)
    returns = masked[:, 3*n:3*n+2]
    group = torch.cat([
        torch.logsumexp(flight, dim=2), masked[:, 2*n:3*n],
        torch.logsumexp(returns, dim=1, keepdim=True),
        masked[:, 3*n+2:3*n+3],
    ], dim=1)
    selected = group.argmax(dim=1)
    rows = torch.arange(masked.size(0), device=masked.device)
    flight_idx = selected.clamp(max=n-1)
    flight_action = 2*flight_idx + flight[rows, flight_idx].argmax(dim=1)
    load_action = n+selected
    return_action = 3*n+returns.argmax(dim=1)
    wait_action = torch.full_like(selected, 3*n+2)
    return torch.where(selected < n, flight_action,
                       torch.where(selected < 2*n, load_action,
                                   torch.where(selected == 2*n, return_action,
                                               wait_action)))


def select_best_complete(output: dict[str, torch.Tensor]) -> torch.Tensor:
    """Select per instance by completeness, unserved, limit, then raw C0/C1.

    The finite penalty used during REINFORCE remains a training surrogate. It
    cannot guarantee that a complete feasible trajectory beats every invalid
    trajectory; this selector provides that guarantee for inference and model
    validation without changing the physical objective or training gradient.
    """
    cost = output["cost"]
    feasible = output["feasible"].bool()
    unserved = output["unserved"]
    limit_hit = output["rollout_limit_hit"].bool()
    if (cost.ndim != 2 or feasible.shape != cost.shape
            or unserved.shape != cost.shape or limit_hit.shape != cost.shape):
        raise ValueError("POMO output shape mismatch")
    if not bool(torch.isfinite(cost).all()):
        raise FloatingPointError("Nonfinite POMO raw objective")
    complete = feasible & (unserved == 0) & ~limit_hit
    candidate = torch.ones_like(complete)
    has_complete = complete.any(dim=1, keepdim=True)
    candidate = torch.where(has_complete, complete, candidate)
    minimum_unserved = unserved.masked_fill(~candidate, torch.iinfo(unserved.dtype).max)
    minimum_unserved = minimum_unserved.min(dim=1, keepdim=True).values
    candidate = candidate & (unserved == minimum_unserved)
    no_limit = candidate & ~limit_hit
    candidate = torch.where(no_limit.any(dim=1, keepdim=True), no_limit, candidate)
    return cost.masked_fill(~candidate, float("inf")).argmin(dim=1)
