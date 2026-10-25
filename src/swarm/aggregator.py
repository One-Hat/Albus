"""
Logit-space ensemble aggregator and multiplicative weight update rule with diversity floor.
Implements Section 6 of the Build Spec:
- Ensemble: logit(p) = sum_a w_a * logit(p_a)
- Weights: w_a <- w_a * exp(-eta * Brier_a), normalized, with floor 0.2/N.
"""
from __future__ import annotations
import math
from typing import Dict


def logit(p: float) -> float:
    """Compute log-odds ln(p / (1 - p)) with numerical safety bounds."""
    clipped = min(1.0 - 1e-6, max(1e-6, float(p)))
    return math.log(clipped / (1.0 - clipped))


def sigmoid(x: float) -> float:
    """Compute logistic sigmoid 1 / (1 + exp(-x)) with overflow protection."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def aggregate_logit(
    agent_probs: Dict[str, float],
    weights: Dict[str, float],
    clip_bounds: tuple[float, float] = (0.02, 0.98),
) -> float:
    """
    Combine agent probabilities in logit space:
    logit(p) = sum_a w_a * logit(p_a)
    """
    if not agent_probs:
        raise ValueError("Cannot aggregate empty agent_probs.")

    common_agents = [a for a in agent_probs if a in weights]
    if not common_agents:
        # Equal weight fallback if weights dict doesn't match keys
        w_norm = {a: 1.0 / len(agent_probs) for a in agent_probs}
    else:
        total_w = sum(weights[a] for a in common_agents)
        if total_w <= 0:
            w_norm = {a: 1.0 / len(common_agents) for a in common_agents}
        else:
            w_norm = {a: weights[a] / total_w for a in common_agents}

    weighted_logit = sum(w_norm[a] * logit(agent_probs[a]) for a in w_norm)
    raw_p = sigmoid(weighted_logit)
    return min(clip_bounds[1], max(clip_bounds[0], raw_p))


def update_weights_multiplicative(
    weights: Dict[str, float],
    brier_scores: Dict[str, float],
    eta: float = 3.0,
    floor_factor: float = 0.2,
) -> Dict[str, float]:
    """
    Update agent weights via exponentiated gradient / multiplicative weights:
    w_a <- w_a * exp(-eta * Brier_a), then normalize and apply floor at floor_factor / N.
    """
    n = len(weights)
    if n == 0:
        return {}
    if n == 1:
        only_key = next(iter(weights))
        return {only_key: 1.0}

    raw_updated: Dict[str, float] = {}
    for aid, w_prev in weights.items():
        brier_a = brier_scores.get(aid, 0.25)
        raw_updated[aid] = max(1e-12, float(w_prev) * math.exp(-eta * float(brier_a)))

    total_raw = sum(raw_updated.values())
    normalized = {aid: val / total_raw for aid, val in raw_updated.items()}

    # Enforce floor at floor_factor / N (e.g. 0.2 / N) while keeping sum(w) == 1.0
    floor_val = min(floor_factor / n, 1.0 / n)

    # Iterative projection onto the floored simplex {w : w_i >= floor_val, sum(w) = 1}
    fixed_keys = set()
    result = dict(normalized)
    while True:
        violators = {k for k, v in result.items() if k not in fixed_keys and v < floor_val}
        if not violators:
            break
        fixed_keys.update(violators)
        for k in violators:
            result[k] = floor_val
        free_keys = [k for k in result if k not in fixed_keys]
        if not free_keys:
            result = {k: 1.0 / n for k in weights}
            break
        remaining_mass = 1.0 - len(fixed_keys) * floor_val
        free_sum = sum(result[k] for k in free_keys)
        if free_sum <= 0:
            for k in free_keys:
                result[k] = remaining_mass / len(free_keys)
        else:
            for k in free_keys:
                result[k] = result[k] * (remaining_mass / free_sum)

    return result
