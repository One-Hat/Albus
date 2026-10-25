"""
Self-Calibrating Swarm Engine (Section 6 of Build Spec):
- Brier score, Brier Skill Score (BSS), Log Loss, 10-bin ECE, Reliability Diagram, Bootstrap CIs.
- Walk-forward Isotonic Regression recalibration once >= 100 forecasts have resolved.
- Strict separation of 'retrospective_clean' and 'live' forecast histories (Design Decision 8).
"""
from __future__ import annotations
import math
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple
import numpy as np

from swarm.aggregator import aggregate_logit, update_weights_multiplicative


def fit_predict_isotonic(
    train_x: np.ndarray,
    train_y: np.ndarray,
    query_x: float,
    y_min: float = 0.02,
    y_max: float = 0.98,
) -> float:
    """
    Exact Pool-Adjacent-Violators Algorithm (PAVA) for non-decreasing isotonic regression,
    with linear interpolation and clipping to [y_min, y_max].
    """
    order = np.argsort(train_x, kind="mergesort")
    x_sorted = np.asarray(train_x[order], dtype=float)
    y_sorted = np.asarray(train_y[order], dtype=float)

    # Pool-Adjacent-Violators (block values, weights, min_x, max_x)
    blocks: List[List[float]] = []
    for xi, yi in zip(x_sorted, y_sorted):
        blocks.append([yi, 1.0, xi, xi])
        while len(blocks) >= 2 and blocks[-2][0] > blocks[-1][0]:
            b2 = blocks.pop()
            b1 = blocks.pop()
            w_sum = b1[1] + b2[1]
            v_avg = (b1[0] * b1[1] + b2[0] * b2[1]) / w_sum
            blocks.append([v_avg, w_sum, b1[2], b2[3]])

    # Expand back to knots for interpolation
    knot_x = np.empty(len(x_sorted), dtype=float)
    knot_y = np.empty(len(x_sorted), dtype=float)
    idx = 0
    for val, count, _, _ in blocks:
        c = int(count)
        knot_x[idx : idx + c] = x_sorted[idx : idx + c]
        knot_y[idx : idx + c] = val
        idx += c

    # Deduplicate identical x knots before interpolation
    uniq_x, inv = np.unique(knot_x, return_inverse=True)
    uniq_y = np.zeros_like(uniq_x)
    counts = np.zeros_like(uniq_x)
    np.add.at(uniq_y, inv, knot_y)
    np.add.at(counts, inv, 1.0)
    uniq_y /= counts

    pred = float(np.interp(query_x, uniq_x, uniq_y))
    return float(np.clip(pred, y_min, y_max))


def compute_brier(
    predictions: Sequence[float] | np.ndarray,
    outcomes: Sequence[int] | np.ndarray,
) -> float:
    """Compute mean Brier score: mean((p - o)^2). Lower is better."""
    p = np.asarray(predictions, dtype=float)
    o = np.asarray(outcomes, dtype=float)
    if p.size == 0:
        return 0.0
    return float(np.mean((p - o) ** 2))


def compute_brier_skill_score(
    predictions: Sequence[float] | np.ndarray,
    outcomes: Sequence[int] | np.ndarray,
) -> float:
    """
    Compute Brier Skill Score (BSS) relative to the climatological base rate:
    BSS = 1 - (Brier / Brier_baserate).
    """
    p = np.asarray(predictions, dtype=float)
    o = np.asarray(outcomes, dtype=float)
    if p.size == 0:
        return 0.0
    base_rate = float(np.mean(o))
    brier_ref = float(np.mean((base_rate - o) ** 2))
    if brier_ref <= 1e-12:
        return 0.0
    brier_model = compute_brier(p, o)
    return float(1.0 - (brier_model / brier_ref))


def compute_log_loss(
    predictions: Sequence[float] | np.ndarray,
    outcomes: Sequence[int] | np.ndarray,
    eps: float = 1e-6,
) -> float:
    """Compute binary cross-entropy / log loss."""
    p = np.clip(np.asarray(predictions, dtype=float), eps, 1.0 - eps)
    o = np.asarray(outcomes, dtype=float)
    if p.size == 0:
        return 0.0
    loss = -(o * np.log(p) + (1.0 - o) * np.log(1.0 - p))
    return float(np.mean(loss))


def compute_ece_and_reliability(
    predictions: Sequence[float] | np.ndarray,
    outcomes: Sequence[int] | np.ndarray,
    n_bins: int = 10,
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Compute Expected Calibration Error (ECE) with `n_bins` equal-width bins on [0, 1]
    and return structured reliability diagram bin records.
    """
    p = np.asarray(predictions, dtype=float)
    o = np.asarray(outcomes, dtype=float)
    n = p.size
    bins_data: List[Dict[str, Any]] = []
    ece = 0.0

    for b in range(n_bins):
        low = b / n_bins
        high = (b + 1) / n_bins
        if b < n_bins - 1:
            mask = (p >= low) & (p < high)
        else:
            mask = (p >= low) & (p <= high)

        count = int(np.sum(mask))
        if count > 0 and n > 0:
            avg_pred = float(np.mean(p[mask]))
            avg_out = float(np.mean(o[mask]))
            gap = abs(avg_pred - avg_out)
            ece += (count / n) * gap
        else:
            avg_pred = None
            avg_out = None
            gap = 0.0

        bins_data.append({
            "bin_index": b,
            "bin_low": round(low, 2),
            "bin_high": round(high, 2),
            "count": count,
            "mean_predicted": avg_pred,
            "empirical_frequency": avg_out,
            "calibration_gap": gap,
        })

    return float(ece), bins_data


def bootstrap_brier_ci(
    predictions: Sequence[float] | np.ndarray,
    outcomes: Sequence[int] | np.ndarray,
    n_boot: int = 500,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """Compute bootstrap confidence interval for the Brier score."""
    p = np.asarray(predictions, dtype=float)
    o = np.asarray(outcomes, dtype=float)
    n = p.size
    if n < 2:
        val = compute_brier(p, o)
        return (val, val)

    rng = np.random.default_rng(seed)
    sq_errors = (p - o) ** 2
    boot_means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = float(np.mean(sq_errors[idx]))

    alpha = (1.0 - ci) / 2.0
    return (float(np.quantile(boot_means, alpha)), float(np.quantile(boot_means, 1.0 - alpha)))


class CalibrationEngine:
    """
    Maintains agent weights, performs walk-forward logit ensemble aggregation and
    isotonic regression recalibration (once >= 100 forecasts have resolved),
    and tracks separate histories for 'retrospective_clean' and 'live' forecasts.
    """

    def __init__(
        self,
        agent_ids: Sequence[str],
        eta: float = 3.0,
        floor_factor: float = 0.2,
        min_isotonic_samples: int = 100,
    ):
        if not agent_ids:
            raise ValueError("agent_ids cannot be empty.")
        self.agent_ids = list(agent_ids)
        n = len(self.agent_ids)
        self._unfloored_weights: Dict[str, float] = {aid: 1.0 / n for aid in self.agent_ids}
        self.weights: Dict[str, float] = {aid: 1.0 / n for aid in self.agent_ids}
        self.eta = eta
        self.floor_factor = floor_factor
        self.min_isotonic_samples = min_isotonic_samples

        # Separate history per forecast kind (Design Decision 8: never mix in one Brier number)
        self.history: Dict[Literal["retrospective_clean", "live"], List[Dict[str, Any]]] = {
            "retrospective_clean": [],
            "live": [],
        }

    def predict_ensemble(
        self,
        agent_predictions: Dict[str, float],
        kind: Literal["retrospective_clean", "live"] = "retrospective_clean",
    ) -> Tuple[float, float]:
        """
        Compute (raw_logit_ensemble_p, walk_forward_recalibrated_p).
        If at least `min_isotonic_samples` past forecasts of `kind` have resolved,
        fit IsotonicRegression strictly on past resolved data only.
        """
        raw_p = aggregate_logit(agent_predictions, self.weights)
        past_records = self.history[kind]

        if len(past_records) >= self.min_isotonic_samples:
            past_raw = np.array([r["raw_ensemble_p"] for r in past_records], dtype=float)
            past_outcomes = np.array([r["outcome"] for r in past_records], dtype=float)
            # Only fit isotonic if past_outcomes has both 0 and 1
            if len(np.unique(past_outcomes)) > 1:
                cal_p = fit_predict_isotonic(past_raw, past_outcomes, raw_p, y_min=0.02, y_max=0.98)
                return raw_p, cal_p

        return raw_p, raw_p

    def step_round(
        self,
        question_id: str,
        agent_predictions: Dict[str, float],
        outcome: int,
        kind: Literal["retrospective_clean", "live"] = "retrospective_clean",
    ) -> Dict[str, Any]:
        """
        Score a single resolved question walk-forward:
        1. Produce walk-forward ensemble prediction using current weights (before seeing outcome).
        2. Compute per-agent Brier scores (p_a - o)^2.
        3. Update weights w_a <- w_a * exp(-eta * Brier_a) with floor 0.2/N.
        """
        if outcome not in (0, 1):
            raise ValueError(f"Outcome must be 0 or 1, got {outcome}")

        # 1. Walk-forward ensemble prediction prior to weight update
        raw_ens_p, cal_ens_p = self.predict_ensemble(agent_predictions, kind=kind)
        equal_weights = {aid: 1.0 / len(agent_predictions) for aid in agent_predictions}
        equal_ens_p = aggregate_logit(agent_predictions, equal_weights)

        # 2. Per-agent Brier scores
        brier_scores: Dict[str, float] = {}
        for aid, p_a in agent_predictions.items():
            brier_scores[aid] = (float(p_a) - float(outcome)) ** 2

        # 3. Record in kind-specific history
        round_record = {
            "question_id": question_id,
            "outcome": int(outcome),
            "raw_ensemble_p": raw_ens_p,
            "calibrated_ensemble_p": cal_ens_p,
            "equal_weight_p": equal_ens_p,
            "agent_predictions": dict(agent_predictions),
            "agent_briers": brier_scores,
            "weights_before": dict(self.weights),
        }
        self.history[kind].append(round_record)

        # 4. Update unfloored weights multiplicatively, then project onto diversity floor 0.2/N
        self._unfloored_weights = update_weights_multiplicative(
            self._unfloored_weights,
            brier_scores,
            eta=self.eta,
            floor_factor=0.0,
        )
        self.weights = update_weights_multiplicative(
            self._unfloored_weights,
            {aid: 0.0 for aid in self.agent_ids},
            eta=0.0,
            floor_factor=self.floor_factor,
        )
        round_record["weights_after"] = dict(self.weights)
        return round_record

    def get_metrics_summary(
        self,
        kind: Literal["retrospective_clean", "live"] = "retrospective_clean",
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Compute comprehensive calibration metrics for the specified forecast kind.
        Never mixes 'retrospective_clean' and 'live' forecasts.
        """
        records = self.history[kind]
        n = len(records)
        if n == 0:
            return {
                "kind": kind,
                "resolved_count": 0,
                "isotonic_active": False,
                "swarm_brier": 0.0,
                "swarm_brier_ci_95": (0.0, 0.0),
                "brier_skill_score": 0.0,
                "log_loss": 0.0,
                "ece": 0.0,
                "reliability_bins": [],
                "equal_weight_brier": 0.0,
                "base_rate_brier": 0.0,
                "agent_briers": {},
                "current_weights": dict(self.weights),
            }

        outcomes = np.array([r["outcome"] for r in records], dtype=int)
        cal_preds = np.array([r["calibrated_ensemble_p"] for r in records], dtype=float)
        eq_preds = np.array([r["equal_weight_p"] for r in records], dtype=float)

        base_rate = float(np.mean(outcomes))
        base_preds = np.full(n, base_rate, dtype=float)

        swarm_brier = compute_brier(cal_preds, outcomes)
        swarm_ci = bootstrap_brier_ci(cal_preds, outcomes, seed=seed)
        bss = compute_brier_skill_score(cal_preds, outcomes)
        ll = compute_log_loss(cal_preds, outcomes)
        ece, rel_bins = compute_ece_and_reliability(cal_preds, outcomes, n_bins=10)
        eq_brier = compute_brier(eq_preds, outcomes)
        br_brier = compute_brier(base_preds, outcomes)

        agent_briers: Dict[str, float] = {}
        for aid in self.agent_ids:
            aid_preds = [
                r["agent_predictions"][aid]
                for r in records
                if aid in r["agent_predictions"]
            ]
            aid_outs = [
                r["outcome"]
                for r in records
                if aid in r["agent_predictions"]
            ]
            if aid_preds:
                agent_briers[aid] = compute_brier(aid_preds, aid_outs)

        return {
            "kind": kind,
            "resolved_count": n,
            "isotonic_active": n >= self.min_isotonic_samples,
            "swarm_brier": swarm_brier,
            "swarm_brier_ci_95": swarm_ci,
            "brier_skill_score": bss,
            "log_loss": ll,
            "ece": ece,
            "reliability_bins": rel_bins,
            "equal_weight_brier": eq_brier,
            "base_rate_brier": br_brier,
            "agent_briers": agent_briers,
            "current_weights": dict(self.weights),
        }
