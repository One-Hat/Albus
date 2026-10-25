"""
Unit and Gate tests for Phase 7: Resolver and calibration
(aggregator.py, calibrate.py, resolver.py).
Following AGENTS.md Rule 2: Tests written first.
"""
import math
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import pytest

from data.prices import PriceLoader
from extract.quote_check import FilingDiff
from swarm.aggregator import (
    aggregate_logit,
    logit,
    sigmoid,
    update_weights_multiplicative,
)
from swarm.calibrate import (
    CalibrationEngine,
    compute_brier,
    compute_brier_skill_score,
    compute_ece_and_reliability,
    compute_log_loss,
)
from swarm.ledger import ForecastLedger
from swarm.questions import Question, QuestionGenerator
from swarm.resolver import ResolverJob

EASTERN = ZoneInfo("America/New_York")


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp(prefix="test_phase7_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


def test_gate_hand_computed_metrics_and_logit_ensemble():
    """
    Gate requirement: Metrics and logit-space aggregation match a hand-computed example.
    """
    # 1. Hand-computed logit aggregation:
    # Suppose 2 agents: p_1 = 0.80, p_2 = 0.20, weights w_1 = 0.75, w_2 = 0.25
    # logit(0.80) = ln(4) = 1.3862943611
    # logit(0.20) = ln(0.25) = -1.3862943611
    # weighted logit = 0.75 * ln(4) + 0.25 * (-ln(4)) = 0.5 * ln(4) = ln(2) = 0.6931471806
    # p_ens = sigmoid(ln(2)) = 2 / (1 + 2) = 2/3 = 0.6666666667
    p_ens = aggregate_logit({"a1": 0.80, "a2": 0.20}, {"a1": 0.75, "a2": 0.25})
    assert pytest.approx(p_ens, abs=1e-6) == 2.0 / 3.0

    # 2. Hand-computed Brier score:
    # Predictions: [0.8, 0.2, 0.6, 0.4], Outcomes: [1, 0, 1, 0]
    # Squared errors: (0.8-1)^2=0.04, (0.2-0)^2=0.04, (0.6-1)^2=0.16, (0.4-0)^2=0.16
    # Mean Brier = (0.04 + 0.04 + 0.16 + 0.16) / 4 = 0.40 / 4 = 0.10
    preds = [0.8, 0.2, 0.6, 0.4]
    outcomes = [1, 0, 1, 0]
    brier = compute_brier(preds, outcomes)
    assert pytest.approx(brier, abs=1e-6) == 0.10

    # 3. Hand-computed Brier Skill Score (BSS) vs base rate:
    # Base rate o_bar = (1 + 0 + 1 + 0) / 4 = 0.50
    # Base rate Brier = 0.25
    # BSS = 1 - (0.10 / 0.25) = 1 - 0.40 = 0.60
    bss = compute_brier_skill_score(preds, outcomes)
    assert pytest.approx(bss, abs=1e-6) == 0.60

    # 4. Hand-computed Log Loss:
    # -1/4 * (ln(0.8) + ln(0.8) + ln(0.6) + ln(0.6)) = -0.5 * (ln(0.8) + ln(0.6))
    expected_ll = -0.5 * (math.log(0.8) + math.log(0.6))
    ll = compute_log_loss(preds, outcomes)
    assert pytest.approx(ll, abs=1e-6) == expected_ll

    # 5. Hand-computed ECE with 10 bins:
    # Bin [0.2, 0.3): pred=0.2, out=0 -> |0.2 - 0.0| = 0.2 (weight 1/4)
    # Bin [0.4, 0.5): pred=0.4, out=0 -> |0.4 - 0.0| = 0.4 (weight 1/4)
    # Bin [0.6, 0.7): pred=0.6, out=1 -> |0.6 - 1.0| = 0.4 (weight 1/4)
    # Bin [0.8, 0.9): pred=0.8, out=1 -> |0.8 - 1.0| = 0.2 (weight 1/4)
    # ECE = 0.25 * 0.2 + 0.25 * 0.4 + 0.25 * 0.4 + 0.25 * 0.2 = 0.30
    ece, rel_bins = compute_ece_and_reliability(preds, outcomes, n_bins=10)
    assert pytest.approx(ece, abs=1e-6) == 0.30
    assert len(rel_bins) == 10

    # 6. Multiplicative weight update with floor 0.2/N:
    # N = 2 -> floor is 0.2 / 2 = 0.10
    w_init = {"a1": 0.5, "a2": 0.5}
    briers = {"a1": 0.0, "a2": 1.0}
    w_next = update_weights_multiplicative(w_init, briers, eta=3.0, floor_factor=0.2)
    assert pytest.approx(sum(w_next.values()), abs=1e-6) == 1.0
    assert w_next["a1"] > w_next["a2"]
    assert w_next["a2"] >= 0.10 - 1e-9


def test_gate_simulate_10_agents_converges_within_200_rounds():
    """
    Section 6 & Phase 7 Gate Proof Test:
    Simulate 10 agents with known skill. The weights must converge to the best agent within 200 rounds.
    Also verifies walk-forward isotonic recalibration after 100 resolved forecasts.
    """
    rng = np.random.default_rng(42)
    agent_ids = [f"agent_{i}" for i in range(10)]
    engine = CalibrationEngine(agent_ids=agent_ids, eta=3.0, floor_factor=0.2)

    # Agent 0 is the sharpest/best agent (very small logit noise around true prob)
    # Agents 1..9 have progressively larger noise and calibration bias
    noise_stds = [0.15, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.2, 2.5, 3.0]
    biases = [0.0, 0.3, -0.4, 0.5, -0.6, 0.8, -0.8, 1.0, -1.2, 1.5]

    for round_idx in range(200):
        true_p = float(rng.uniform(0.10, 0.90))
        outcome = int(rng.random() < true_p)
        true_logit = logit(true_p)

        agent_preds = {}
        for idx, aid in enumerate(agent_ids):
            noisy_logit = true_logit + biases[idx] + float(rng.normal(0.0, noise_stds[idx]))
            p_a = float(np.clip(sigmoid(noisy_logit), 0.02, 0.98))
            agent_preds[aid] = p_a

        engine.step_round(
            question_id=f"sim_q_{round_idx}",
            agent_predictions=agent_preds,
            outcome=outcome,
            kind="retrospective_clean",
        )

    weights = engine.weights
    # Best agent must be agent_0 with the strictly highest weight
    best_agent = max(weights, key=lambda k: weights[k])
    assert best_agent == "agent_0", f"Expected agent_0 to have highest weight, got {best_agent} ({weights})"
    assert weights["agent_0"] > 0.25, f"Expected agent_0 weight to converge strongly, got {weights['agent_0']}"

    # Every agent must respect the 0.2/N = 0.02 diversity floor
    min_floor = 0.2 / len(agent_ids)
    for aid, w in weights.items():
        assert w >= min_floor - 1e-6, f"Agent {aid} fell below floor {min_floor}: {w}"

    # Verify isotonic regression recalibrator was used walk-forward after 100 rounds
    summary = engine.get_metrics_summary(kind="retrospective_clean", seed=42)
    assert summary["resolved_count"] == 200
    assert summary["isotonic_active"] is True
    # Calibrated swarm Brier must beat equal-weight swarm and base rate
    assert summary["swarm_brier"] < summary["equal_weight_brier"]
    assert summary["swarm_brier"] < summary["base_rate_brier"]


def test_resolver_job_matured_questions_and_kind_separation(temp_dir):
    """
    Verify ResolverJob finds matured questions, computes outcomes from prices,
    updates Brier scores and weights, writes reliability data, and never mixes
    retrospective_clean and live forecasts in one Brier number (Design Decision 8).
    """
    price_loader = PriceLoader()
    dates = pd.date_range("2023-11-01", periods=20, freq="B", tz=EASTERN)

    # Stock AAPL drops from 100 to 90 (-10%), Sector ETF XLK rises from 100 to 105 (+5%)
    # -> Excess return is negative (-15%), so event "excess return is negative" is True (outcome = 1)
    df_aapl = pd.DataFrame(
        {
            "Open": np.linspace(100.0, 90.0, 20),
            "Close": np.linspace(100.0, 90.0, 20),
            "Adj Close": np.linspace(100.0, 90.0, 20),
        },
        index=dates,
    )
    df_xlk = pd.DataFrame(
        {
            "Open": np.linspace(100.0, 105.0, 20),
            "Close": np.linspace(100.0, 105.0, 20),
            "Adj Close": np.linspace(100.0, 105.0, 20),
        },
        index=dates,
    )
    price_loader.register_price_series("AAPL", df_aapl)
    price_loader.register_price_series("XLK", df_xlk)

    q_retro = Question(
        question_id="q_retro_1",
        ticker="AAPL",
        event_def="5-trading-day excess return vs sector ETF is negative",
        horizon_days=5,
        resolves_at=datetime(2023, 11, 15, 16, 0, tzinfo=EASTERN),
        kind="retrospective_clean",
    )
    q_live_pending = Question(
        question_id="q_live_1",
        ticker="AAPL",
        event_def="5-trading-day excess return vs sector ETF is negative",
        horizon_days=5,
        resolves_at=datetime(2025, 12, 31, 16, 0, tzinfo=EASTERN),
        kind="live",
    )

    ledger_path = temp_dir / "forecasts.jsonl"
    ledger = ForecastLedger(ledger_path=ledger_path)
    made_dt = datetime(2023, 11, 3, 18, 30, tzinfo=EASTERN)

    for aid, p_val in [("agent_1", 0.80), ("agent_2", 0.40)]:
        ledger.append_forecast(
            question_id="q_retro_1",
            agent_id=aid,
            model="nvidia/nemotron-3-nano-30b-a3b",
            p_yes=p_val,
            rationale="Test rationale",
            evidence_ids=["AAPL_2023"],
            made_at=made_dt,
        )
        ledger.append_forecast(
            question_id="q_live_1",
            agent_id=aid,
            model="nvidia/nemotron-3-nano-30b-a3b",
            p_yes=p_val,
            rationale="Live rationale",
            evidence_ids=["AAPL_2023"],
            made_at=made_dt,
        )

    cal_engine = CalibrationEngine(agent_ids=["agent_1", "agent_2"], eta=3.0)
    reliability_out = temp_dir / "reliability.json"
    resolver = ResolverJob(
        price_loader=price_loader,
        ledger=ledger,
        calibration_engine=cal_engine,
        benchmark_ticker="XLK",
        reliability_output_path=reliability_out,
    )

    as_of = datetime(2023, 11, 20, 18, 0, tzinfo=EASTERN)
    res_report = resolver.resolve_matured(
        questions=[q_retro, q_live_pending],
        filing_timestamps={"q_retro_1": made_dt, "q_live_1": made_dt},
        as_of=as_of,
    )

    # Only q_retro_1 matured; q_live_1 is still pending
    assert res_report["newly_resolved_count"] == 1
    assert res_report["outcomes"]["q_retro_1"] == 1
    assert "q_live_1" not in res_report["outcomes"]

    # agent_1 predicted 0.80 for outcome 1 -> lower Brier than agent_2 (0.40) -> higher weight
    assert cal_engine.weights["agent_1"] > cal_engine.weights["agent_2"]
    assert reliability_out.exists()

    # Check separate reporting for retrospective_clean vs live
    retro_metrics = cal_engine.get_metrics_summary(kind="retrospective_clean")
    live_metrics = cal_engine.get_metrics_summary(kind="live")
    assert retro_metrics["resolved_count"] == 1
    assert live_metrics["resolved_count"] == 0
