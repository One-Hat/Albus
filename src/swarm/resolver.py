"""
Resolver Job for matured forecasting questions:
Finds matured questions, computes binary outcomes from split-adjusted prices,
updates agent Brier scores and multiplicative weights, and writes reliability data to disk.
"""
from __future__ import annotations
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set
from zoneinfo import ZoneInfo

from data.prices import PriceLoader
from swarm.calibrate import CalibrationEngine
from swarm.ledger import Forecast, ForecastLedger
from swarm.questions import Question

EASTERN = ZoneInfo("America/New_York")


class ResolverJob:
    """
    Job that resolves matured questions against realized market prices,
    updates calibration weights, and persists reliability diagrams.
    """

    def __init__(
        self,
        price_loader: PriceLoader,
        ledger: ForecastLedger,
        calibration_engine: CalibrationEngine,
        benchmark_ticker: str = "XLK",
        reliability_output_path: str | Path = "data/ledger/reliability.json",
    ):
        self.price_loader = price_loader
        self.ledger = ledger
        self.calibration_engine = calibration_engine
        self.benchmark_ticker = benchmark_ticker
        self.reliability_output_path = Path(reliability_output_path)
        self.reliability_output_path.parent.mkdir(parents=True, exist_ok=True)
        self.resolved_question_ids: Set[str] = set()

    def compute_outcome(
        self,
        question: Question,
        filing_accepted_at: datetime,
    ) -> int:
        """
        Compute binary outcome (1 or 0) for a matured question from realized prices.
        Default event definition: N-trading-day excess return vs benchmark ETF is negative.
        """
        exec_open_dt, _ = self.price_loader.get_execution_open(
            question.ticker,
            filing_accepted_at,
        )
        excess_ret = self.price_loader.calculate_excess_return(
            ticker=question.ticker,
            benchmark_ticker=self.benchmark_ticker,
            start_time=exec_open_dt,
            trading_days=question.horizon_days,
        )
        if "positive" in question.event_def.lower():
            return 1 if excess_ret > 0.0 else 0
        # Default event_def: "... excess return vs sector ETF is negative"
        return 1 if excess_ret < 0.0 else 0

    def resolve_matured(
        self,
        questions: Sequence[Question],
        filing_timestamps: Optional[Dict[str, datetime]] = None,
        as_of: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Find all matured questions (resolves_at <= as_of) that have not yet been resolved,
        compute their outcomes, update Brier scores and weights, and write reliability data.
        """
        # Verify hash-chained ledger integrity before resolving
        self.ledger.verify(raise_on_error=True)

        reference_now = (as_of or datetime.now(EASTERN)).astimezone(EASTERN)
        all_forecasts = self.ledger.read_all()

        forecasts_by_q: Dict[str, List[Forecast]] = defaultdict(list)
        for fc in all_forecasts:
            forecasts_by_q[fc.question_id].append(fc)

        outcomes_map: Dict[str, int] = {}
        newly_resolved = 0

        for q in questions:
            if q.question_id in self.resolved_question_ids:
                continue

            resolves_et = q.resolves_at.astimezone(EASTERN)
            if resolves_et > reference_now:
                # Still pending
                continue

            q_forecasts = forecasts_by_q.get(q.question_id, [])
            if not q_forecasts:
                continue

            # Ensure forecasts were logged before outcome resolution
            valid_forecasts = [
                fc for fc in q_forecasts if fc.made_at.astimezone(EASTERN) < resolves_et
            ]
            if not valid_forecasts:
                continue

            filing_dt = (
                filing_timestamps.get(q.question_id)
                if filing_timestamps and q.question_id in filing_timestamps
                else min(fc.made_at for fc in valid_forecasts)
            )

            outcome = self.compute_outcome(q, filing_dt)
            agent_preds = {fc.agent_id: fc.p_yes for fc in valid_forecasts}

            self.calibration_engine.step_round(
                question_id=q.question_id,
                agent_predictions=agent_preds,
                outcome=outcome,
                kind=q.kind,
            )

            self.resolved_question_ids.add(q.question_id)
            outcomes_map[q.question_id] = outcome
            newly_resolved += 1

        reliability_payload = {
            "updated_at": reference_now.isoformat(),
            "retrospective_clean": self.calibration_engine.get_metrics_summary(
                kind="retrospective_clean"
            ),
            "live": self.calibration_engine.get_metrics_summary(kind="live"),
            "weights": dict(self.calibration_engine.weights),
        }
        with open(self.reliability_output_path, "w", encoding="utf-8") as f:
            json.dump(reliability_payload, f, indent=2)

        return {
            "newly_resolved_count": newly_resolved,
            "total_resolved_count": len(self.resolved_question_ids),
            "outcomes": outcomes_map,
            "weights": dict(self.calibration_engine.weights),
            "reliability_path": str(self.reliability_output_path),
        }
