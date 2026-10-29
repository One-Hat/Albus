"""Pydantic models mirroring contracts/types.ts exactly.

Any modification to these models must be synced with contracts/types.ts
and vice versa, validated by tests/test_contracts_drift.py.
"""
from typing import Dict, List, Literal, Optional, Tuple
from pydantic import BaseModel, Field

Source = Literal["REAL", "MOCK"]
VerdictLabel = Literal["pass", "weak", "reject"]
CalibrationKind = Literal["retrospective_clean", "live"]
StageName = Literal["ingest", "extract", "research", "swarm", "critic", "report"]
StageStatus = Literal["start", "progress", "done", "error"]


class ModelsConfig(BaseModel):
    small: str
    mid: str
    large: str


class RunMeta(BaseModel):
    id: str
    created_at: str
    source: Source
    contract_version: str
    provider: str
    models: ModelsConfig
    hypothesis: str
    universe: List[str]


class Check(BaseModel):
    name: str
    passed: bool
    evidence: str


class Verdict(BaseModel):
    label: VerdictLabel
    checks: List[Check]
    narrative: str


class EquityPoint(BaseModel):
    date: str
    value: float


class Backtest(BaseModel):
    n_events: int
    n_dropped: int
    dropped_reasons: Dict[str, int]
    k_variants: int
    sharpe: float
    sharpe_ci: Tuple[float, float]
    max_drawdown: float
    total_return: float
    turnover: float
    cost_drag: float
    equity_curve: List[EquityPoint]


class Evidence(BaseModel):
    claim_id: str
    ticker: str
    text: str
    quote: str
    accession: str
    filing_url: str
    accepted_at: str


class WeightsOverTime(BaseModel):
    round: int
    weights: Dict[str, float]


class Swarm(BaseModel):
    n_agents: int
    n_forecasts: int
    weights_over_time: List[WeightsOverTime]


class ReliabilityBin(BaseModel):
    lo: float
    hi: float
    mean_pred: float
    freq: float
    n: int


class Calibration(BaseModel):
    kind: CalibrationKind
    n_resolved: int
    brier: Dict[str, float]
    brier_ci: Dict[str, Tuple[float, float]]
    ece: float
    reliability: List[ReliabilityBin]


class Ledger(BaseModel):
    length: int
    head_hash: str
    verified: bool


class PipelineStats(BaseModel):
    filings_fetched: int
    parsed_ok: int
    quote_verify_rate: float
    repair_rate: float
    dropped_claims: int


class RunSummary(RunMeta):
    verdict: Verdict
    backtest: Backtest
    evidence: List[Evidence]
    swarm: Swarm
    calibration: List[Calibration]
    ledger: Ledger
    stats: PipelineStats


class RunIndexItem(BaseModel):
    id: str
    created_at: str
    source: Source
    hypothesis: str
    verdict: VerdictLabel


class StageEvent(BaseModel):
    stage: StageName
    status: StageStatus
    done: Optional[int] = None
    total: Optional[int] = None
    message: Optional[str] = None
