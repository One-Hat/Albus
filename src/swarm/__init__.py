"""
Forecasting Swarm & Self-Calibrating Engine module (Phases 6 & 7):
Question generator, 8 Nano + 3 Super personas, hash-chained ForecastLedger,
logit-space aggregator, walk-forward isotonic calibration engine, and resolver job.
"""
from swarm.agents import (
    ALL_PERSONAS,
    NANO_PERSONAS,
    SUPER_PERSONAS,
    PersonaSpec,
    SwarmAgentOutput,
    SwarmRunner,
)
from swarm.aggregator import (
    aggregate_logit,
    logit,
    sigmoid,
    update_weights_multiplicative,
)
from swarm.calibrate import (
    CalibrationEngine,
    bootstrap_brier_ci,
    compute_brier,
    compute_brier_skill_score,
    compute_ece_and_reliability,
    compute_log_loss,
)
from swarm.ledger import (
    GENESIS_HASH,
    Forecast,
    ForecastLedger,
    LedgerTamperError,
    compute_record_hash,
)
from swarm.questions import Question, QuestionGenerator
from swarm.resolver import ResolverJob

__all__ = [
    "ALL_PERSONAS",
    "CalibrationEngine",
    "Forecast",
    "ForecastLedger",
    "GENESIS_HASH",
    "LedgerTamperError",
    "NANO_PERSONAS",
    "PersonaSpec",
    "Question",
    "QuestionGenerator",
    "ResolverJob",
    "SUPER_PERSONAS",
    "SwarmAgentOutput",
    "SwarmRunner",
    "aggregate_logit",
    "bootstrap_brier_ci",
    "compute_brier",
    "compute_brier_skill_score",
    "compute_ece_and_reliability",
    "compute_log_loss",
    "compute_record_hash",
    "logit",
    "sigmoid",
    "update_weights_multiplicative",
]
