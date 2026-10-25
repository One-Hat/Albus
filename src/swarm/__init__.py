"""
Forecasting Swarm module (Phase 6):
Question generator, 8 Nano + 3 Super personas, and hash-chained ForecastLedger.
"""
from swarm.agents import (
    ALL_PERSONAS,
    NANO_PERSONAS,
    SUPER_PERSONAS,
    PersonaSpec,
    SwarmAgentOutput,
    SwarmRunner,
)
from swarm.ledger import (
    GENESIS_HASH,
    Forecast,
    ForecastLedger,
    LedgerTamperError,
    compute_record_hash,
)
from swarm.questions import Question, QuestionGenerator

__all__ = [
    "ALL_PERSONAS",
    "Forecast",
    "ForecastLedger",
    "GENESIS_HASH",
    "LedgerTamperError",
    "NANO_PERSONAS",
    "PersonaSpec",
    "Question",
    "QuestionGenerator",
    "SUPER_PERSONAS",
    "SwarmAgentOutput",
    "SwarmRunner",
    "compute_record_hash",
]
