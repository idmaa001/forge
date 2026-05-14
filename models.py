"""
FORGE — Core data models
Shared contracts used by all four pillars.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
import uuid


# ─────────────────────────────────────────────
# Shared enums
# ─────────────────────────────────────────────

class ModelProvider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI    = "openai"
    GOOGLE    = "google"
    MISTRAL   = "mistral"
    OLLAMA    = "ollama"   # local Llama, Mistral, etc.


class Severity(str, Enum):
    LOW      = "low"
    MEDIUM   = "medium"
    HIGH     = "high"
    CRITICAL = "critical"


# ─────────────────────────────────────────────
# Pillar 1 — Evals
# ─────────────────────────────────────────────

@dataclass
class EvalItem:
    """A single prompt–expected-answer pair in an evaluation dataset."""
    prompt:   str
    expected: str
    category: str = "general"          # e.g. "hallucination", "bias", "reasoning"
    metadata: dict[str, Any] = field(default_factory=dict)
    item_id:  str = field(default_factory=lambda: str(uuid.uuid4())[:8])


@dataclass
class EvalScore:
    """Scores for one dimension of a single model response."""
    dimension:   str           # "accuracy" | "hallucination" | "bias" | "reasoning"
    score:       float         # 0.0 – 1.0
    explanation: str = ""
    passed:      bool = True


@dataclass
class EvalResult:
    """Full result for one EvalItem run against one model."""
    item_id:      str
    model:        str
    provider:     ModelProvider
    prompt:       str
    expected:     str
    actual:       str
    scores:       list[EvalScore]
    latency_ms:   float
    token_count:  int
    timestamp:    datetime = field(default_factory=datetime.utcnow)
    run_id:       str = field(default_factory=lambda: str(uuid.uuid4())[:12])

    @property
    def overall_score(self) -> float:
        if not self.scores:
            return 0.0
        return round(sum(s.score for s in self.scores) / len(self.scores), 4)

    @property
    def passed(self) -> bool:
        return all(s.passed for s in self.scores)


# ─────────────────────────────────────────────
# Pillar 2 — Observability
# ─────────────────────────────────────────────

@dataclass
class ObservationSpan:
    """One traced LLM call."""
    trace_id:     str
    span_name:    str
    model:        str
    provider:     ModelProvider
    prompt_hash:  str
    latency_ms:   float
    input_tokens: int
    output_tokens: int
    response_quality_score: float = 1.0   # rolling scorer result
    drift_flagged: bool = False
    timestamp:    datetime = field(default_factory=datetime.utcnow)


# ─────────────────────────────────────────────
# Pillar 3 — Guardrails
# ─────────────────────────────────────────────

@dataclass
class GuardEvent:
    """Fired whenever a guardrail intercepts something."""
    event_id:       str = field(default_factory=lambda: str(uuid.uuid4()))
    violation_type: str = ""         # "prompt_injection" | "pii" | "toxicity" | "policy"
    severity:       Severity = Severity.MEDIUM
    action_taken:   str = "block"    # "block" | "redact" | "warn" | "escalate"
    input_snippet:  str = ""         # first 120 chars, never full prompt
    rule_triggered: str = ""
    timestamp:      datetime = field(default_factory=datetime.utcnow)


# ─────────────────────────────────────────────
# Pillar 4 — Agent Governance
# ─────────────────────────────────────────────

@dataclass
class AgentStep:
    """One step in an agent's execution trace."""
    step_number: int
    thought:     str
    tool_called: str | None
    tool_input:  dict[str, Any] | None
    observation: str
    risk_delta:  float = 0.0


@dataclass
class AgentAuditRecord:
    """Immutable audit record for one agent run."""
    audit_id:     str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id:     str = ""
    agent_version: str = "0.0.0"
    run_id:       str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    steps:        list[AgentStep] = field(default_factory=list)
    tools_used:   list[str] = field(default_factory=list)
    risk_score:   float = 0.0          # 0–10
    escalated:    bool = False
    timestamp:    datetime = field(default_factory=datetime.utcnow)
