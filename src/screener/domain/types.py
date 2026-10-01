"""Scalar aliases, closed vocabularies and enums.

Everything downstream refers to these names, so a typo becomes a validation error rather
than a silently new category (DESIGN.md §6.4, §11).
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Literal, NewType

# --------------------------------------------------------------------------------------
# Identity
# --------------------------------------------------------------------------------------

RunId = NewType("RunId", str)
MessageId = NewType("MessageId", str)

#: The dedupe unit: arXiv's version-less base id plus the version we actually saw.
PaperKey = tuple[str, int]

#: Days after announcement at which outcomes are measured (DESIGN.md §6.6.2).
Rung = Literal[14, 90, 180]
RUNGS: tuple[Rung, ...] = (14, 90, 180)

Stage = Literal["triage", "review", "verify"]
DeliveryKind = Literal["digest", "weekly_recap", "second_look"]

#: 'empty' is a real outcome: the window produced nothing after the gate (§12.1).
Status = Literal["running", "ok", "empty", "degraded", "failed"]

#: Cohort bands for the maturity loop (§6.6.1).
ScoreBand = Literal["delivered", "above_min", "mid", "low", "gate_only"]

Lens = Literal["capability", "method", "safety", "adoption"]

# --------------------------------------------------------------------------------------
# Closed vocabularies
# --------------------------------------------------------------------------------------

#: The only subfield names in the system. Defined by profile.boost_topics (§5) and used by
#: selection's per_topic_cap, so no free-form string ever reaches the cap logic.
Topic = Literal[
    "evaluation & benchmarks",
    "multi-agent coordination",
    "memory & context",
    "computer use",
    "safety & oversight",
    "agentic RL",
    "infrastructure/protocols",
]

TOPICS: tuple[Topic, ...] = (
    "evaluation & benchmarks",
    "multi-agent coordination",
    "memory & context",
    "computer use",
    "safety & oversight",
    "agentic RL",
    "infrastructure/protocols",
)

#: Words that route a free-form tag to a Topic (§6.4).
#:
#: Explicit data rather than derived from the topic names. Deriving them does not work: the
#: generic domain words ("multi", "agent") match everything, and "agentic RL" derives *zero*
#: anchors because "agentic" is one of those generic words — so that topic could never be
#: matched at all.
#:
#: Matching: an anchor of >= 6 characters matches a tag word that shares its first 6, which
#: handles plurals ("benchmark"/"benchmarks", "protocol"/"protocols") while keeping "agent"
#: away from "agentic". Shorter anchors ("rl", "gui", "mcp") must match a whole word, so "rl"
#: does not fire on "worldly".
TOPIC_ANCHORS: dict[Topic, tuple[str, ...]] = {
    "evaluation & benchmarks": ("benchmark", "evaluation", "eval", "leaderboard", "metric"),
    "multi-agent coordination": ("coordination", "orchestration", "delegation", "handoff"),
    "memory & context": ("memory", "context", "compaction", "recall"),
    "computer use": ("computer", "browser", "desktop", "gui", "screenshot"),
    "safety & oversight": ("safety", "oversight", "misalignment", "jailbreak", "verifier"),
    "agentic RL": ("agentic", "rl", "reinforcement", "policy", "post-training"),
    "infrastructure/protocols": ("infrastructure", "protocol", "mcp", "tooling", "runtime"),
}

#: The five *judged* dimensions: the LLM reads the text and scores quality. It never sees or
#: produces a measured signal, so it cannot hallucinate a citation count (§6.1).
QualityDimension = Literal[
    "relevance",
    "novelty",
    "rigor",
    "evidence_strength",
    "reproducibility",
]

#: The three *measured* dimensions: read from external sources at T+14, never inferred (§6.1).
MeasuredDimension = Literal[
    "citation_signal",
    "repo_signal",
    "venue_signal",
]

Dimension = Literal[
    "relevance",
    "novelty",
    "rigor",
    "evidence_strength",
    "reproducibility",
    "citation_signal",
    "repo_signal",
    "venue_signal",
]


class SoftFlag(StrEnum):
    """Penalised by the composite, never disqualifying (§6.5)."""

    NO_CODE = "no_code"
    SINGLE_BASELINE = "single_baseline"
    NO_ABLATION = "no_ablation"
    NO_ERROR_BARS = "no_error_bars"
    SELF_REPORTED_ONLY = "self_reported_only"
    NARROW_BENCHMARK = "narrow_benchmark"
    OVERCLAIMED = "overclaimed"
    NO_LIMITATIONS = "no_limitations"


class HardFlag(StrEnum):
    """Disqualifying. Discovered at the gate (§6.2) or later by review, in which case it
    sets Ranking.disposition='gated'."""

    NOT_AGENTIC = "not_agentic"
    PURE_SURVEY = "pure_survey"
    NO_TECHNICAL_CONTRIBUTION = "no_technical_contribution"
    MARKETING_WHITEPAPER = "marketing_whitepaper"
    WITHDRAWN = "withdrawn"


#: Default rubric weights (§6.1). Sum must be exactly 1.0 — asserted by test.
#:
#: The measured half carries 0.45 rather than the 0.50 a clean split would give, because
#: measurement showed what is actually there at T+14 (§6.1.0): repo stars discriminate, venue is
#: rare, and citations are zero. `citation_signal` keeps a dimension but a small weight it will
#: normally renormalise away, instead of a large weight that silently contributes nothing.
RUBRIC_WEIGHTS: dict[Dimension, float] = {
    # judged: 0.55
    "relevance": 0.12,
    "novelty": 0.16,
    "rigor": 0.13,
    "evidence_strength": 0.08,
    "reproducibility": 0.06,
    # measured: 0.45
    "citation_signal": 0.08,
    "repo_signal": 0.25,
    "venue_signal": 0.12,
}

QUALITY_DIMENSIONS: tuple[QualityDimension, ...] = (
    "relevance",
    "novelty",
    "rigor",
    "evidence_strength",
    "reproducibility",
)
MEASURED_DIMENSIONS: tuple[MeasuredDimension, ...] = (
    "citation_signal",
    "repo_signal",
    "venue_signal",
)

#: Triage ordering weights: §6.1 renormalised over the four dimensions triage emits (§6.3).
TRIAGE_WEIGHTS: dict[str, float] = {
    "relevance": 0.27,
    "impact": 0.27,
    "novelty": 0.26,
    "rigor": 0.20,
}

Timestamp = datetime
Day = date
