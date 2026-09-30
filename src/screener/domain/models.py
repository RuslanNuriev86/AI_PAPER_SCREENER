"""Core records. Pure data: no I/O, no vendor imports, no behaviour beyond validation.

Mirrors DESIGN.md §6.2 (GateResult), §6.3 (TriageScores), §6.4 (Scores, Ranking,
Selection, Enrichment), §7 (Review) and §11 (the rest).
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal, Self

from pydantic import BaseModel, Field, field_validator, model_validator

from screener.domain.types import (
    ENRICHMENT_DIMENSIONS,
    RUBRIC_WEIGHTS,
    HardFlag,
    Lens,
    PaperKey,
    RunId,
    ScoreBand,
    SoftFlag,
    Stage,
    Status,
    Topic,
)

# --------------------------------------------------------------------------------------
# Stage 1 — fetch
# --------------------------------------------------------------------------------------


class Paper(BaseModel):
    """An arXiv paper at one version. `(arxiv_id, version)` is the dedupe unit (§10)."""

    arxiv_id: str
    version: int = 1
    title: str
    abstract: str
    authors: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    primary_category: str = ""
    submitted_at: datetime
    updated_at: datetime | None = None
    abs_url: str = ""
    pdf_url: str = ""
    comment: str | None = None
    code_url: str | None = None
    first_seen_at: datetime | None = None

    @property
    def key(self) -> PaperKey:
        return (self.arxiv_id, self.version)


# --------------------------------------------------------------------------------------
# Stage 2 — gate
# --------------------------------------------------------------------------------------


class RelevanceHint(BaseModel):
    """Deterministic layer-1 output (§5). Evidence handed to the LLM, never a decision."""

    score: float = 0.0
    matched: list[str] = Field(default_factory=list)
    excluded: list[str] = Field(default_factory=list)


class GateResult(BaseModel):
    """The single gate contract (§6.2). `keep` is authoritative — never truthiness."""

    paper: Paper
    keep: bool
    reason: HardFlag | None = None
    hint: RelevanceHint = Field(default_factory=RelevanceHint)

    @model_validator(mode="after")
    def _reason_iff_rejected(self) -> Self:
        if self.keep and self.reason is not None:
            raise ValueError("a kept paper cannot carry a rejection reason")
        if not self.keep and self.reason is None:
            raise ValueError("a rejected paper must carry a reason")
        return self


# --------------------------------------------------------------------------------------
# Stage 3 — enrich (v1.5; all fields optional and absence is not a negative signal)
# --------------------------------------------------------------------------------------


class Enrichment(BaseModel):
    arxiv_id: str
    citations: int | None = None
    influential_citations: int | None = None
    hf_upvotes: int | None = None
    hf_daily_rank: int | None = None
    stars: int | None = None
    institutions: list[str] | None = None
    venue: str | None = None
    sources_ok: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------------------
# Stage 4 — triage (v1)
# --------------------------------------------------------------------------------------


class TriageScores(BaseModel):
    agentic: bool
    relevance: float = Field(ge=0, le=10)
    novelty: float = Field(ge=0, le=10)
    impact: float = Field(ge=0, le=10)
    rigor: float = Field(ge=0, le=10)
    is_survey: bool = False
    soft_flags: list[SoftFlag] = Field(default_factory=list)
    hard_flag: HardFlag | None = None
    reason: str = ""

    @field_validator("reason")
    @classmethod
    def _short_reason(cls, v: str) -> str:
        return v[:140]


# --------------------------------------------------------------------------------------
# Stage 5 — review
# --------------------------------------------------------------------------------------


class Scores(BaseModel):
    """§6.1 rubric. `None` means *unobservable*, not zero — see `rank()`."""

    relevance: float = Field(ge=0, le=10)
    novelty: float = Field(ge=0, le=10)
    rigor: float = Field(ge=0, le=10)
    evidence_strength: float = Field(ge=0, le=10)
    impact_forecast: float = Field(ge=0, le=10)
    reproducibility: float = Field(ge=0, le=10)
    pedigree: float | None = None
    early_signal: float | None = None

    def observable(self) -> dict[str, float]:
        data = self.model_dump()
        return {k: float(v) for k, v in data.items() if v is not None}


def truncate_prose(value: str, limit: int) -> str:
    """Fit prose to `limit`, preferring a sentence boundary, then a word boundary.

    The length bounds in §7 exist so an item fits a Telegram message. Enforcing them by
    *rejecting* the record throws away a perfectly good summary over a few characters — which
    happened on the first live run, where `what_they_did` came back ~430 chars against a 420
    limit and the paper was dropped. Truncation enforces the same contract without losing the
    paper; semantic constraints (enums, ranges, required fields) stay hard failures.
    """
    text = " ".join(value.split())
    if len(text) <= limit:
        return text

    # `window` is exactly `limit` long, so any slice below is already bounded. Adding the
    # ellipsis must then fit *inside* the limit — appending it to a `limit`-length slice
    # returns limit + 1 and still fails max_length, which is precisely what the first live
    # run hit.
    window = text[:limit]
    ellipsis = "…"
    for boundary in (". ", "! ", "? "):
        cut = window.rfind(boundary)
        if cut > limit // 2:
            # cut <= limit - 2 for a two-character boundary, so cut + 1 <= limit - 1.
            return text[: cut + 1].strip()
    cut = window.rfind(" ")
    if cut > limit // 2:
        return text[:cut].rstrip(" ,;:")[: limit - len(ellipsis)] + ellipsis
    return text[: limit - len(ellipsis)].rstrip() + ellipsis


class Review(BaseModel):
    """The reviewed item (§7).

    Prose fields are truncated to their bound rather than rejected: a summary that is 20
    characters long is still a valid summary, and dropping it would cost a paper for cosmetic
    reasons. Everything else (enums, ranges, required fields) remains a hard validation error.
    """

    tldr: str = Field(max_length=220)
    what_they_did: str = Field(max_length=420)
    why_it_matters: str = Field(max_length=420)
    caveats: str = Field(max_length=240)
    lenses: list[Lens] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    evidence_quotes: list[str] = Field(default_factory=list)
    scores: Scores
    soft_flags: list[SoftFlag] = Field(default_factory=list)
    hard_flag: HardFlag | None = None

    @field_validator("tldr", mode="before")
    @classmethod
    def _fit_tldr(cls, v: object) -> object:
        return truncate_prose(v, 220) if isinstance(v, str) else v

    @field_validator("what_they_did", mode="before")
    @classmethod
    def _fit_what(cls, v: object) -> object:
        return truncate_prose(v, 420) if isinstance(v, str) else v

    @field_validator("why_it_matters", mode="before")
    @classmethod
    def _fit_why(cls, v: object) -> object:
        return truncate_prose(v, 420) if isinstance(v, str) else v

    @field_validator("caveats", mode="before")
    @classmethod
    def _fit_caveats(cls, v: object) -> object:
        return truncate_prose(v, 240) if isinstance(v, str) else v

    def text_blob(self) -> str:
        return " ".join([self.tldr, self.what_they_did, self.why_it_matters, self.caveats])


class VerifyVerdict(BaseModel):
    supported: bool
    unsupported_claims: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------------------
# Persistence-facing records
# --------------------------------------------------------------------------------------


class Assessment(BaseModel):
    """One `assessments` row (§10). `stage` decides which payload field is populated."""

    arxiv_id: str
    version: int = 1
    run_id: RunId
    stage: Stage
    prompt_version: str
    model: str
    created_at: datetime
    cost_usd: float = 0.0
    triage: TriageScores | None = None
    scores: Scores | None = None
    review: Review | None = None
    verdict: VerifyVerdict | None = None
    soft_flags: list[SoftFlag] = Field(default_factory=list)
    hard_flag: HardFlag | None = None


# --------------------------------------------------------------------------------------
# Stage 6 — rank and select
# --------------------------------------------------------------------------------------


class Ranking(BaseModel):
    """Emitted only by `rank()`. Cannot exist without an Assessment (§11)."""

    arxiv_id: str
    version: int = 1
    assessment: Assessment
    review: Review
    score: float
    components: dict[str, float] = Field(default_factory=dict)
    effective_weights: dict[str, float] = Field(default_factory=dict)
    soft_flags: list[SoftFlag] = Field(default_factory=list)
    disposition: Literal["eligible", "gated"] = "eligible"
    hard_flag: HardFlag | None = None
    topics: list[Topic] = Field(default_factory=list)
    lab: str | None = None
    tags: list[str] = Field(default_factory=list)

    @property
    def key(self) -> PaperKey:
        return (self.arxiv_id, self.version)


class Selection(BaseModel):
    """Selection policy. Single home for every cap (§6.4)."""

    max_papers: int = 6
    min_score: float = 6.5
    per_topic_cap: int = 2
    per_lab_cap: int = 3
    max_tag_jaccard: float = 0.60
    headline_score: float = 8.5


# --------------------------------------------------------------------------------------
# Stage 7 — compose
# --------------------------------------------------------------------------------------


class DigestItem(BaseModel):
    arxiv_id: str
    version: int = 1
    chunk_index: int


class Digest(BaseModel):
    """compose() output. `items` is the item→chunk map that makes feedback attributable."""

    chunks: list[str]
    items: list[DigestItem]
    kind: Literal["digest", "weekly_recap", "second_look"] = "digest"
    second_look: Any | None = None

    def message_map(self, ids: list[str]) -> dict[str, str]:
        """arxiv_id -> the message id of the chunk it landed in."""
        by_chunk: dict[int, str] = {}
        for idx, mid in enumerate(ids):
            by_chunk[idx] = mid
        out: dict[str, str] = {}
        for item in self.items:
            if item.chunk_index in by_chunk:
                out[item.arxiv_id] = by_chunk[item.chunk_index]
        return out


# --------------------------------------------------------------------------------------
# Run bookkeeping
# --------------------------------------------------------------------------------------


class RunStats(BaseModel):
    stage_counts: dict[str, int] = Field(default_factory=dict)
    stage_seconds: dict[str, float] = Field(default_factory=dict)
    gated_by_reason: dict[str, int] = Field(default_factory=dict)
    cost_usd: float = 0.0
    notes: list[str] = Field(default_factory=list)


class LedgerEntry(BaseModel):
    stage: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    usd: float = 0.0


class Run(BaseModel):
    run_id: RunId
    started_at: datetime
    finished_at: datetime | None = None
    status: Status = "running"
    mode: Literal["daily", "weekly"] = "daily"
    stats: RunStats = Field(default_factory=RunStats)
    config_hash: str = ""
    all_papers: int = 0
    gated: int = 0
    ranked: list[Ranking] = Field(default_factory=list)
    digest: Digest | None = None


# --------------------------------------------------------------------------------------
# Maturity loop (§6.6) — v0 ships enrolment and the T+14 rung only
# --------------------------------------------------------------------------------------


class WatchlistEntry(BaseModel):
    arxiv_id: str
    enrolled_version: int = 1
    cohort_date: date
    score_band: ScoreBand
    day0_score: float | None = None
    day0_impact_forecast: float | None = None
    delivered: bool = False


class OutcomeSignal(BaseModel):
    arxiv_id: str
    citations: int | None = None
    influential_citations: int | None = None
    stars: int | None = None
    hf_upvotes: int | None = None
    venue: str | None = None
    social_mentions: int | None = None
    revisions: int | None = None
    code_url: str | None = None
    sources_ok: list[str] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)


class Outcome(BaseModel):
    arxiv_id: str
    rung_days: int
    actual_age_days: int
    status: Literal["measured", "missed"]
    measured_at: datetime
    signal: OutcomeSignal
    matured_impact: float | None = None
    components_present: list[str] = Field(default_factory=list)


class RevisitConfig(BaseModel):
    rungs: list[int] = Field(default_factory=lambda: [14])
    max_lateness_days: dict[int, int] = Field(default_factory=lambda: {14: 21, 90: 45, 180: 90})
    cohort: dict[str, int | Literal["all"]] = Field(default_factory=dict)
    max_resurfaced_per_digest: int = 1
    probe_sources: list[str] = Field(default_factory=lambda: ["github", "hf_daily"])
    window_days: int = 90


class RevisitRun(BaseModel):
    """A `revisits` row (§10): one execution of the revisit job.

    Separate from `RevisitConfig` on purpose — the config is the *plan*, this is what
    happened. Conflating them left the execution record with nowhere to put
    `due`/`measured`/`missed`.
    """

    revisit_id: str
    run_date: date
    due: int = 0
    measured: int = 0
    missed: int = 0
    per_source_errors: dict[str, str] = Field(default_factory=dict)
    cost_usd: float = 0.0
    stats: RunStats = Field(default_factory=RunStats)


class OutcomeScale(BaseModel):
    """config/outcome_scale.yaml (§6.6.3). v0 uses the T+14 stars/upvotes mapping only."""

    stars: dict[int, int] = Field(default_factory=lambda: {9: 1000, 7: 150, 5: 30, 3: 5, 1: 0})
    hf_upvotes: dict[int, int] = Field(default_factory=lambda: {9: 200, 7: 80, 5: 30, 3: 10, 1: 0})
    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "stars": 0.5,
            "hf_upvotes": 0.3,
            "code_release": 0.1,
            "revisions": 0.1,
        }
    )


class CalibrationReport(BaseModel):
    report_date: date
    rung_days: int
    n: int = 0
    spearman: float | None = None
    mae: float | None = None
    false_negative_count: int = 0
    false_negatives: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------------------
# Config surface
# --------------------------------------------------------------------------------------


class Profile(BaseModel):
    name: str = "LLM agents & agentic systems"
    categories: list[str] = Field(default_factory=list)
    strong_terms: list[str] = Field(default_factory=list)
    weak_terms: list[str] = Field(default_factory=list)
    exclude_patterns: list[str] = Field(default_factory=list)
    boost_topics: list[Topic] = Field(default_factory=list)
    lookback_days: int = 5
    #: Minimum relevance-hint score for layer 1 to pass a paper (§5). One strong term scores
    #: 2.0, one weak term 1.0, and each exclusion subtracts 1.0 — so a default of 2.0 means
    #: "a weak term alone is not enough", which is the entire point of having two tiers.
    #: Measured on 2,639 live arXiv papers: with a 1.0 threshold, 498 of 1,035 gate-passers
    #: (48%) qualified on the single word "agent" — the exact failure mode §1.1 describes.
    min_gate_score: float = 2.0
    notes: list[str] = Field(default_factory=list)


class Prompt(BaseModel):
    """A versioned prompt file, never a string literal (§3)."""

    name: str
    version: str
    body: str

    def render(self, **kw: str) -> str:
        return self.body.format(**kw)


def rubric_weight_vector(components: dict[str, float]) -> float:
    """Sum of a component map. Exists so tests can assert 1.0 without importing dicts."""
    return float(sum(components.values()))


def default_weights() -> dict[str, float]:
    """§6.1 weights as a plain dict, for tests and diagnostics."""
    return {k: float(v) for k, v in RUBRIC_WEIGHTS.items()}


__all__ = [
    "ENRICHMENT_DIMENSIONS",
    "Assessment",
    "CalibrationReport",
    "Digest",
    "DigestItem",
    "Enrichment",
    "GateResult",
    "LedgerEntry",
    "Outcome",
    "OutcomeScale",
    "OutcomeSignal",
    "Paper",
    "Profile",
    "Prompt",
    "Ranking",
    "RelevanceHint",
    "Review",
    "RevisitConfig",
    "RevisitRun",
    "Run",
    "RunStats",
    "Scores",
    "Selection",
    "TriageScores",
    "VerifyVerdict",
    "WatchlistEntry",
    "default_weights",
    "rubric_weight_vector",
    "truncate_prose",
]
