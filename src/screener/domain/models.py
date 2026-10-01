"""Core records. Pure data: no I/O, no vendor imports, no behaviour beyond validation.

Mirrors DESIGN.md §6.2 (GateResult), §6.3 (TriageScores), §6.4 (Scores, Ranking,
Selection, Enrichment), §7 (Review) and §11 (the rest).
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal, Self

from pydantic import BaseModel, Field, field_validator, model_validator

from screener.domain.types import (
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
    """What the adapters actually read at the rating point (§6.1). Never inferred."""

    arxiv_id: str
    #: The paper version these signals belong to. `enrichment` is keyed including the version, so
    #: defaulting it would file a revised paper's signals against its first version.
    version: int = 1
    citations: int | None = None
    influential_citations: int | None = None
    hf_upvotes: int | None = None
    hf_daily_rank: int | None = None
    stars: int | None = None
    repo_url: str | None = None
    #: How long before the paper the repo already existed. A repo older than
    #: `signals.REPO_YOUNG_DAYS` is someone else's project, so its stars are not this paper's
    #: traction — the 34,432-star repo in the T+30 sample (§6.1.0) is exactly that case.
    repo_created_days_before_paper: int | None = None
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


class QualityScores(BaseModel):
    """The judged half of the rubric (§6.1) — the LLM's half, and only its half.

    The model never sees citation counts or star counts and cannot emit them, so a measured
    signal cannot be hallucinated into the rating. Those live in `SignalScores`, which only an
    adapter can populate.
    """

    relevance: float = Field(ge=0, le=10)
    novelty: float = Field(ge=0, le=10)
    rigor: float = Field(ge=0, le=10)
    evidence_strength: float = Field(ge=0, le=10)
    reproducibility: float = Field(ge=0, le=10)

    def observable(self) -> dict[str, float]:
        return {k: float(v) for k, v in self.model_dump().items() if v is not None}


class SignalScores(BaseModel):
    """The measured half of the rubric (§6.1), plus the raw values it came from.

    `None` means *not measured*, which is not the same as zero and is renormalised away rather
    than scored (§6.5). At T+14 that is the normal case for citations and venue: measurement
    showed zero cited papers and no venue in every cohort sampled (§6.1.0).
    """

    arxiv_id: str
    age_days: int
    citation_signal: float | None = None
    repo_signal: float | None = None
    venue_signal: float | None = None
    # Raw values, so the digest can print what was actually read rather than a score alone.
    citations: int | None = None
    stars: int | None = None
    venue: str | None = None
    repo_url: str | None = None
    sources_ok: list[str] = Field(default_factory=list)

    def observable(self) -> dict[str, float]:
        return {
            dim: float(v)
            for dim, v in (
                ("citation_signal", self.citation_signal),
                ("repo_signal", self.repo_signal),
                ("venue_signal", self.venue_signal),
            )
            if v is not None
        }

    def describe(self) -> list[str]:
        """The raw evidence, as the reader should see it — with age, and absence made visible."""
        bits: list[str] = []
        if self.stars is not None:
            bits.append(f"{self.stars}★ repo")
        elif self.repo_url:
            bits.append("repo, no stars read")
        else:
            bits.append("no repo linked")
        bits.append(
            f"{self.citations} citations" if self.citations is not None else "citations unread"
        )
        bits.append(f"{self.venue}" if self.venue else "no venue yet")
        return bits


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
    scores: QualityScores
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
    scores: QualityScores | None = None
    review: Review | None = None
    verdict: VerifyVerdict | None = None
    soft_flags: list[SoftFlag] = Field(default_factory=list)
    hard_flag: HardFlag | None = None


# --------------------------------------------------------------------------------------
# Stage 6 — rank and select
# --------------------------------------------------------------------------------------


class RatingBasis(BaseModel):
    """Why a paper scored what it scored (§7.3).

    Deliberately keeps the two halves apart. The judged half is an opinion about text; the
    measured half is a reading from an external source with a timestamp. Printing them as one
    blended number is how a rating becomes unfalsifiable, so the render methods never do.

    There is no `impact_forecast` anywhere: with the rating point at T+14 there is no predicted
    impact to disclose, which removes the whole class of "the model thinks this will be big"
    claims from the digest (§5.2).
    """

    age_days: int
    quality: QualityScores | None = None
    signals: SignalScores | None = None
    quality_score: float | None = None  # judged half, 0-10
    measured_score: float | None = None  # measured half, 0-10
    quality_weight: float = 0.0  # renormalised share actually applied
    measured_weight: float = 0.0
    components: dict[str, float] = Field(default_factory=dict)
    effective_weights: dict[str, float] = Field(default_factory=dict)
    soft_flags: list[SoftFlag] = Field(default_factory=list)
    soft_flag_penalty: float = 0.0

    def render_compact(self) -> str:
        """One line for the digest, naming both halves and the raw evidence behind the second."""
        parts = [f"{self.quality_score:.1f}q" if self.quality_score is not None else "—q"]
        parts.append(f"{self.measured_score:.1f}m" if self.measured_score is not None else "—m")
        head = " + ".join(parts)
        evidence = ", ".join(self.signals.describe()) if self.signals else "no signals read"
        tail = f" - {self.soft_flag_penalty:.1f} flags" if self.soft_flag_penalty else ""
        return f"{head} ({self.age_days}d) · {evidence}{tail}"

    def render_full(self) -> str:
        """The breakdown for `screener explain`: every dimension, weight and contribution."""
        lines = [f"age at rating: T+{self.age_days}"]
        if self.quality is not None:
            lines.append(f"judged half (weight {self.quality_weight:.2f}):")
            for dim, raw in self.quality.model_dump().items():
                weight = self.effective_weights.get(dim, 0.0)
                lines.append(f"  {dim:<18} {raw:>4.1f} x {weight:.3f} = {raw * weight:>5.2f}")
        if self.signals is not None:
            lines.append(f"measured half (weight {self.measured_weight:.2f}):")
            for dim, raw in self.signals.observable().items():
                weight = self.effective_weights.get(dim, 0.0)
                lines.append(f"  {dim:<18} {raw:>4.1f} x {weight:.3f} = {raw * weight:>5.2f}")
            lines.append("  evidence: " + ", ".join(self.signals.describe()))
            lines.append(f"  sources: {', '.join(self.signals.sources_ok) or 'none'}")
        if self.soft_flag_penalty:
            lines.append(
                f"penalty: -{self.soft_flag_penalty:.1f} for {len(self.soft_flags)} flag(s): "
                + ", ".join(f.value for f in self.soft_flags)
            )
        lines.append(f"composite: {sum(self.components.values()) - self.soft_flag_penalty:.2f}")
        return "\n".join(lines)


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
    signals: SignalScores | None = None
    basis: RatingBasis | None = None

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
    """Raw measured value -> 0-10, with written anchors (§6.1.1).

    The T+14 anchors are deliberately compressed relative to a mature-paper scale: a naive
    "50 citations is a 9" would score every T+14 paper between 0 and 1 and flatten the measured
    half to noise. Re-derive from observed cohort quantiles rather than intuition.
    """

    #: Stars at T+14. Measured sample: 7, 4, 2 across the ~10% of papers with a findable repo.
    stars: dict[int, int] = Field(
        default_factory=lambda: {10: 200, 9: 120, 8: 60, 7: 30, 6: 15, 5: 8, 4: 4, 3: 2, 1: 0}
    )
    #: Citations at T+14. Measured zero for every cohort sampled (§6.1.0); the anchors exist so
    #: the T+90 rung can carry this dimension once a source that merges preprint and published
    #: records is wired in.
    citations: dict[int, int] = Field(
        default_factory=lambda: {10: 20, 9: 12, 8: 8, 7: 5, 6: 3, 5: 2, 3: 1, 1: 0}
    )
    hf_upvotes: dict[int, int] = Field(default_factory=lambda: {9: 100, 7: 40, 5: 15, 3: 5, 1: 0})
    weights: dict[str, float] = Field(
        default_factory=lambda: {"stars": 0.55, "citations": 0.25, "venue": 0.20}
    )
    #: Substring -> score, first match wins. A stated acceptance is decisive when present, and
    #: measured absent in every T+14 cohort, so it renormalises away rather than scoring zero.
    venue_markers: dict[str, int] = Field(
        default_factory=lambda: {
            "neurips": 10,
            "nips": 10,
            "icml": 10,
            "iclr": 10,
            "cvpr": 10,
            "iccv": 10,
            "osdi": 10,
            "sosp": 10,
            "acl": 9,
            "emnlp": 9,
            "naacl": 9,
            "eccv": 9,
            "aaai": 9,
            "kdd": 9,
            "icse": 9,
            "fse": 9,
            "ijcai": 8,
            "www": 8,
            "sigir": 8,
            "accepted": 8,
            "to appear": 8,
            "journal": 7,
        }
    )
    venue_default: int = 6


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
    #: The rating point (§5.2). The cohort is the papers that have just reached this age, never
    #: day-0 papers: at T+14 the impact signals are measurable facts rather than forecasts.
    cohort_age_days: int = 14
    #: How far back the cohort window reaches, so a missed day self-heals rather than silently
    #: losing a paper's only rating opportunity.
    catch_up_days: int = 2
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
    "QualityScores",
    "Ranking",
    "RatingBasis",
    "RelevanceHint",
    "Review",
    "RevisitConfig",
    "RevisitRun",
    "Run",
    "RunStats",
    "Selection",
    "SignalScores",
    "TriageScores",
    "VerifyVerdict",
    "WatchlistEntry",
    "default_weights",
    "rubric_weight_vector",
    "truncate_prose",
]
