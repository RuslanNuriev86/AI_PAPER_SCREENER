"""Stage 5 in v0: one LLM call per paper producing the full Review (§7).

v0 deliberately has no triage/review cascade. It gates, then sends every survivor through
`llm_deep` in one call that scores and summarises together — enough to exercise the real
contracts (`Review`, `Scores`, the renormalising ranker of §6.5) without pretending to be
the full funnel. v1 replaces `run_reviews` with triage-on-many + review-on-few; `review_one`
is already the review half.
"""

from __future__ import annotations

import asyncio
from datetime import datetime

import structlog

from screener.config import read_prompt
from screener.domain.models import (
    Assessment,
    LedgerEntry,
    Paper,
    Prompt,
    Ranking,
    RelevanceHint,
    Review,
    RunStats,
)
from screener.domain.relevance import normalise_topics
from screener.domain.scoring import composite, observable_dimensions
from screener.domain.types import RunId
from screener.ledger import Ledger, estimate
from screener.ports import LLM

log = structlog.get_logger(__name__)

PROMPT_VERSION = "v0"
MAX_CONCURRENCY = 4


def _payload(paper: Paper, hint: RelevanceHint, notes: list[str]) -> str:
    """The user message. The gate hint travels as *evidence*, never as a decision (§5)."""
    authors = ", ".join(paper.authors[:12])
    note_lines = "\n".join(f"- {n}" for n in notes) or "- (none yet)"
    return (
        f"ARXIV_ID: {paper.arxiv_id}v{paper.version}\n"
        f"TITLE: {paper.title}\n"
        f"AUTHORS: {authors}\n"
        f"CATEGORIES: {', '.join(paper.categories)}\n"
        f"COMMENT: {paper.comment or '(none)'}\n"
        f"ABSTRACT:\n{paper.abstract}\n\n"
        f"[deterministic gate evidence — not a decision]\n"
        f"term_hits: {', '.join(hint.matched) or 'none'}\n"
        f"term_exclusions: {', '.join(hint.excluded) or 'none'}\n"
        f"gate_score: {hint.score}\n\n"
        f"[reader notes so far]\n{note_lines}"
    )


async def review_one(
    paper: Paper,
    hint: RelevanceHint,
    llm: LLM,
    *,
    model: str,
    prompt_body: str,
    run_id: RunId,
    now: datetime,
    notes: list[str],
) -> tuple[Ranking | None, LedgerEntry | None, str | None]:
    """One review call. Returns (ranking, ledger_entry, error).

    A failure here is a per-paper failure, not a run failure: the paper is dropped and the
    reason logged (§12.1).
    """
    prompt = Prompt(name="review", version=PROMPT_VERSION, body=prompt_body)
    payload = _payload(paper, hint, notes)

    try:
        review = await llm.parse(
            model=model, prompt=prompt, payload=payload, schema=Review, temperature=0.0
        )
    except Exception as exc:
        log.warning("review.failed", arxiv_id=paper.arxiv_id, error=str(exc))
        return None, None, str(exc)

    # Token counts are estimates until the provider reports usage; the ledger's job is to
    # enforce a ceiling, not to bill (§ledger module docstring).
    in_tokens = max(1, len(payload) // 4)
    entry = LedgerEntry(
        stage="review",
        model=model,
        input_tokens=in_tokens,
        output_tokens=700,
        usd=estimate(model, in_tokens, 700),
    )
    assessment = Assessment(
        arxiv_id=paper.arxiv_id,
        version=paper.version,
        run_id=run_id,
        stage="review",
        prompt_version=PROMPT_VERSION,
        model=model,
        created_at=now,
        cost_usd=entry.usd,
        scores=review.scores,
        review=review,
        soft_flags=list(review.soft_flags),
        hard_flag=review.hard_flag,
    )
    return _to_ranking(paper, review, assessment), entry, None


def _to_ranking(paper: Paper, review: Review, assessment: Assessment) -> Ranking:
    """§6.5: hard flags disqualify, soft flags penalise, missing dimensions renormalise."""
    observable = observable_dimensions(review.scores)
    score, components, weights = composite(observable, soft_flag_count=len(review.soft_flags))
    return Ranking(
        arxiv_id=paper.arxiv_id,
        version=paper.version,
        assessment=assessment,
        review=review,
        score=score,
        components=components,
        effective_weights=weights,
        soft_flags=list(review.soft_flags),
        disposition="gated" if review.hard_flag is not None else "eligible",
        hard_flag=review.hard_flag,
        topics=normalise_topics(review.tags),
        lab=None,  # needs enrichment (v1.5), so per_lab_cap does not apply at v0
        tags=list(review.tags),
    )


async def run_reviews(
    papers: list[Paper],
    hints: dict[tuple[str, int], RelevanceHint],
    llm: LLM,
    *,
    model: str,
    profile_notes: list[str],
    run_id: RunId,
    now: datetime,
    ledger: Ledger,
    stats: RunStats,
) -> tuple[list[Ranking], int]:
    """Review every kept paper with bounded concurrency. Returns (ranked, failures).

    The budget check happens *before* taking each paper, so a breach stops new work while
    whatever is already verified still ships — the behaviour a raising cap made impossible.
    """
    prompt_body = read_prompt("review.v0.md")
    ranked: list[Ranking] = []
    failures = 0
    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async def worker(paper: Paper) -> Ranking | None:
        nonlocal failures
        hinted = hints.get(paper.key, RelevanceHint())
        if not ledger.affordable(estimate(model, max(1, len(paper.abstract) // 3), 900)):
            log.info("review.budget_exhausted", arxiv_id=paper.arxiv_id)
            return None
        async with sem:
            ranking, entry, error = await review_one(
                paper,
                hinted,
                llm,
                model=model,
                prompt_body=prompt_body,
                run_id=run_id,
                now=now,
                notes=profile_notes,
            )
        if error is not None or ranking is None:
            failures += 1
            return None
        if entry is not None:
            ledger.charge(entry)
            stats.cost_usd = ledger.spent
        return ranking

    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(worker(p)) for p in papers]
    for t in tasks:
        result = t.result()
        if result is not None:
            ranked.append(result)

    log.info("review.done", ranked=len(ranked), failures=failures, spent=round(ledger.spent, 4))
    return ranked, failures
