"""fetch -> gate (§4 stages 1-2). Thin by design: the logic lives in the adapters and in
`domain/relevance.py`, so these functions are only sequencing and bookkeeping."""

from __future__ import annotations

from datetime import datetime, timedelta

import structlog

from screener.domain.models import GateResult, Paper, Profile
from screener.domain.relevance import gate as gate_one
from screener.domain.types import RunId
from screener.ports import PaperSource, Repository

log = structlog.get_logger(__name__)


async def fetch(
    source: PaperSource, profile: Profile, now: datetime
) -> tuple[list[Paper], datetime, datetime]:
    """Fetch the sliding window. Returns (papers, since, until)."""
    since = now - timedelta(days=profile.lookback_days)
    papers = await source.fetch(since, now, profile)
    return papers, since, now


def dedupe_revisions(papers: list[Paper], repo: Repository) -> list[Paper]:
    """Drop a cosmetically-revised paper whose base id we already delivered at this version.

    This is the one §6.2 rule that needs delivery state, and it lives here rather than in
    `gate()` precisely so the gate stays pure and exhaustively testable (§6.2).
    """
    if not papers:
        return []
    by_base: dict[str, list[Paper]] = {}
    for p in papers:
        by_base.setdefault(p.arxiv_id, []).append(p)

    delivered = repo.delivered_versions(list(by_base))
    out: list[Paper] = []
    for p in papers:
        already = delivered.get(p.arxiv_id, set())
        if already and p.version in already:
            log.info("dedupe.already_delivered", arxiv_id=p.arxiv_id, version=p.version)
            continue
        out.append(p)
    return out


def run_gate(papers: list[Paper], profile: Profile) -> list[GateResult]:
    """Pure gate over every fresh paper. Rejections carry a reason and are persisted (§6.2)."""
    results = [gate_one(p, profile) for p in papers]
    kept = sum(1 for r in results if r.keep)
    log.info("gate.done", total=len(results), kept=kept, rejected=len(results) - kept)
    return results


def save_gate(run_id: RunId, results: list[GateResult], repo: Repository) -> None:
    """Persist every gate decision. This IS the seen-set (§10)."""
    repo.save_gate_results(run_id, results)
