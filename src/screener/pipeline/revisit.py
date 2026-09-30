"""The revisit pipeline (§4 flow B, §6.6). Separate command, separate failure domain.

Not on the delivery path: its own budget, its own row, and a total probe outage cannot delay
or degrade a digest (§2, principle 8). v0 measures the T+14 rung only — but enrolment and
this rung ship at v0 because outcome labels cannot be collected retroactively.
"""

from __future__ import annotations

from datetime import datetime

import structlog

from screener.config import Settings
from screener.domain.maturity import due_rungs, grade
from screener.domain.models import Outcome, RevisitRun, RunStats
from screener.ledger import Ledger
from screener.ports import Clock, Heartbeat, ImpactProbe, Repository

log = structlog.get_logger(__name__)


async def execute_revisit(
    cfg: Settings,
    *,
    clock: Clock,
    repo: Repository,
    probe: ImpactProbe,
    heartbeat: Heartbeat,
    ledger: Ledger | None = None,
) -> RevisitRun:
    now: datetime = clock.now()
    revisit_id = repo.begin_revisit(now)
    ledger = ledger or Ledger(cfg.screener_budget_usd * 0.05)
    stats = RunStats()
    measured = 0
    missed = 0
    errors: dict[str, str] = {}

    with ledger.soft_cap(cfg.screener_budget_usd * 0.05):
        entries = repo.watchlist()
        already = repo.measured_rungs()
        due = due_rungs(entries, now, cfg.revisit, already)
        stats.stage_counts["due"] = len(due)

        for rung, entry in due:
            papers = repo.papers_by_key([(entry.arxiv_id, entry.enrolled_version)])
            paper = papers.get((entry.arxiv_id, entry.enrolled_version))
            if paper is None:
                errors[entry.arxiv_id] = "paper row missing"
                continue
            try:
                signals = await probe.measure([paper], rung, now)  # type: ignore[arg-type]
            except Exception as exc:
                errors[entry.arxiv_id] = str(exc)[:120]
                continue

            signal = signals.get(entry.arxiv_id)
            if signal is None:
                continue
            outcome: Outcome = grade(signal, cfg.outcome_scale, now, rung_days=rung)
            actual_age = (now.date() - entry.cohort_date).days
            outcome = outcome.model_copy(update={"actual_age_days": actual_age})
            repo.save_outcomes([outcome])
            if outcome.status == "measured":
                measured += 1
            else:
                missed += 1

        if not due:
            log.info("revisit.nothing_due", watchlist=len(entries))

    result = RevisitRun(
        revisit_id=revisit_id,
        run_date=now.date(),
        due=len(due),
        measured=measured,
        missed=missed,
        per_source_errors=errors,
        cost_usd=ledger.spent,
        stats=stats,
    )
    repo.finish_revisit(result)
    await heartbeat.ping(ok=not errors, detail=f"revisit due={len(due)} measured={measured}")
    log.info("revisit.done", due=len(due), measured=measured, missed=missed, errors=len(errors))
    return result
