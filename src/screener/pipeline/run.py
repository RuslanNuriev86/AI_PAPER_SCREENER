"""The orchestrator (§11). This is the file a new engineer opens first, and it should be
readable in one screen.

The two things that must be right, and are easy to get wrong:

1. `begin_run` inserts the `runs` row **before** any write that FKs to it. The earlier design
   called `record_delivery` without ever creating a run, so the first real send would have
   violated the foreign key.
2. The budget cap is **soft**. A breach stops taking new work and still ships what is
   verified, recorded `degraded` — a raising cap cannot ship anything (§12.1).
"""

from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path

import structlog

from screener.config import Settings
from screener.domain.compose import compose as compose_digest
from screener.domain.models import (
    Digest,
    GateResult,
    Paper,
    Ranking,
    Run,
    RunStats,
    Selection,
    WatchlistEntry,
)
from screener.domain.scoring import select
from screener.domain.signals import attach, map_signals
from screener.domain.types import PaperKey, Status
from screener.ledger import Ledger
from screener.pipeline.deliver import (
    DeliveryFailed,
    deliver,
    load_outbox,
    pending_outbox,
    retire_outbox,
)
from screener.pipeline.gate import dedupe_revisions, run_gate, save_gate
from screener.pipeline.review import run_reviews
from screener.ports import (
    LLM,
    Clock,
    Enricher,
    Heartbeat,
    Notifier,
    PaperSource,
    Repository,
)

log = structlog.get_logger(__name__)


async def execute(
    cfg: Settings,
    *,
    clock: Clock,
    repo: Repository,
    source: PaperSource,
    llm: LLM,
    notifier: Notifier,
    heartbeat: Heartbeat,
    ledger: Ledger | None = None,
    enricher: Enricher | None = None,
) -> Run:
    from screener.pipeline.gate import fetch

    now = clock.now()
    outbox = Path(cfg.screener_outbox)
    run_id = repo.begin_run(
        now, cfg.config_hash(), "weekly" if cfg.screener_mode == "weekly" else "daily"
    )
    ledger = ledger or Ledger(cfg.screener_budget_usd)
    stats = RunStats()
    ranked: list[Ranking] = []
    digest: Digest | None = None
    status: Status = "ok"

    try:
        with ledger.soft_cap(cfg.screener_budget_usd):
            # --- 1. fetch, never parallelised across connections (§14) ------------------
            # A source outage is an operational condition, not a crash (§12.1). The run ends
            # with a one-line notice to the reader, because ambiguity is the failure mode
            # nobody forgives: "silence must not be ambiguous".
            try:
                papers, _, _ = await fetch(source, cfg.profile, now)
            except Exception as exc:
                detail = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
                log.error("run.source_unavailable", error_type=type(exc).__name__, error=detail)
                stats.notes.append(f"source unavailable: {detail}")
                repo.finish_run(run_id, "failed", stats, ledger.spent)
                await _notify_source_unavailable(notifier, cfg, now, detail)
                await heartbeat.ping(ok=False, detail=detail[:180])
                return Run(
                    run_id=run_id,
                    started_at=now,
                    finished_at=clock.now(),
                    status="failed",
                    mode="daily",
                    stats=stats,
                    config_hash=cfg.config_hash(),
                )
            repo.save_papers(papers)
            stats.stage_counts["fetched"] = len(papers)

            # --- retry a previously unsent digest first (§9) --------------------------
            unsent = pending_outbox(outbox)
            if unsent is not None and not cfg.dry_run:
                await _retry_outbox(unsent, notifier, heartbeat)

            # --- the seen-set decides what is genuinely new ---------------------------
            window_keys: list[PaperKey] = [p.key for p in papers]
            already = repo.seen(window_keys)
            fresh = [p for p in papers if p.key not in already]
            fresh = dedupe_revisions(fresh, repo)
            stats.stage_counts["fresh"] = len(fresh)

            if not fresh:
                status = "empty"
                log.info("run.empty", scanned=len(papers))
            else:
                # --- 2. gate (pure), then persist every decision: this IS the seen-set --
                results: list[GateResult] = run_gate(fresh, cfg.profile)
                save_gate(run_id, results, repo)
                gated_by_reason = _tally(results)
                stats.gated_by_reason = gated_by_reason
                kept = [r.paper for r in results if r.keep]
                hints = {r.paper.key: r.hint for r in results}
                stats.stage_counts["gated_kept"] = len(kept)

                # v0 has no triage cascade, so the free deterministic hint orders the review
                # tier: `review_top_k` is what bounds both latency and spend (§4, §13.1).
                # Every gate-passer is still *enrolled* below, so the un-reviewed ones land in
                # the `gate_only` band and stay visible to the false-negative audit (§6.6.1).
                shortlist = sorted(kept, key=lambda p: -hints[p.key].score)[: cfg.review_top_k]
                stats.stage_counts["shortlisted"] = len(shortlist)

                if not shortlist:
                    status = "empty"
                else:
                    # --- 3. one LLM call per shortlisted paper (v0) ---------------------
                    ranked, failures = await run_reviews(
                        shortlist,
                        hints,
                        llm,
                        model=cfg.llm_deep,
                        profile_notes=cfg.profile.notes,
                        run_id=run_id,
                        now=now,
                        ledger=ledger,
                        stats=stats,
                    )
                    # Persist the *prose* as well as the scores. `rankings` holds the score
                    # breakdown but no review text, so without this the summaries exist only in
                    # the delivered message: replay could not rebuild a digest, and the audit
                    # trail §6.5 promises would have nothing to explain. A Ranking carries its
                    # Assessment by construction, so this needs no extra plumbing.
                    # --- 3b. measured half (§6.1) -------------------------------------
                    # Enrichment runs *after* review and independently of it: the LLM never sees
                    # these values, so it cannot be influenced by a star count, and a source that
                    # fails leaves its dimension absent rather than zero (§6.5).
                    papers_by_key: dict[PaperKey, Paper] = {p.key: p for p in shortlist}
                    if enricher is not None:
                        enrichments = await enricher.enrich(shortlist)
                        ranked = [
                            attach(
                                r,
                                map_signals(
                                    papers_by_key[r.key],
                                    enrichments.get(r.arxiv_id),
                                    now=now,
                                    scale=cfg.outcome_scale,
                                ),
                            )
                            for r in ranked
                        ]
                        stats.stage_counts["enriched"] = sum(
                            1 for r in ranked if r.signals and r.signals.sources_ok
                        )
                        # Persist what was read. Rendering the basis into the digest is not
                        # enough: without this, `explain` can show the judged half and nothing
                        # else, and the §6.5 audit promise has no data behind it.
                        repo.save_enrichment([e for e in enrichments.values() if e.sources_ok], now)

                    for r in ranked:
                        repo.save_assessment(run_id, r.assessment)
                    repo.save_rankings(run_id, ranked)
                    stats.stage_counts["ranked"] = len(ranked)
                    stats.stage_counts["review_failures"] = failures
                    if failures >= cfg.screener_degrade_after_review_failures:
                        # A digest built from a fraction of the shortlist is a degraded
                        # product, and reporting `ok` would hide that (§4, §1.1).
                        status = "degraded"
                        stats.notes.append(
                            f"{failures} review failures "
                            f"(>= {cfg.screener_degrade_after_review_failures})"
                        )

                    # --- 4. deterministic selection (pure) ----------------------------
                    picks = select(ranked, cfg.selection)
                    stats.stage_counts["picked"] = len(picks)
                    below, best_below = _below_threshold(ranked, cfg.selection)

                    # --- 5. compose (pure) -------------------------------------------
                    unsent_notice = unsent is not None
                    digest = compose_digest(
                        picks,
                        papers_by_key,
                        now,
                        scanned=len(papers),
                        relevant=len(kept),
                        cost_usd=ledger.spent,
                        gated_by_reason=gated_by_reason,
                        below_threshold=below,
                        best_below=best_below,
                        unsent_notice=unsent_notice,
                    )

                    # --- 6. deliver ---------------------------------------------------
                    # Nothing cleared the bar: record it and stay quiet. A daily "0 picks"
                    # message is noise, and §12.1's rule for an empty outcome is to send
                    # nothing and log the status rather than manufacture content.
                    if not picks:
                        stats.notes.append(f"no pick cleared min_score={cfg.selection.min_score}")
                        # Degraded outranks empty: "nothing cleared the bar" and "the run was
                        # impaired" are different facts, and neither a review-failure nor a
                        # budget breach may be masked by the quieter label.
                        status = (
                            "degraded" if (ledger.exhausted or status == "degraded") else "empty"
                        )
                        repo.finish_run(run_id, status, stats, ledger.spent)
                        await heartbeat.ping(ok=True, detail=status)
                        return Run(
                            run_id=run_id,
                            started_at=now,
                            finished_at=clock.now(),
                            status=status,
                            stats=stats,
                            config_hash=cfg.config_hash(),
                            all_papers=len(papers),
                            gated=len(kept),
                            ranked=ranked,
                            digest=digest,
                        )

                    try:
                        ids = await deliver(
                            notifier, digest, now=now, dry_run=cfg.dry_run, outbox=outbox
                        )
                        repo.record_delivery(
                            run_id, picks, "digest", digest.message_map([str(i) for i in ids])
                        )
                    except DeliveryFailed:
                        status = "degraded"

                    # --- 7. enrol labels: the non-renewable asset (§2, principle 7) ----
                    repo.enrol(_enrolment(ranked, picks, kept, cfg.selection, now))

            if ledger.exhausted:
                status = "degraded"

        if ledger.exhausted and status == "ok":
            status = "degraded"

        # No summary ledger row: `runs.cost_usd` is the total, and `cost_ledger` is the
        # per-call audit trail. A zero-cost "run" entry was pure noise.
        repo.finish_run(run_id, status, stats, ledger.spent)
        await heartbeat.ping(ok=status in {"ok", "empty", "degraded"}, detail=status)

    except Exception as exc:
        repo.finish_run(run_id, "failed", stats, ledger.spent)
        await heartbeat.ping(ok=False, detail=str(exc)[:180])
        raise

    log.info(
        "run.done",
        run_id=str(run_id),
        status=status,
        spent=round(ledger.spent, 4),
        papers=len(papers),
        picks=stats.stage_counts.get("picked", 0),
    )
    return Run(
        run_id=run_id,
        started_at=now,
        finished_at=clock.now(),
        status=status,
        stats=stats,
        config_hash=cfg.config_hash(),
        all_papers=len(papers),
        gated=stats.stage_counts.get("gated_kept", 0),
        ranked=ranked,
        digest=digest,
    )


def _tally(results: list[GateResult]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in results:
        if not r.keep and r.reason is not None:
            out[r.reason.value] = out.get(r.reason.value, 0) + 1
    return out


def _below_threshold(ranked: list[Ranking], sel: Selection) -> tuple[int, float | None]:
    below = [r for r in ranked if r.disposition == "eligible" and r.score < sel.min_score]
    best = max((r.score for r in below), default=None)
    return len(below), best


def _enrolment(
    ranked: list[Ranking],
    picks: list[Ranking],
    gated: list[Paper],
    sel: Selection,
    now: datetime,
) -> list[WatchlistEntry]:
    """Enrol every gate-passing paper, stratified by score band (§6.6.1).

    Measuring only what we shipped answers "were our picks good?" and can never answer "what
    did we miss?" — which is the more expensive error. So papers that cleared the gate but
    never reached review are enrolled as `gate_only`, and their matured outcome is the only
    evidence that the funnel is dropping things it should have kept.
    """
    from screener.domain.maturity import enrol_band

    picked = {(p.arxiv_id, p.version) for p in picks}
    scored = {(r.arxiv_id, r.version) for r in ranked}
    entries: list[WatchlistEntry] = [
        WatchlistEntry(
            arxiv_id=r.arxiv_id,
            enrolled_version=r.version,
            cohort_date=now.date(),
            score_band=enrol_band(  # type: ignore[arg-type]
                r.score, delivered=(r.arxiv_id, r.version) in picked, min_score=sel.min_score
            ),
            day0_score=r.score,
            # No forecast exists any more (§5.2). What is stored is the *measured* reading
            # at the rating point, which the T+90 rung then checks for stability.
            day0_impact_forecast=(r.basis.measured_score if r.basis else None),
            delivered=(r.arxiv_id, r.version) in picked,
        )
        for r in ranked
    ]
    entries.extend(
        WatchlistEntry(
            arxiv_id=p.arxiv_id,
            enrolled_version=p.version,
            cohort_date=now.date(),
            score_band="gate_only",
            day0_score=None,
            day0_impact_forecast=None,
            delivered=False,
        )
        for p in gated
        if p.key not in scored
    )
    return entries


async def _notify_source_unavailable(
    notifier: Notifier, cfg: Settings, now: datetime, detail: str
) -> None:
    """Tell the reader the digest is missing, and why.

    §12.1 is explicit that a skipped day must not be silent — an absent digest and a broken
    pipeline look identical from the outside otherwise. Failure to notify is not fatal: the
    heartbeat still fires.
    """
    if cfg.dry_run:
        return
    message = (
        f"⚠️ <b>No digest today</b> ({now.strftime('%a %Y-%m-%d')})\n"
        f"<i>reason: source unavailable — {escape(detail[:160])}</i>"
    )
    try:
        await notifier.send([message])
    except Exception as exc:
        log.warning("run.notice_failed", error=str(exc)[:160])


async def _retry_outbox(path: Path, notifier: Notifier, heartbeat: Heartbeat) -> None:
    chunks, _items = load_outbox(path)
    try:
        await notifier.send(chunks)
    except Exception as exc:
        await heartbeat.ping(ok=False, detail=f"outbox retry failed: {exc}")
        return
    retire_outbox(path)
    log.info("deliver.outbox_retried", path=str(path))
