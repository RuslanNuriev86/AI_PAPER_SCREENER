"""The budget cap and the end-to-end run.

The ledger tests pin the behaviour the design changed: the cap is **soft**, so a breach stops
spending and still ships. The e2e test drives the real orchestrator against fakes, which is
what proves the contract is wired up rather than merely written down.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import pytest

from screener.config import Settings
from screener.domain.models import (
    LedgerEntry,
    OutcomeSignal,
    Paper,
    Profile,
    Prompt,
    RevisitConfig,
    Selection,
)
from screener.domain.types import MessageId, Rung
from screener.ledger import Ledger, estimate
from screener.pipeline.gate import dedupe_revisions
from screener.pipeline.revisit import execute_revisit
from screener.pipeline.run import execute as run_pipeline
from tests.factories import make_paper

NOW = datetime(2025, 10, 1, 12, 0, tzinfo=UTC)


# --- ledger --------------------------------------------------------------------------------


def test_soft_cap_does_not_raise_on_breach() -> None:
    """A raising cap cannot ship a partial digest, which §12.1 requires."""
    ledger = Ledger(cap_usd=0.10)
    with ledger.soft_cap(0.10):
        ledger.charge(LedgerEntry(stage="review", model="gpt-4o", usd=0.50))
    assert ledger.exhausted
    assert ledger.status == "degraded"
    assert ledger.spent == pytest.approx(0.50)


def test_affordable_is_false_when_the_next_call_would_breach() -> None:
    ledger = Ledger(cap_usd=0.10)
    assert ledger.affordable(0.05)
    ledger.charge(LedgerEntry(stage="review", model="m", usd=0.08))
    assert not ledger.affordable(0.05)
    assert ledger.exhausted, "a refused call must mark the ledger exhausted"


def test_status_is_ok_while_inside_the_cap() -> None:
    ledger = Ledger(cap_usd=1.0)
    ledger.charge(LedgerEntry(stage="review", model="m", usd=0.01))
    assert ledger.status == "ok"
    assert not ledger.exhausted


def test_estimate_varies_by_model() -> None:
    cheap = estimate("gpt-4o-mini", 1_000_000, 1_000_000)
    dear = estimate("gpt-4o", 1_000_000, 1_000_000)
    assert dear > cheap


def test_unknown_model_falls_back_to_a_default_price() -> None:
    assert estimate("some-new-model", 1000, 1000) > 0


# --- fakes ---------------------------------------------------------------------------------


class FakeSource:
    def __init__(self, papers: list[Paper]) -> None:
        self.papers = papers
        self.calls = 0

    async def fetch(self, since: datetime, until: datetime, profile: Profile) -> list[Paper]:
        self.calls += 1
        return list(self.papers)


class FakeNotifier:
    def __init__(self, *, fail: bool = False) -> None:
        self.sent: list[Sequence[str]] = []
        self.fail = fail

    async def send(self, chunks: Sequence[str]) -> list[MessageId]:
        if self.fail:
            raise RuntimeError("telegram is down")
        self.sent.append(chunks)
        return [MessageId(f"msg-{i}") for i in range(len(chunks))]


class FakeHeartbeat:
    def __init__(self) -> None:
        self.pings: list[bool] = []

    async def ping(self, *, ok: bool, detail: str = "") -> None:
        self.pings.append(ok)

    async def aclose(self) -> None:
        return None


class FakeProbe:
    def __init__(self, stars: int | None = 42) -> None:
        self.stars = stars
        self.seen: list[str] = []

    async def measure(
        self, papers: Sequence[Paper], rung: Rung, now: datetime
    ) -> Mapping[str, OutcomeSignal]:
        out: dict[str, OutcomeSignal] = {}
        for p in papers:
            self.seen.append(p.arxiv_id)
            out[p.arxiv_id] = OutcomeSignal(arxiv_id=p.arxiv_id, stars=self.stars)
        return out


class ExplodingProbe:
    """Simulates a dead probe source (GitHub quota exhausted, network down)."""

    async def measure(self, papers, rung, now):  # type: ignore[no-untyped-def]
        raise RuntimeError("github api is down")


class StubClock:
    def now(self) -> datetime:
        return NOW


class GoodLLM:
    """Returns a valid, *distinct* Review for every paper.

    The prose must differ per paper: `select()` drops near-duplicate summaries (§7), so a stub
    that returns one fixed paragraph would make every multi-pick assertion fail for a reason
    the test is not about.
    """

    def __init__(self, *, score: float = 8.0) -> None:
        self.score = score
        self.calls = 0

    async def parse[T](  # type: ignore[valid-type]
        self, *, model: str, prompt: Prompt, payload: str, schema: type[T], temperature: float = 0.0
    ) -> T:
        self.calls += 1
        from screener.domain.models import Review
        from tests.factories import review_body as _body
        from tests.factories import rubric_scores

        title = payload.split("TITLE:", 1)[-1].split("\n", 1)[0].strip()
        index = TITLES.index(title) if title in TITLES else 0
        body = _body(index)
        review = Review(
            tldr=body["tldr"],
            what_they_did=body["what_they_did"],
            why_it_matters=body["why_it_matters"],
            caveats=body["caveats"],
            lenses=["method"],
            tags=[body["tag"]],
            scores=rubric_scores(self.score),
        )
        return schema.model_validate(review.model_dump())


class FlakyLLM(GoodLLM):
    """Fails on one specific paper, to prove a per-paper failure does not kill the run."""

    def __init__(self, fail_on: str) -> None:
        super().__init__()
        self.fail_on = fail_on

    async def parse[T](  # type: ignore[valid-type]
        self, *, model: str, prompt: Prompt, payload: str, schema: type[T], temperature: float = 0.0
    ) -> T:
        if self.fail_on in payload:
            raise RuntimeError("model refused")
        return await super().parse(
            model=model, prompt=prompt, payload=payload, schema=schema, temperature=temperature
        )


#: Distinct titles per paper. `make_paper` defaults every paper to the same title, and the
#: stub LLM derives its prose from the title — so identical titles would make the
#: summary-collision guard drop all but one pick and every assertion would fail.
TITLES = [
    "Retrieval Routing for Web Navigation Agents",
    "Credit Assignment in Long-Horizon Agent Planning",
    "Memory Compaction for Multi-Session Agent Chat",
    "Sandbox Isolation for Computer-Use Agents",
    "Delegation Graphs for Multi-Agent Handoffs",
    "Tool Selection under Partial Observability",
]


def distinct_papers(n: int):
    return [make_paper(arxiv_id=f"2509.{i:05d}", title=TITLES[i % len(TITLES)]) for i in range(n)]


def _settings(**kw: object) -> Settings:
    base = Settings(
        contact_email="test@example.com",
        screener_db=":memory:",
        screener_mode="live",
        profile=Profile(
            categories=["cs.AI", "cs.CL", "cs.LG", "cs.MA"],
            strong_terms=["LLM agent", "agentic", "tool use", "trajectory", "agent benchmark"],
            weak_terms=["agent", "planning"],
            exclude_patterns=["agent-based model"],
            boost_topics=["evaluation & benchmarks", "memory & context"],
        ),
        selection=Selection(min_score=6.5, max_papers=6, per_topic_cap=2),
        revisit=RevisitConfig(rungs=[14]),
    )
    for k, v in kw.items():
        setattr(base, k, v)
    return base


# --- end to end ----------------------------------------------------------------------------


async def test_full_run_produces_a_digest_and_delivers(repo) -> None:
    cfg = _settings()
    notifier = FakeNotifier()
    heartbeat = FakeHeartbeat()
    llm = GoodLLM(score=8.0)
    papers = distinct_papers(3)

    run = await run_pipeline(
        cfg,
        clock=StubClock(),
        repo=repo,
        source=FakeSource(papers),
        llm=llm,
        notifier=notifier,
        heartbeat=heartbeat,
    )

    assert run.status == "ok"
    assert run.digest is not None
    assert len(run.digest.items) == 3
    assert notifier.sent, "the digest should have been delivered"
    assert heartbeat.pings == [True]


async def test_run_enrols_every_gate_passing_paper_with_a_band(repo) -> None:
    """Enrolment is the non-renewable asset, so it must happen on the very first run."""
    cfg = _settings()
    papers = distinct_papers(3)
    await run_pipeline(
        cfg,
        clock=StubClock(),
        repo=repo,
        source=FakeSource(papers),
        llm=GoodLLM(score=8.0),
        notifier=FakeNotifier(),
        heartbeat=FakeHeartbeat(),
    )
    entries = repo.watchlist()
    assert len(entries) == 3
    assert {e.score_band for e in entries} == {"delivered"}
    assert all(e.day0_impact_forecast == 8.0 for e in entries)


async def test_second_run_of_the_same_window_is_idempotent(repo) -> None:
    """The seen-set is v0's only state, and this is the property it exists for."""
    cfg = _settings()
    papers = distinct_papers(3)
    source, llm = FakeSource(papers), GoodLLM(score=8.0)

    first = await run_pipeline(
        cfg,
        clock=StubClock(),
        repo=repo,
        source=source,
        llm=llm,
        notifier=FakeNotifier(),
        heartbeat=FakeHeartbeat(),
    )
    assert first.status == "ok"
    calls_after_first = llm.calls

    second = await run_pipeline(
        cfg,
        clock=StubClock(),
        repo=repo,
        source=source,
        llm=llm,
        notifier=FakeNotifier(),
        heartbeat=FakeHeartbeat(),
    )
    assert second.status == "empty", "nothing is fresh on the second pass"
    assert llm.calls == calls_after_first, "no paper may be re-reviewed"
    assert second.digest is None


async def test_empty_window_is_reported_not_faked(repo) -> None:
    cfg = _settings()
    run = await run_pipeline(
        cfg,
        clock=StubClock(),
        repo=repo,
        source=FakeSource([]),
        llm=GoodLLM(score=8.0),
        notifier=FakeNotifier(),
        heartbeat=FakeHeartbeat(),
    )
    assert run.status == "empty"
    assert run.digest is None


async def test_all_papers_gated_leaves_no_digest_but_still_records_the_run(repo) -> None:
    cfg = _settings()
    off_topic = [make_paper(arxiv_id="2509.77777", categories=["astro-ph.GA"])]
    run = await run_pipeline(
        cfg,
        clock=StubClock(),
        repo=repo,
        source=FakeSource(off_topic),
        llm=GoodLLM(score=8.0),
        notifier=FakeNotifier(),
        heartbeat=FakeHeartbeat(),
    )
    assert run.status == "empty"
    assert repo.counts()["gate_results"] == 1
    assert repo.counts()["runs"] == 1


async def test_a_single_llm_failure_drops_one_paper_and_keeps_the_rest(repo) -> None:
    cfg = _settings()
    papers = distinct_papers(3)
    run = await run_pipeline(
        cfg,
        clock=StubClock(),
        repo=repo,
        source=FakeSource(papers),
        llm=FlakyLLM(fail_on="2509.00001"),
        notifier=FakeNotifier(),
        heartbeat=FakeHeartbeat(),
    )
    assert run.stats.stage_counts["review_failures"] == 1
    assert len(run.digest.items) == 2  # type: ignore[union-attr]


class AlwaysFailingLLM:
    async def parse(self, **kw):  # type: ignore[no-untyped-def]
        raise RuntimeError("provider is down")


async def test_mass_review_failure_marks_the_run_degraded_not_ok(repo) -> None:
    """§4: >=3 per-paper failures is a degraded run.

    Without this, a run in which *every* review failed still reported `ok` — the exact
    silent-degradation mode §1.1 warns about.
    """
    cfg = _settings()
    papers = distinct_papers(4)
    run = await run_pipeline(
        cfg,
        clock=StubClock(),
        repo=repo,
        source=FakeSource(papers),
        llm=AlwaysFailingLLM(),
        notifier=FakeNotifier(),
        heartbeat=FakeHeartbeat(),
    )
    assert run.stats.stage_counts["review_failures"] == 4
    assert run.status == "degraded"
    assert run.status != "ok"


async def test_a_couple_of_review_failures_is_still_ok(repo) -> None:
    """Below the threshold the digest is not materially affected, so it stays `ok`."""
    cfg = _settings()
    cfg.screener_degrade_after_review_failures = 3
    papers = distinct_papers(4)
    run = await run_pipeline(
        cfg,
        clock=StubClock(),
        repo=repo,
        source=FakeSource(papers),
        llm=FlakyLLM(fail_on="2509.00001"),
        notifier=FakeNotifier(),
        heartbeat=FakeHeartbeat(),
    )
    assert run.stats.stage_counts["review_failures"] == 1
    assert run.status == "ok"


async def test_budget_exhaustion_ships_a_partial_digest_and_marks_degraded(repo) -> None:
    """The core §12.1 requirement: overspend is refused, silence is not an option."""
    cfg = _settings()
    cfg.screener_budget_usd = 0.0001  # far below one call
    papers = distinct_papers(4)
    run = await run_pipeline(
        cfg,
        clock=StubClock(),
        repo=repo,
        source=FakeSource(papers),
        llm=GoodLLM(score=8.0),
        notifier=FakeNotifier(),
        heartbeat=FakeHeartbeat(),
        ledger=Ledger(0.0001),
    )
    assert run.status == "degraded"
    assert run.stats.stage_counts["ranked"] == 0
    assert run.status != "failed", "a budget breach is degradation, not failure"


async def test_telegram_failure_degrades_and_writes_the_outbox(repo, tmp_path, monkeypatch) -> None:
    """The digest must survive a dead notifier: outbox + heartbeat, never silence."""
    monkeypatch.chdir(tmp_path)
    cfg = _settings()
    run = await run_pipeline(
        cfg,
        clock=StubClock(),
        repo=repo,
        source=FakeSource([make_paper()]),
        llm=GoodLLM(score=8.0),
        notifier=FakeNotifier(fail=True),
        heartbeat=FakeHeartbeat(),
    )
    assert run.status == "degraded"
    outbox = tmp_path / "outbox"
    assert outbox.exists() and list(outbox.glob("*.html")), "the digest must be recoverable"


async def test_dedupe_revisions_drops_an_already_delivered_version(repo) -> None:
    paper = make_paper()
    repo.save_papers([paper])
    run_id = repo.begin_run(NOW, "h", "daily")
    from screener.domain.models import Assessment, Ranking, Review, Scores

    review = Review(
        tldr="x",
        what_they_did="y",
        why_it_matters="z",
        caveats="w",
        scores=Scores(
            relevance=8,
            novelty=8,
            rigor=8,
            evidence_strength=8,
            impact_forecast=8,
            reproducibility=8,
        ),
    )
    repo.record_delivery(
        run_id,
        [
            Ranking(
                arxiv_id=paper.arxiv_id,
                assessment=Assessment(
                    arxiv_id=paper.arxiv_id,
                    run_id=run_id,
                    stage="review",
                    prompt_version="v0",
                    model="m",
                    created_at=NOW,
                    scores=review.scores,
                    review=review,
                ),
                review=review,
                score=8.0,
            )
        ],
        "digest",
        {paper.arxiv_id: "msg-1"},
    )
    assert dedupe_revisions([paper], repo) == []


# --- revisit -------------------------------------------------------------------------------


async def test_revisit_measures_a_due_rung_and_grades_it(repo) -> None:
    from datetime import timedelta

    cfg = _settings()
    paper = make_paper()
    repo.save_papers([paper])
    from screener.domain.models import WatchlistEntry

    repo.enrol(
        [
            WatchlistEntry(
                arxiv_id=paper.arxiv_id,
                enrolled_version=1,
                cohort_date=(NOW - timedelta(days=15)).date(),
                score_band="mid",
                day0_score=5.5,
                day0_impact_forecast=6.0,
            )
        ]
    )
    probe = FakeProbe(stars=200)
    result = await execute_revisit(
        cfg, clock=StubClock(), repo=repo, probe=probe, heartbeat=FakeHeartbeat()
    )
    assert result.due == 1
    assert result.measured == 1
    assert probe.seen == [paper.arxiv_id]
    row = repo._conn.execute("SELECT * FROM outcomes").fetchone()
    assert row["status"] == "measured"
    assert row["matured_impact"] is not None


async def test_revisit_is_idempotent_per_rung(repo) -> None:
    from datetime import timedelta

    cfg = _settings()
    paper = make_paper()
    repo.save_papers([paper])
    from screener.domain.models import WatchlistEntry

    repo.enrol(
        [
            WatchlistEntry(
                arxiv_id=paper.arxiv_id,
                enrolled_version=1,
                cohort_date=(NOW - timedelta(days=15)).date(),
                score_band="mid",
                day0_score=5.5,
            )
        ]
    )
    probe = FakeProbe()
    first = await execute_revisit(
        cfg, clock=StubClock(), repo=repo, probe=probe, heartbeat=FakeHeartbeat()
    )
    second = await execute_revisit(
        cfg, clock=StubClock(), repo=repo, probe=probe, heartbeat=FakeHeartbeat()
    )
    assert first.measured == 1
    assert second.due == 0
    assert len(probe.seen) == 1, "the same rung must not be measured twice"


async def test_revisit_records_a_probe_outage_instead_of_crashing(repo) -> None:
    from datetime import timedelta

    cfg = _settings()
    paper = make_paper()
    repo.save_papers([paper])
    from screener.domain.models import WatchlistEntry

    repo.enrol(
        [
            WatchlistEntry(
                arxiv_id=paper.arxiv_id,
                enrolled_version=paper.version,
                cohort_date=(NOW - timedelta(days=15)).date(),
                score_band="mid",
                day0_score=5.5,
            )
        ]
    )
    result = await execute_revisit(
        cfg, clock=StubClock(), repo=repo, probe=ExplodingProbe(), heartbeat=FakeHeartbeat()
    )
    assert result.due == 1
    assert result.measured == 0
    assert result.per_source_errors


async def test_the_configured_outbox_is_used_not_the_working_directory(repo, tmp_path) -> None:
    """The outbox path comes from config.

    Without this, a pipeline test resolved `./outbox` relative to the repo and consumed a real
    stranded digest. Isolation is not a nicety here; it lost data once.
    """
    cfg = _settings()
    cfg.screener_outbox = str(tmp_path / "custom-outbox")
    run = await run_pipeline(
        cfg,
        clock=StubClock(),
        repo=repo,
        source=FakeSource([make_paper()]),
        llm=GoodLLM(score=8.0),
        notifier=FakeNotifier(fail=True),
        heartbeat=FakeHeartbeat(),
    )
    assert run.status == "degraded"
    written = list((tmp_path / "custom-outbox").glob("*"))
    assert written, "the undeliverable digest must land in the configured outbox"
    assert not Path("outbox").exists() or not list(Path("outbox").glob("*.html"))


async def test_a_stranded_digest_is_retried_and_retired(repo, tmp_path) -> None:
    """The recovery path end to end: park a digest, then let the next run deliver it."""
    from screener.pipeline.deliver import load_outbox, pending_outbox

    cfg = _settings()
    cfg.screener_outbox = str(tmp_path / "outbox")
    outbox = tmp_path / "outbox"

    first = await run_pipeline(
        cfg,
        clock=StubClock(),
        repo=repo,
        source=FakeSource([make_paper()]),
        llm=GoodLLM(score=8.0),
        notifier=FakeNotifier(fail=True),
        heartbeat=FakeHeartbeat(),
    )
    assert first.status == "degraded"
    parked = pending_outbox(outbox)
    assert parked is not None, "the failed digest must be parked"
    chunks, _ = load_outbox(parked)
    assert chunks and any("Agent Papers" in c for c in chunks)

    # Next run: Telegram works again. The paper is already seen, so nothing fresh — but the
    # parked digest must still be delivered.
    notifier = FakeNotifier()
    second = await run_pipeline(
        cfg,
        clock=StubClock(),
        repo=repo,
        source=FakeSource([make_paper()]),
        llm=GoodLLM(score=8.0),
        notifier=notifier,
        heartbeat=FakeHeartbeat(),
    )
    assert second.status == "empty"
    assert notifier.sent, "the parked digest should have been re-sent"
    assert pending_outbox(outbox) is None, "and retired once delivered"


async def test_review_prose_is_persisted_so_a_digest_can_be_rebuilt(repo) -> None:
    """The bug this catches: `save_assessment` was never called.

    `rankings` stores the score breakdown but no review text, so without an `assessments` row the
    summaries exist only in the delivered message. Replay then rebuilds nothing, and §6.5's
    audit trail has nothing to explain — which is exactly what happened on the first real
    digest, where recovery was impossible.
    """
    from screener.pipeline.replay import replay_run

    cfg = _settings()
    # replay opens its own connection, so it must be pointed at the fixture's file rather than
    # the in-memory default.
    cfg.screener_db = repo.path
    await run_pipeline(
        cfg,
        clock=StubClock(),
        repo=repo,
        source=FakeSource(distinct_papers(3)),
        llm=GoodLLM(score=8.0),
        notifier=FakeNotifier(),
        heartbeat=FakeHeartbeat(),
    )

    stored = repo._conn.execute("SELECT COUNT(*) c FROM assessments").fetchone()["c"]
    assert stored > 0, "the review prose must be persisted, not just the scores"

    result = replay_run(cfg, NOW.date().isoformat())
    assert result.ranked, "replay must be able to rebuild the digest from stored data"
    rebuilt = result.ranked[0].review
    assert rebuilt.tldr, "the review text must survive the round trip"
    assert rebuilt.scores.impact_forecast is not None


async def test_rearm_makes_only_the_shortlisted_papers_fresh_again(repo) -> None:
    """Recovery for a reviewed-but-undelivered digest whose outbox entry was lost."""
    from screener.pipeline.replay import rearm

    cfg = _settings()
    cfg.screener_db = repo.path
    cfg.screener_review_top_k = 2
    # The scenario this command exists for: reviewed and paid for, but the send failed, so no
    # delivery row exists and `dedupe_revisions` will not block the papers.
    await run_pipeline(
        cfg,
        clock=StubClock(),
        repo=repo,
        source=FakeSource(distinct_papers(5)),
        llm=GoodLLM(score=8.0),
        notifier=FakeNotifier(fail=True),
        heartbeat=FakeHeartbeat(),
    )
    deliveries = repo._conn.execute("SELECT COUNT(*) c FROM deliveries").fetchone()["c"]
    assert deliveries == 0, "the scenario requires an undelivered digest"
    seen_before = repo._conn.execute("SELECT COUNT(*) c FROM gate_results").fetchone()["c"]
    assert seen_before == 5

    ids = rearm(cfg, NOW.date().isoformat())
    assert len(ids) == 2, "only the reviewed papers should be re-armed, not the whole window"

    seen_after = repo._conn.execute("SELECT COUNT(*) c FROM gate_results").fetchone()["c"]
    assert seen_after == 3, "the un-reviewed papers stay seen, so this is cheap to recover"

    # And the next run finds them fresh again.
    run = await run_pipeline(
        cfg,
        clock=StubClock(),
        repo=repo,
        source=FakeSource(distinct_papers(5)),
        llm=GoodLLM(score=8.0),
        notifier=FakeNotifier(),
        heartbeat=FakeHeartbeat(),
    )
    assert run.stats.stage_counts["fresh"] == 2
    assert run.status == "ok"


def test_rearm_on_an_unknown_date_is_a_no_op(repo) -> None:
    from screener.pipeline.replay import rearm

    cfg = _settings()
    cfg.screener_db = repo.path
    assert rearm(cfg, "1999-01-01") == []


async def test_rearm_does_not_resurrect_a_paper_that_was_actually_sent(repo) -> None:
    """Recovery must not break at-most-once: a delivered paper stays delivered.

    `dedupe_revisions` blocks re-processing anything with a real `message_id`, and that check
    must survive `rearm` — otherwise "recover a failed digest" would become "re-send things the
    reader already has".
    """
    from screener.pipeline.replay import rearm

    cfg = _settings()
    cfg.screener_db = repo.path
    cfg.screener_review_top_k = 2
    await run_pipeline(
        cfg,
        clock=StubClock(),
        repo=repo,
        source=FakeSource(distinct_papers(5)),
        llm=GoodLLM(score=8.0),
        notifier=FakeNotifier(),
        heartbeat=FakeHeartbeat(),
    )
    assert repo._conn.execute("SELECT COUNT(*) c FROM deliveries").fetchone()["c"] > 0

    rearm(cfg, NOW.date().isoformat())
    run = await run_pipeline(
        cfg,
        clock=StubClock(),
        repo=repo,
        source=FakeSource(distinct_papers(5)),
        llm=GoodLLM(score=8.0),
        notifier=FakeNotifier(),
        heartbeat=FakeHeartbeat(),
    )
    assert run.stats.stage_counts["fresh"] == 0, (
        "papers with a real message_id must not be re-reviewed or re-sent"
    )
