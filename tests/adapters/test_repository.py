"""Repository invariants (§10).

The two design bugs this file exists to prevent from coming back:

* the seen-set must be keyed `(arxiv_id, version)` and must make the 5-day window idempotent
  without re-reviewing non-picks;
* a `deliveries` row must be insertable, which requires the `runs` row to exist first.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from screener.adapters.sqlite_repo import SqliteRepository
from screener.domain.models import (
    GateResult,
    QualityScores,
    Ranking,
    RelevanceHint,
    Review,
    WatchlistEntry,
)
from screener.domain.relevance import gate
from tests.factories import make_paper

NOW = datetime(2025, 10, 1, 12, 0, tzinfo=UTC)


def _review() -> Review:
    return Review(
        tldr="A distinct summary.",
        what_they_did="They did a thing.",
        why_it_matters="It matters for a reason.",
        caveats="Only one benchmark.",
        lenses=["method"],
        tags=["memory & context"],
        scores=QualityScores(
            relevance=7,
            novelty=7,
            rigor=6,
            evidence_strength=6,
            reproducibility=5,
        ),
    )


def _ranking(arxiv_id: str = "2509.18422", version: int = 1) -> Ranking:
    """Defaults to the same id `make_paper()` uses, so `deliveries` FKs resolve."""
    from screener.domain.models import Assessment

    review = _review()
    return Ranking(
        arxiv_id=arxiv_id,
        version=version,
        assessment=Assessment(
            arxiv_id=arxiv_id,
            version=version,
            run_id="r",  # type: ignore[arg-type]
            stage="review",
            prompt_version="v0",
            model="t",
            created_at=NOW,
            scores=review.scores,
            review=review,
        ),
        review=review,
        score=8.0,
    )


# --- seen-set ------------------------------------------------------------------------------


def test_seen_is_empty_before_anything_is_gated(repo: SqliteRepository) -> None:
    assert repo.seen([("2509.00001", 1)]) == set()


def test_gate_results_is_the_seen_set(repo: SqliteRepository) -> None:
    """A paper is 'seen' iff the gate has run on it — so the window is idempotent."""
    run_id = repo.begin_run(NOW, "hash", "daily")
    paper = make_paper()
    repo.save_papers([paper])
    repo.save_gate_results(run_id, [GateResult(paper=paper, keep=True, hint=RelevanceHint())])
    assert repo.seen([paper.key]) == {paper.key}


def test_seen_distinguishes_versions(repo: SqliteRepository) -> None:
    """A revision is a new (arxiv_id, version), so it can legitimately be re-gated."""
    run_id = repo.begin_run(NOW, "hash", "daily")
    v1 = make_paper(version=1)
    repo.save_papers([v1])
    repo.save_gate_results(run_id, [GateResult(paper=v1, keep=True, hint=RelevanceHint())])
    assert repo.seen([("2509.18422", 1)]) == {("2509.18422", 1)}
    assert repo.seen([("2509.18422", 2)]) == set()


def test_a_rejected_paper_is_seen_so_it_is_not_refetched_for_five_days(
    repo: SqliteRepository, profile
) -> None:
    """The expensive mistake the design calls out: re-gating and re-reviewing non-picks."""
    run_id = repo.begin_run(NOW, "hash", "daily")
    rejected = make_paper(arxiv_id="2509.55555", categories=["astro-ph.GA"])
    result = gate(rejected, profile)
    assert not result.keep
    repo.save_papers([rejected])
    repo.save_gate_results(run_id, [result])
    assert repo.seen([rejected.key]) == {rejected.key}


def test_seen_handles_more_keys_than_the_sqlite_parameter_limit(repo: SqliteRepository) -> None:
    keys = [(f"2509.{i:05d}", 1) for i in range(1200)]
    assert repo.seen(keys) == set()


# --- run lifecycle ------------------------------------------------------------------------


def test_delivery_requires_a_run_row_and_succeeds_when_it_exists(
    repo: SqliteRepository,
) -> None:
    """`record_delivery` FKs `runs(run_id)`, so `begin_run` must come first.

    The earlier design called `record_delivery` with a `date` and never created the run, which
    would have failed on the first real send.
    """
    paper = make_paper()
    repo.save_papers([paper])
    run_id = repo.begin_run(NOW, "hash", "daily")
    repo.record_delivery(run_id, [_ranking(paper.arxiv_id)], "digest", {paper.arxiv_id: "msg-1"})
    row = repo.delivery_by_message("msg-1")
    assert row == (str(run_id), paper.arxiv_id)


def test_delivery_without_a_run_row_is_rejected_by_the_foreign_key(
    repo: SqliteRepository,
) -> None:
    import sqlite3

    with pytest.raises(sqlite3.IntegrityError):
        repo.record_delivery("no-such-run", [_ranking()], "digest", {})  # type: ignore[arg-type]


def test_delivered_once_index_blocks_a_second_digest_delivery(repo: SqliteRepository) -> None:
    import sqlite3

    paper = make_paper()
    repo.save_papers([paper])
    first = repo.begin_run(NOW, "hash", "daily")
    repo.record_delivery(first, [_ranking()], "digest", {"2509.18422": "msg-1"})

    second = repo.begin_run(NOW, "hash", "daily")
    with pytest.raises(sqlite3.IntegrityError):
        # A different run cannot re-deliver the same (paper, version, kind).
        repo._conn.execute(
            "INSERT INTO deliveries (run_id, arxiv_id, version, kind, rank, score, message_id,"
            " sent_at) VALUES (?,?,?,?,?,?,?,?)",
            (second, "2509.18422", 1, "digest", 1, 8.0, "msg-2", NOW.isoformat()),
        )


def test_a_weekly_recap_may_redeliver_a_paper_the_digest_already_carried(
    repo: SqliteRepository,
) -> None:
    """The roadmap item the earlier `delivered_once ON (arxiv_id)` index made impossible."""
    paper = make_paper()
    repo.save_papers([paper])
    run_id = repo.begin_run(NOW, "hash", "daily")
    repo.record_delivery(run_id, [_ranking(paper.arxiv_id)], "digest", {paper.arxiv_id: "msg-1"})
    recap = repo.begin_run(NOW, "hash", "weekly")
    repo.record_delivery(recap, [_ranking()], "weekly_recap", {"2509.18422": "msg-2"})
    assert repo.delivery_by_message("msg-2") is not None


def test_a_revised_version_is_deliverable_again(repo: SqliteRepository) -> None:
    v1 = make_paper(version=1)
    v2 = make_paper(version=2)
    repo.save_papers([v1, v2])
    run_id = repo.begin_run(NOW, "hash", "daily")
    repo.record_delivery(run_id, [_ranking(version=1)], "digest", {"2509.18422": "msg-1"})
    later = repo.begin_run(NOW, "hash", "daily")
    repo.record_delivery(later, [_ranking(version=2)], "digest", {"2509.18422": "msg-2"})
    assert repo.delivery_by_message("msg-2") is not None


def test_begin_then_finish_run_records_status_and_cost(repo: SqliteRepository) -> None:
    from screener.domain.models import RunStats

    run_id = repo.begin_run(NOW, "hash", "daily")
    repo.finish_run(run_id, "degraded", RunStats(stage_counts={"fetched": 10}), 0.42)
    row = repo._conn.execute("SELECT * FROM runs WHERE run_id=?", (str(run_id),)).fetchone()
    assert row["status"] == "degraded"
    assert row["cost_usd"] == pytest.approx(0.42)


def test_empty_is_a_recordable_status(repo: SqliteRepository) -> None:
    """§12.1 says an empty window logs `empty` — so it must be a legal status."""
    from screener.domain.models import RunStats

    run_id = repo.begin_run(NOW, "hash", "daily")
    repo.finish_run(run_id, "empty", RunStats(), 0.0)
    row = repo._conn.execute("SELECT status FROM runs WHERE run_id=?", (str(run_id),)).fetchone()
    assert row["status"] == "empty"


# --- assessments and rankings --------------------------------------------------------------


def test_assessments_are_append_only_across_runs(repo: SqliteRepository) -> None:
    """Score history must survive a re-review; the old PK overwrote it."""
    from screener.domain.models import Assessment

    paper = make_paper()
    repo.save_papers([paper])
    for i, model in enumerate(["model-a", "model-b"]):
        run_id = repo.begin_run(NOW, "hash", "daily")
        repo.save_assessment(
            run_id,
            Assessment(
                arxiv_id=paper.arxiv_id,
                version=paper.version,
                run_id=run_id,
                stage="review",
                prompt_version="v0",
                model=model,
                created_at=NOW + timedelta(minutes=i),
                scores=_review().scores,
                review=_review(),
            ),
        )
    rows = repo._conn.execute(
        "SELECT model FROM assessments WHERE arxiv_id=?", (paper.arxiv_id,)
    ).fetchall()
    assert {r["model"] for r in rows} == {"model-a", "model-b"}


def test_saving_the_same_assessment_twice_in_one_run_is_idempotent(
    repo: SqliteRepository,
) -> None:
    from screener.domain.models import Assessment

    paper = make_paper()
    repo.save_papers([paper])
    run_id = repo.begin_run(NOW, "hash", "daily")
    a = Assessment(
        arxiv_id=paper.arxiv_id,
        version=paper.version,
        run_id=run_id,
        stage="review",
        prompt_version="v0",
        model="m",
        created_at=NOW,
        scores=_review().scores,
        review=_review(),
    )
    repo.save_assessment(run_id, a)
    repo.save_assessment(run_id, a)
    row = repo._conn.execute("SELECT COUNT(*) c FROM assessments").fetchone()
    assert row["c"] == 1


def test_rankings_store_the_component_breakdown_for_audit(repo: SqliteRepository) -> None:
    """§6.5 promises the breakdown is stored so a past score stays explainable."""
    paper = make_paper()
    repo.save_papers([paper])
    run_id = repo.begin_run(NOW, "hash", "daily")
    r = _ranking()
    r.components = {"relevance": 1.4}
    r.effective_weights = {"relevance": 0.2}
    repo.save_rankings(run_id, [r])
    row = repo._conn.execute("SELECT * FROM rankings").fetchone()
    assert '"relevance"' in row["components"]
    assert row["effective_weights"]


# --- maturity storage ----------------------------------------------------------------------


def test_outcomes_require_an_enrolled_paper(repo: SqliteRepository) -> None:
    """An outcome cannot attach to a paper that was never scored (§11)."""
    import sqlite3

    from screener.domain.models import Outcome, OutcomeSignal

    with pytest.raises(sqlite3.IntegrityError):
        repo.save_outcomes(
            [
                Outcome(
                    arxiv_id="2509.99999",
                    rung_days=14,
                    actual_age_days=14,
                    status="measured",
                    measured_at=NOW,
                    signal=OutcomeSignal(arxiv_id="2509.99999", stars=5),
                    matured_impact=3.0,
                )
            ]
        )


def test_watchlist_round_trip_and_measured_rungs(repo: SqliteRepository) -> None:
    from screener.domain.models import Outcome, OutcomeSignal

    paper = make_paper()
    repo.save_papers([paper])
    repo.enrol(
        [
            WatchlistEntry(
                arxiv_id=paper.arxiv_id,
                enrolled_version=paper.version,
                cohort_date=NOW.date(),
                score_band="mid",
                day0_score=5.5,
            )
        ]
    )
    entries = repo.watchlist()
    assert len(entries) == 1
    assert entries[0].cohort_date == NOW.date()

    repo.save_outcomes(
        [
            Outcome(
                arxiv_id=paper.arxiv_id,
                rung_days=14,
                actual_age_days=14,
                status="measured",
                measured_at=NOW,
                signal=OutcomeSignal(arxiv_id=paper.arxiv_id, stars=42),
                matured_impact=4.2,
            )
        ]
    )
    assert repo.measured_rungs() == {(paper.arxiv_id, 14)}


def test_papers_by_key_returns_only_what_was_asked_for(repo: SqliteRepository) -> None:
    a, b = make_paper(arxiv_id="2509.00001"), make_paper(arxiv_id="2509.00002")
    repo.save_papers([a, b])
    got = repo.papers_by_key([a.key])
    assert set(got) == {a.key}


def test_state_store_round_trips(repo: SqliteRepository) -> None:
    assert repo.get_state("telegram.get_updates_offset") is None
    repo.set_state("telegram.get_updates_offset", "12345")
    assert repo.get_state("telegram.get_updates_offset") == "12345"
    repo.set_state("telegram.get_updates_offset", "12346")
    assert repo.get_state("telegram.get_updates_offset") == "12346"


def test_migrate_is_idempotent(tmp_path) -> None:
    path = tmp_path / "twice.db"
    first = SqliteRepository(path)
    first.migrate()
    first.close()
    second = SqliteRepository(path)
    second.migrate()  # must not raise
    assert second.counts()["papers"] == 0
    second.close()


def test_counts_covers_the_whole_schema(repo: SqliteRepository) -> None:
    """`counts()` is schema-derived, so it cannot silently lag behind a migration.

    A hand-listed version of this method reported 8 tables while the schema had 14, which made
    `doctor` print a schema that did not exist.
    """
    counts = repo.counts()
    schema = repo._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    assert len(counts) == len(schema) == 14
    for expected in ("papers", "gate_results", "kv_state", "rankings", "outcomes", "calibration"):
        assert expected in counts
