"""Compose (§8) and the maturity loop (§6.6).

The compose tests are the ones that make the digest trustworthy: the 4096 limit is asserted
rather than hoped for, splitting never lands mid-item, and the item→chunk map that feedback
attribution depends on is actually correct.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta

import pytest

from screener.domain.compose import (
    TELEGRAM_LIMIT,
    compose,
    render_footer,
    render_item,
)
from screener.domain.maturity import (
    age_days,
    due_rungs,
    enrol_band,
    grade,
    rung_due_date,
)
from screener.domain.models import (
    Assessment,
    OutcomeScale,
    OutcomeSignal,
    Ranking,
    Review,
    RevisitConfig,
    Scores,
    WatchlistEntry,
)
from tests.factories import make_paper

NOW = datetime(2025, 10, 1, 12, 0, tzinfo=UTC)


def _ranking(i: int, *, score: float = 8.0, tldr_words: int = 12) -> Ranking:
    tldr = " ".join(f"word{j}" for j in range(tldr_words))
    review = Review(
        tldr=tldr[:220],
        what_they_did="They describe a mechanism in detail with several concrete steps.",
        why_it_matters="It makes an otherwise unmeasurable quantity measurable.",
        caveats="Limited to three web benchmarks.",
        lenses=["method"],
        tags=["evaluation & benchmarks"],
        scores=Scores(
            relevance=7,
            novelty=7,
            rigor=6,
            evidence_strength=6,
            impact_forecast=7,
            reproducibility=5,
        ),
    )
    assessment = Assessment(
        arxiv_id=f"2509.{i:05d}",
        run_id="run-1",  # type: ignore[arg-type]
        stage="review",
        prompt_version="v0",
        model="test",
        created_at=NOW,
        scores=review.scores,
        review=review,
    )
    return Ranking(
        arxiv_id=f"2509.{i:05d}",
        assessment=assessment,
        review=review,
        score=score,
        topics=["evaluation & benchmarks"],
    )


# --- compose -------------------------------------------------------------------------------


def test_item_is_rendered_with_all_required_sections() -> None:
    r = _ranking(1)
    paper = make_paper(arxiv_id=r.arxiv_id, title="A Very Specific Paper Title")
    html = render_item(r, paper, 1)
    for section in ("TL;DR", "What they did", "Why it matters", "Weak spot", "abs"):
        assert section in html
    assert "A Very Specific Paper Title" in html


def test_html_is_escaped_so_a_stray_angle_bracket_cannot_break_the_message() -> None:
    r = _ranking(1)
    r.review.tldr = "Uses <script>alert(1)</script> tokens"
    html = render_item(r, make_paper(arxiv_id=r.arxiv_id), 1)
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_no_chunk_exceeds_the_telegram_limit() -> None:
    """The 4096 limit is asserted in code, not hoped for (§8)."""
    picks = [_ranking(i, score=9.0 - i * 0.01) for i in range(12)]
    papers = {(r.arxiv_id, 1): make_paper(arxiv_id=r.arxiv_id) for r in picks}
    digest = compose(
        picks,
        papers,
        NOW,
        scanned=200,
        relevant=50,
        cost_usd=0.41,
        gated_by_reason={"pure_survey": 6},
    )
    assert digest.chunks
    for chunk in digest.chunks:
        assert len(chunk) <= TELEGRAM_LIMIT


def test_long_picks_split_across_messages_without_breaking_items() -> None:
    picks = [_ranking(i) for i in range(14)]
    papers = {(r.arxiv_id, 1): make_paper(arxiv_id=r.arxiv_id) for r in picks}
    digest = compose(picks, papers, NOW, scanned=10, relevant=10, cost_usd=0.0)
    assert len(digest.chunks) > 1, "14 items must not fit in one message"
    # Every item's arxiv_id appears in exactly one chunk, and that chunk is the mapped one.
    for item in digest.items:
        in_chunks = [i for i, c in enumerate(digest.chunks) if item.arxiv_id in c]
        assert len(in_chunks) == 1
        assert in_chunks[0] == item.chunk_index


def test_message_map_joins_chunk_ids_back_to_papers() -> None:
    """This is the mechanism that makes feedback attributable (§10)."""
    picks = [_ranking(i) for i in range(8)]
    papers = {(r.arxiv_id, 1): make_paper(arxiv_id=r.arxiv_id) for r in picks}
    digest = compose(picks, papers, NOW, scanned=10, relevant=10, cost_usd=0.0)
    ids = [f"msg-{i}" for i in range(len(digest.chunks))]
    mapping = digest.message_map(ids)
    assert len(mapping) == len(picks)
    for item in digest.items:
        assert mapping[item.arxiv_id] == ids[item.chunk_index]


def test_footer_reports_skips_including_the_best_near_miss() -> None:
    footer = render_footer({"pure_survey": 6, "not_agentic": 3}, below_threshold=4, best_below=6.1)
    assert "6 surveys" in footer
    assert "3 not agentic" in footer
    assert "best 6.1" in footer


def test_compose_requires_a_paper_and_says_so() -> None:
    with pytest.raises(KeyError):
        compose([_ranking(1)], {}, NOW, scanned=1, relevant=1, cost_usd=0.0)


def test_empty_digest_is_a_valid_outcome() -> None:
    """Zero picks is acceptable and must not crash (§1.2)."""
    digest = compose([], {}, NOW, scanned=214, relevant=38, cost_usd=0.02)
    assert len(digest.chunks) == 1
    assert "0 picks" in digest.chunks[0]


# --- maturity ------------------------------------------------------------------------------


def test_rung_due_date_is_cohort_plus_rung() -> None:
    assert (
        rung_due_date(datetime(2025, 9, 17, tzinfo=UTC).date(), 14)
        == datetime(2025, 10, 1, tzinfo=UTC).date()
    )


def _entry(arxiv_id: str, days_ago: int) -> WatchlistEntry:
    return WatchlistEntry(
        arxiv_id=arxiv_id,
        cohort_date=(NOW - timedelta(days=days_ago)).date(),
        score_band="mid",
        day0_score=5.0,
    )


def test_rung_is_not_due_before_its_age() -> None:
    cfg = RevisitConfig(rungs=[14])
    due = due_rungs([_entry("a", 13)], NOW, cfg, set())
    assert due == []


def test_rung_is_due_once_the_age_is_reached() -> None:
    cfg = RevisitConfig(rungs=[14])
    due = due_rungs([_entry("a", 14)], NOW, cfg, set())
    assert [(r, e.arxiv_id) for r, e in due] == [(14, "a")]


def test_late_rung_is_still_measured() -> None:
    """A sleeping host must measure late, not lose the label (§6.6.2)."""
    cfg = RevisitConfig(rungs=[14], max_lateness_days={14: 21})
    due = due_rungs([_entry("a", 20)], NOW, cfg, set())
    assert len(due) == 1


def test_rung_past_max_lateness_is_skipped_not_backfilled() -> None:
    """A measurement at T+45 is not a T+14 measurement, so it is refused."""
    cfg = RevisitConfig(rungs=[14], max_lateness_days={14: 21})
    due = due_rungs([_entry("a", 40)], NOW, cfg, set())
    assert due == []


def test_already_measured_rung_is_not_repeated() -> None:
    """Idempotent per (arxiv_id, rung), so re-running a day re-measures nothing."""
    cfg = RevisitConfig(rungs=[14])
    due = due_rungs([_entry("a", 14)], NOW, cfg, {("a", 14)})
    assert due == []


def test_age_days_is_exact() -> None:
    assert age_days((NOW - timedelta(days=14)).date(), NOW) == 14


def test_grade_interpolates_between_anchors() -> None:
    scale = OutcomeScale(stars={9: 1000, 5: 30, 1: 0}, hf_upvotes={}, weights={"stars": 1.0})
    mid = grade(OutcomeSignal(arxiv_id="a", stars=100), scale, NOW)
    assert mid.status == "measured"
    assert mid.matured_impact is not None
    assert 0 < mid.matured_impact < 10


def test_grade_renormalises_when_a_signal_is_absent() -> None:
    """Missing signal is not zero signal: a paper with no repo is graded on what it has."""
    scale = OutcomeScale(
        stars={9: 1000, 1: 0}, hf_upvotes={9: 200, 1: 0}, weights={"stars": 0.5, "hf_upvotes": 0.5}
    )
    only_hf = grade(OutcomeSignal(arxiv_id="a", hf_upvotes=200), scale, NOW)
    assert only_hf.status == "measured"
    assert only_hf.matured_impact == pytest.approx(9.0, abs=0.01)
    assert only_hf.components_present == ["hf_upvotes"]


def test_grade_marks_missed_when_nothing_was_measured() -> None:
    scale = OutcomeScale(stars={9: 1000, 1: 0}, hf_upvotes={}, weights={"stars": 1.0})
    outcome = grade(OutcomeSignal(arxiv_id="a"), scale, NOW)
    assert outcome.status == "missed"
    assert outcome.matured_impact is None


def test_grade_keeps_zero_stars_distinct_from_no_repo() -> None:
    """0 stars is a measurement; a missing repo is not. Conflating them would bias the labels."""
    scale = OutcomeScale(stars={9: 1000, 1: 0}, hf_upvotes={}, weights={"stars": 1.0})
    zero = grade(OutcomeSignal(arxiv_id="a", stars=0), scale, NOW)
    absent = grade(OutcomeSignal(arxiv_id="a", stars=None), scale, NOW)
    # 0 stars maps to the bottom anchor (1.0 in outcome_scale.yaml), not to nothing: the
    # distinction that matters is measured-vs-missed, because a missed rung must be
    # excluded from calibration while a zero is a real observation.
    assert zero.status == "measured"
    assert zero.matured_impact is not None and zero.matured_impact <= 1.0
    assert absent.status == "missed" and absent.matured_impact is None


def test_enrol_band_stratifies_by_outcome() -> None:
    assert enrol_band(None, delivered=False, min_score=6.5) == "gate_only"
    assert enrol_band(9.0, delivered=True, min_score=6.5) == "delivered"
    assert enrol_band(7.0, delivered=False, min_score=6.5) == "above_min"
    assert enrol_band(5.0, delivered=False, min_score=6.5) == "mid"
    assert enrol_band(1.0, delivered=False, min_score=6.5) == "low"


def test_due_rungs_is_deterministically_ordered() -> None:
    cfg = RevisitConfig(rungs=[14])
    entries = [_entry("b", 14), _entry("a", 14), _entry("c", 15)]
    due = due_rungs(entries, NOW, cfg, set())
    assert [e.arxiv_id for _, e in due] == ["a", "b", "c"]


def test_rendered_digest_has_no_unresolved_template_markers() -> None:
    r = _ranking(1)
    html = render_item(r, make_paper(arxiv_id=r.arxiv_id), 1)
    assert not re.search(r"\{[a-z_]+\}", html), "a format placeholder leaked into the output"
