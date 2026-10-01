"""The measured half of the rating (§6.1, §6.1.0).

Measurement drove every rule here: citations are zero at T+14 in every cohort sampled, only
10-20% of papers have a findable repo, and one cohort contained a 34,432-star repository that
belonged to a pre-existing project. Rating on signals without these guards would be worse than
not rating on them at all.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from screener.domain.models import (
    Assessment,
    Enrichment,
    OutcomeScale,
    QualityScores,
    Ranking,
    Review,
)
from screener.domain.signals import REPO_YOUNG_DAYS, anchor_score, attach, map_signals, venue_score
from tests.factories import make_paper

NOW = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)


# --- anchors ------------------------------------------------------------------------------


def test_anchor_score_clamps_below_and_above() -> None:
    anchors = {9: 100, 5: 10, 1: 0}
    assert anchor_score(anchors, -5) == 1.0
    assert anchor_score(anchors, 0) == 1.0
    assert anchor_score(anchors, 10) == 5.0
    assert anchor_score(anchors, 10_000) == 9.0, "a viral repo must not produce an unbounded score"


def test_anchor_score_interpolates_between_points() -> None:
    score = anchor_score({1: 0, 9: 100}, 50)
    assert 1.0 < score < 9.0


def test_venue_first_marker_wins_and_is_none_when_absent() -> None:
    scale = OutcomeScale(venue_markers={"neurips": 10, "accepted": 8}, venue_default=6)
    assert venue_score("NeurIPS 2026", scale) == 10.0
    assert venue_score("Accepted at Some Workshop", scale) == 8.0
    assert venue_score("Journal of Things", scale) == 6.0
    assert venue_score(None, scale) is None
    assert venue_score("", scale) is None


# --- mapping ------------------------------------------------------------------------------


def test_nothing_measured_is_absent_rather_than_zero() -> None:
    """At T+14 this is the *normal* case, not an edge case (§6.1.0)."""
    signals = map_signals(make_paper(), Enrichment(arxiv_id="x"), now=NOW, scale=OutcomeScale())
    assert signals.observable() == {}
    assert signals.stars is None and signals.citations is None


def test_a_zero_citation_count_is_absent_not_a_penalty() -> None:
    """Measured: citations are zero for every T+14 paper, so a zero carries no information.

    Scoring it 0.0 would make "not yet cited" look like "measured badly" and would drag the
    whole cohort down by the dimension's weight.
    """
    paper = make_paper(submitted_at=NOW - timedelta(days=14))
    enrichment = Enrichment(arxiv_id=paper.arxiv_id, citations=0)
    signals = map_signals(paper, enrichment, now=NOW, scale=OutcomeScale())
    assert signals.citations == 0, "the raw reading is still reported"
    assert signals.citation_signal is None, "but it does not score"
    assert "citation_signal" not in signals.observable()


def test_a_non_zero_citation_count_does_score() -> None:
    paper = make_paper(submitted_at=NOW - timedelta(days=90))
    enrichment = Enrichment(arxiv_id=paper.arxiv_id, citations=8)
    signals = map_signals(paper, enrichment, now=NOW, scale=OutcomeScale())
    assert signals.citation_signal is not None and signals.citation_signal > 0


def test_stars_map_through_the_anchors() -> None:
    paper = make_paper(submitted_at=NOW - timedelta(days=14))
    enrichment = Enrichment(arxiv_id=paper.arxiv_id, stars=60, repo_created_days_before_paper=3)
    signals = map_signals(paper, enrichment, now=NOW, scale=OutcomeScale())
    assert signals.stars == 60
    assert signals.repo_signal is not None and signals.repo_signal > 0


def test_a_repo_that_predates_the_paper_does_not_lend_it_stars() -> None:
    """The 34,432-star case: a paper linking to a pre-existing popular project."""
    paper = make_paper(submitted_at=NOW - timedelta(days=14))
    enrichment = Enrichment(
        arxiv_id=paper.arxiv_id, stars=34432, repo_created_days_before_paper=REPO_YOUNG_DAYS + 100
    )
    signals = map_signals(paper, enrichment, now=NOW, scale=OutcomeScale())
    assert signals.stars is None, "stars that predate the paper are not this paper's traction"
    assert signals.repo_signal is None


def test_age_is_recorded_on_the_signals() -> None:
    paper = make_paper(submitted_at=NOW - timedelta(days=14))
    signals = map_signals(paper, Enrichment(arxiv_id=paper.arxiv_id), now=NOW, scale=OutcomeScale())
    assert signals.age_days == 14


def test_describe_names_absence_instead_of_hiding_it() -> None:
    signals = map_signals(make_paper(), Enrichment(arxiv_id="x"), now=NOW, scale=OutcomeScale())
    text = ", ".join(signals.describe())
    assert "no repo linked" in text
    assert "citations unread" in text
    assert "no venue yet" in text


# --- attaching to a ranking ---------------------------------------------------------------


def _ranking() -> Ranking:
    scores = QualityScores(relevance=8, novelty=8, rigor=7, evidence_strength=6, reproducibility=6)
    review = Review(
        tldr="t",
        what_they_did="w",
        why_it_matters="y",
        caveats="c",
        lenses=["method"],
        tags=[],
        scores=scores,
    )
    assessment = Assessment(
        arxiv_id="2509.00001",
        run_id="r",
        stage="review",
        prompt_version="v0",  # type: ignore[arg-type]
        model="m",
        created_at=NOW,
        scores=scores,
        review=review,
    )
    return Ranking(arxiv_id="2509.00001", assessment=assessment, review=review, score=0.0)


def test_attach_records_both_halves_and_a_basis() -> None:
    from screener.domain.models import SignalScores

    signals = SignalScores(
        arxiv_id="2509.00001",
        age_days=14,
        citation_signal=0.0,
        repo_signal=5.0,
        citations=0,
        stars=9,
        sources_ok=["github"],
    )
    ranked = attach(_ranking(), signals)
    assert ranked.basis is not None
    assert ranked.basis.quality_score == pytest.approx(7.0)
    assert ranked.basis.measured_score == pytest.approx(2.5)
    assert ranked.basis.quality_weight > ranked.basis.measured_weight
    assert ranked.score > 0


def test_the_compact_basis_names_ages_and_raw_evidence() -> None:
    from screener.domain.models import SignalScores

    signals = SignalScores(
        arxiv_id="2509.00001",
        age_days=14,
        citation_signal=0.0,
        repo_signal=5.0,
        citations=0,
        stars=9,
    )
    line = attach(_ranking(), signals).basis.render_compact()  # type: ignore[union-attr]
    assert "14d" in line, "the age at which signals were read must be visible"
    assert "9★" in line, "the raw value, not just a score"
    assert "0 citations" in line
    assert "no venue yet" in line


def test_a_gated_paper_stays_gated_however_many_stars_it_has() -> None:
    from screener.domain.models import SignalScores
    from screener.domain.types import HardFlag

    ranked = _ranking().model_copy(
        update={"disposition": "gated", "hard_flag": HardFlag.PURE_SURVEY}
    )
    signals = SignalScores(arxiv_id="2509.00001", age_days=14, repo_signal=10.0, stars=9999)
    assert attach(ranked, signals).disposition == "gated"
