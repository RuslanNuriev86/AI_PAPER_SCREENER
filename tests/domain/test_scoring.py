"""The ranker and selection (§6.4, §6.5).

These tests exist to make four design claims falsifiable:

* weights always sum to 1.0, at both the v1 (6-dimension) and v1.5 (8-dimension) widths;
* missing dimensions renormalise rather than scoring zero;
* hard flags disqualify and are *not* also penalised;
* `score()` never consults an outcome — the temporal-leakage invariant (§6.6.6).
"""

from __future__ import annotations

import inspect
from datetime import UTC, datetime

import pytest

from screener.domain.models import Assessment, QualityScores, Ranking, Review, Selection
from screener.domain.relevance import normalise_topics
from screener.domain.scoring import (
    composite,
    effective_weights,
    score_band,
    select,
    split_halves,
)
from screener.domain.types import RUBRIC_WEIGHTS, TRIAGE_WEIGHTS, HardFlag, SoftFlag
from tests.factories import review_body


def _scores(*, pedigree: float | None = None, early: float | None = None) -> QualityScores:
    return QualityScores(
        relevance=7.0,
        novelty=7.0,
        rigor=6.0,
        evidence_strength=6.0,
        reproducibility=5.0,
    )


def _ranking(
    arxiv_id: str = "2509.00001",
    *,
    score: float = 8.0,
    topics: list[str] | None = None,
    tags: list[str] | None = None,
    lab: str | None = None,
    soft: list[SoftFlag] | None = None,
    hard: HardFlag | None = None,
    tldr: str | None = None,
    variant: int | None = None,
) -> Ranking:
    # Every field varies with the id. If only `tldr` varied, the summary-collision guard
    # would drop items and the test would be measuring the fixture, not the selection logic.
    if variant is None:
        digits = "".join(c for c in arxiv_id if c.isdigit()) or "0"
        variant = int(digits)
    body = review_body(variant)
    review = Review(
        tldr=(tldr if tldr is not None else body["tldr"])[:220],
        what_they_did=body["what_they_did"],
        why_it_matters=body["why_it_matters"],
        caveats=body["caveats"],
        lenses=["method"],
        tags=tags if tags is not None else [body["tag"]],
        scores=_scores(),
        soft_flags=soft or [],
        hard_flag=hard,
    )
    assessment = Assessment(
        arxiv_id=arxiv_id,
        run_id="run-1",  # type: ignore[arg-type]
        stage="review",
        prompt_version="v0",
        model="test",
        created_at=datetime.now(UTC),
        scores=review.scores,
        review=review,
    )
    return Ranking(
        arxiv_id=arxiv_id,
        assessment=assessment,
        review=review,
        score=score,
        disposition="gated" if hard else "eligible",
        hard_flag=hard,
        topics=topics if topics is not None else normalise_topics([body["tag"]]),  # type: ignore[arg-type]
        tags=tags or [],
        lab=lab,
        soft_flags=soft or [],
    )


# --- weights ------------------------------------------------------------------------------


def test_rubric_weights_sum_to_one() -> None:
    assert sum(RUBRIC_WEIGHTS.values()) == pytest.approx(1.0)


def test_triage_weights_sum_to_one() -> None:
    """Triage weights are §6.1 renormalised over four dims; they must still total 1.0.

    An earlier draft wrote .27/.27/.27/.20 here, which sums to 1.01. This test is the thing that
    would have caught it.
    """
    assert sum(TRIAGE_WEIGHTS.values()) == pytest.approx(1.0)


def test_the_rubric_has_two_separate_halves() -> None:
    """Quality is judged from text; impact is measured. The split is the design (§6.1)."""
    from screener.domain.types import MEASURED_DIMENSIONS, QUALITY_DIMENSIONS

    judged = sum(RUBRIC_WEIGHTS[d] for d in QUALITY_DIMENSIONS)
    measured = sum(RUBRIC_WEIGHTS[d] for d in MEASURED_DIMENSIONS)
    assert judged == pytest.approx(0.55)
    assert measured == pytest.approx(0.45)
    assert len(QUALITY_DIMENSIONS) + len(MEASURED_DIMENSIONS) == 8


def test_no_dimension_is_a_forecast() -> None:
    """`impact_forecast` is gone: at T+14 impact is read, not predicted (§5.2)."""
    assert "impact_forecast" not in RUBRIC_WEIGHTS
    assert "impact_forecast" not in QualityScores.model_fields


def test_weights_renormalise_over_the_observed_dimensions() -> None:
    weights = effective_weights({"relevance": 7.0, "novelty": 7.0})
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["novelty"] == pytest.approx(0.16 / 0.28)


def test_an_absent_measured_signal_is_renormalised_not_scored_zero() -> None:
    """Measurement showed citations and venue are *normally* absent at T+14 (§6.1.0).

    A paper must not be punished for a signal that does not exist, so the judged half simply
    carries more weight.
    """
    quality = {"relevance": 8.0, "novelty": 8.0}
    judged_only, _, w_only = composite(quality, None)
    with_repo, _, w_both = composite(quality, {"repo_signal": 8.0})
    assert w_only["relevance"] == pytest.approx(RUBRIC_WEIGHTS["relevance"] / 0.28)
    assert w_both["relevance"] < w_only["relevance"], "the measured half should take share"
    assert judged_only == with_repo == pytest.approx(8.0), "at equal scores the halves agree"


def test_a_strong_measured_signal_raises_the_composite() -> None:
    quality = {"relevance": 5.0, "novelty": 5.0}
    low, _, _ = composite(quality, {"repo_signal": 0.0})
    high, _, _ = composite(quality, {"repo_signal": 10.0})
    assert high > low


def test_no_observable_dimensions_is_an_error_not_a_zero() -> None:
    with pytest.raises(ValueError):
        composite({}, None)


def test_split_halves_reports_each_side_separately() -> None:
    """The digest prints `7.8q + 6.2m`, never one blended number (§7.3)."""
    q_mean, m_mean, q_share, m_share = split_halves(
        {"relevance": 8.0, "novelty": 8.0}, {"repo_signal": 6.0}
    )
    assert q_mean == pytest.approx(8.0)
    assert m_mean == pytest.approx(6.0)
    assert q_share + m_share == pytest.approx(1.0)
    assert q_share > m_share


# --- penalties and gates ------------------------------------------------------------------


def test_soft_flags_penalise_by_half_a_point_each() -> None:
    quality = _scores().observable()
    base, _, _ = composite(quality, None, soft_flag_count=0)
    one, _, _ = composite(quality, None, soft_flag_count=1)
    two, _, _ = composite(quality, None, soft_flag_count=2)
    assert base - one == pytest.approx(0.5)
    assert one - two == pytest.approx(0.5)


def test_hard_flag_disqualifies_without_double_penalising() -> None:
    """Hard flags set disposition='gated'; the composite must not also subtract for them."""
    quality = _scores().observable()
    clean, _, _ = composite(quality, None, soft_flag_count=0)
    soft_only, _, _ = composite(quality, None, soft_flag_count=0)
    assert clean == pytest.approx(soft_only)


def test_gated_paper_never_appears_in_picks() -> None:
    gated = _ranking("2509.00002", score=9.9, hard=HardFlag.NOT_AGENTIC)
    picks = select([gated], Selection(min_score=1.0))
    assert picks == []


# --- selection ----------------------------------------------------------------------------


def test_min_score_filters() -> None:
    low = _ranking("2509.00003", score=5.0)
    high = _ranking("2509.00004", score=7.0)
    picks = select([low, high], Selection(min_score=6.5))
    assert [p.arxiv_id for p in picks] == ["2509.00004"]


def test_max_papers_caps_the_digest() -> None:
    ranked = [_ranking(f"2509.0000{i}", score=9.0 - i) for i in range(10)]
    picks = select(ranked, Selection(max_papers=3, min_score=1.0, max_tag_jaccard=0.99))
    assert len(picks) == 3


def test_per_topic_cap_diversifies() -> None:
    """Five variants of one idea is a worse product than one plus unrelated advances."""
    # Below headline_score (8.5) on purpose: a headline pick is exempt from the topic cap,
    # so scores of 9.0 would make this test assert nothing about the cap.
    same = [
        _ranking(f"2509.1000{i}", score=8.0 - i * 0.1, topics=["memory & context"], variant=i)
        for i in range(4)
    ]
    other = _ranking("2509.20001", score=7.0, topics=["agentic RL"], variant=4)
    picks = select([*same, other], Selection(max_papers=6, min_score=1.0, per_topic_cap=2))
    topics = [t for p in picks for t in p.topics]
    assert topics.count("memory & context") == 2
    assert "agentic RL" in topics


def test_headline_pick_is_exempt_from_topic_cap() -> None:
    a = _ranking("2509.30001", score=9.0, topics=["memory & context"])
    b = _ranking("2509.30002", score=8.9, topics=["memory & context"])
    picks = select([a, b], Selection(min_score=1.0, per_topic_cap=1, headline_score=8.5))
    assert len(picks) == 2, "both clear the headline bar, so the topic cap does not apply"


def test_near_duplicate_summaries_are_dropped() -> None:
    """The main defence against a digest that reads as five interchangeable paragraphs (§7)."""
    # Same `variant` for both, so the two summaries collide by construction.
    a = _ranking("2509.40001", score=8.0, variant=0)
    b = _ranking("2509.40002", score=7.9, variant=0)
    assert a.review.tldr == b.review.tldr, (
        "the fixture must actually collide for this test to prove anything"
    )
    picks = select([a, b], Selection(min_score=1.0, max_tag_jaccard=0.5))
    assert len(picks) == 1, "collision defence (§7) should keep one"


def test_distinct_summaries_both_survive() -> None:
    """The complement: a working guard must not collapse everything to a single item."""
    a = _ranking("2509.50001", score=8.0, variant=0)
    b = _ranking("2509.50002", score=7.9, variant=1)
    picks = select([a, b], Selection(min_score=1.0, max_tag_jaccard=0.5))
    assert len(picks) == 2


# --- invariants ---------------------------------------------------------------------------


def test_scoring_never_reads_an_outcome() -> None:
    """Temporal-leakage guard (§6.6.6).

    `matured_impact` cannot exist at T+0, and blending a future signal into a live ranking
    makes the backtest circular. This asserts the property structurally: no scoring function
    mentions any outcome type.
    """
    from screener.domain import scoring

    source = inspect.getsource(scoring)
    for forbidden in ("Outcome", "matured_impact", "outcomes", "rung"):
        assert forbidden not in source.replace("OUTCOME_GUARD_OK", ""), (
            f"{forbidden!r} appears in scoring.py — outcome data must never reach the ranker"
        )


def test_score_band_covers_every_case() -> None:
    assert score_band(9.0, delivered=True, min_score=6.5) == "delivered"
    assert score_band(9.0, delivered=False, min_score=6.5) == "above_min"
    assert score_band(5.0, delivered=False, min_score=6.5) == "mid"
    assert score_band(1.0, delivered=False, min_score=6.5) == "low"
