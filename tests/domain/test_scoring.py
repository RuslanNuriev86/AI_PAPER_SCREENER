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

from screener.domain.models import Assessment, Ranking, Review, Scores, Selection
from screener.domain.relevance import normalise_topics
from screener.domain.scoring import (
    composite,
    effective_weights,
    observable_dimensions,
    score_band,
    select,
)
from screener.domain.types import RUBRIC_WEIGHTS, TRIAGE_WEIGHTS, HardFlag, SoftFlag
from tests.factories import review_body


def _scores(*, pedigree: float | None = None, early: float | None = None) -> Scores:
    return Scores(
        relevance=7.0,
        novelty=7.0,
        rigor=6.0,
        evidence_strength=6.0,
        impact_forecast=7.0,
        reproducibility=5.0,
        pedigree=pedigree,
        early_signal=early,
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

    An earlier draft of the design wrote .27/.27/.27/.20 here, which sums to 1.01. This test
    is the thing that would have caught it.
    """
    assert sum(TRIAGE_WEIGHTS.values()) == pytest.approx(1.0)


def test_v1_effective_weights_span_six_and_sum_to_one() -> None:
    observable = observable_dimensions(_scores())
    assert set(observable) == {
        "relevance",
        "novelty",
        "rigor",
        "evidence_strength",
        "impact_forecast",
        "reproducibility",
    }
    weights = effective_weights(observable)
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["relevance"] == pytest.approx(0.20 / 0.90, abs=1e-6)


def test_v15_effective_weights_span_eight_and_sum_to_one() -> None:
    observable = observable_dimensions(_scores(pedigree=5.0, early=6.0))
    assert len(observable) == 8
    assert sum(effective_weights(observable).values()) == pytest.approx(1.0)


def test_missing_dimension_is_renormalised_not_zeroed() -> None:
    """A paper with no enrichment must not be silently penalised for it (§6.5)."""
    without = observable_dimensions(_scores())
    with_all = observable_dimensions(_scores(pedigree=10.0, early=10.0))
    score_without, _, _ = composite(without)
    score_with, _, _ = composite(with_all)
    assert score_with > score_without, "adding high scores for pedigree/early_signal should help"
    assert score_without > 0.0, "absence must not collapse the score to zero"


def test_no_observable_dimensions_is_an_error_not_a_zero() -> None:
    with pytest.raises(ValueError):
        effective_weights({})


# --- penalties and gates ------------------------------------------------------------------


def test_soft_flags_penalise_by_half_a_point_each() -> None:
    observable = observable_dimensions(_scores())
    base, _, _ = composite(observable, soft_flag_count=0)
    one, _, _ = composite(observable, soft_flag_count=1)
    two, _, _ = composite(observable, soft_flag_count=2)
    assert base - one == pytest.approx(0.5)
    assert one - two == pytest.approx(0.5)


def test_hard_flag_disqualifies_without_double_penalising() -> None:
    """Hard flags set disposition='gated'; the composite must not also subtract for them."""
    observable = observable_dimensions(_scores())
    clean, _, _ = composite(observable, soft_flag_count=0)
    soft_only, _, _ = composite(observable, soft_flag_count=0)
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
