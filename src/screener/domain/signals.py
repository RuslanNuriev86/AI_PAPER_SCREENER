"""Signal mapping: raw measured values -> the measured half of the rubric (§6.1, §6.6.3).

Nothing here is inferred. Each function takes a value an adapter actually read, plus the age at
which it was read, and maps it through anchors that live in `config/outcome_scale.yaml` so the
scale can be re-derived from observed cohort quantiles without touching code.

Two rules the measurement in §6.1.0 forced:

* **Absent is not zero.** A paper with no findable repo scores `None` on `repo_signal`, not 0 —
  the score() renormalises it away. Measurement showed only 10-20% of T+14 papers have a
  detectable repo, so scoring absence as zero would punish four fifths of every cohort for a
  signal that does not exist.
* **A star count is only about this paper if the repo is this paper's.** One sampled cohort
  contained a 34,432-star repository, which is a paper linking to a pre-existing popular project.
  Stars count only when the repo was created near the paper.
"""

from __future__ import annotations

from datetime import datetime
from itertools import pairwise

import structlog

from screener.domain.models import (
    Enrichment,
    OutcomeScale,
    Paper,
    Ranking,
    SignalScores,
)

log = structlog.get_logger(__name__)

#: A repo created more than this many days before the paper is not the paper's own artifact.
REPO_YOUNG_DAYS = 45


def anchor_score(anchors: dict[int, int], value: float) -> float:
    """Map a raw count onto 0-10 through written anchors, linearly between them.

    Anchors are `{score: count}`. Values below the smallest anchor clamp to its score, values
    above the largest clamp to 10, so a single viral repo cannot produce an unbounded score.
    """
    if not anchors:
        return 0.0
    points = sorted((int(score), float(count)) for score, count in anchors.items())
    if value <= points[0][1]:
        return float(points[0][0])
    if value >= points[-1][1]:
        return float(points[-1][0])
    for (lo_score, lo_count), (hi_score, hi_count) in pairwise(points):
        if lo_count <= value <= hi_count:
            if hi_count == lo_count:
                return float(hi_score)
            frac = (value - lo_count) / (hi_count - lo_count)
            return lo_score + frac * (hi_score - lo_score)
    return float(points[-1][0])


def _citation_signal(citations: int | None, age_days: int, scale: OutcomeScale) -> float | None:
    """Citations, but only when the count is informative.

    A zero is the universal baseline at T+14 — measured zero for *every* paper in every cohort
    sampled (§6.1.0) — so scoring it as 0.0 would treat "not yet cited" as "measured badly" and
    silently penalise the whole cohort by exactly the weight of this dimension. It is therefore
    returned as absent, and §6.5's renormalisation redistributes its weight rather than letting
    a signal that cannot yet discriminate act as a penalty.

    Once a source that merges preprint and published records is wired in (the T+90 rung, §6.1.0),
    a non-zero count becomes a real measurement and is scored normally.
    """
    if citations is None or citations == 0:
        return None
    return anchor_score(scale.citations, citations)


def venue_score(venue: str | None, scale: OutcomeScale) -> float | None:
    """A stated venue is decisive when present, and absent far more often (§6.1.0)."""
    if not venue:
        return None
    lowered = venue.lower()
    for marker, score in scale.venue_markers.items():
        if marker in lowered:
            return float(score)
    return float(scale.venue_default)


def map_signals(
    paper: Paper, enrichment: Enrichment | None, *, now: datetime, scale: OutcomeScale
) -> SignalScores:
    """Turn one paper's enrichment into the measured half of its rating."""
    age_days = max(0, (now - paper.submitted_at).days)
    if enrichment is None:
        return SignalScores(arxiv_id=paper.arxiv_id, age_days=age_days)

    stars = enrichment.stars
    repo_usable = enrichment.repo_created_days_before_paper is None or (
        enrichment.repo_created_days_before_paper <= REPO_YOUNG_DAYS
    )
    if stars is not None and not repo_usable:
        # The repo predates the paper, so its stars are not this paper's traction.
        log.info(
            "signals.repo_predates_paper",
            arxiv_id=paper.arxiv_id,
            stars=stars,
            days=enrichment.repo_created_days_before_paper,
        )
        stars = None

    return SignalScores(
        arxiv_id=paper.arxiv_id,
        age_days=age_days,
        citation_signal=_citation_signal(enrichment.citations, age_days, scale),
        repo_signal=anchor_score(scale.stars, stars) if stars is not None else None,
        venue_signal=venue_score(enrichment.venue, scale),
        citations=enrichment.citations,
        stars=stars,
        venue=enrichment.venue,
        repo_url=enrichment.repo_url,
        sources_ok=list(enrichment.sources_ok),
    )


def attach(ranking: Ranking, signals: SignalScores) -> Ranking:
    """Fold the measured half into a ranking and record the basis that produced the score.

    Called after enrichment. Hard-flag disposition is preserved: a gated paper stays gated no
    matter how many stars its repo has, because the gate is a policy decision, not a score.
    """
    from screener.domain.models import QualityScores, RatingBasis
    from screener.domain.scoring import composite, split_halves

    quality: QualityScores = ranking.review.scores
    judged = quality.observable()
    measured = signals.observable()
    score, components, weights = composite(
        judged, measured, soft_flag_count=len(ranking.soft_flags)
    )
    q_mean, m_mean, q_share, m_share = split_halves(judged, measured)
    basis = RatingBasis(
        age_days=signals.age_days,
        quality=quality,
        signals=signals,
        quality_score=q_mean,
        measured_score=m_mean,
        quality_weight=q_share,
        measured_weight=m_share,
        components=components,
        effective_weights=weights,
        soft_flags=list(ranking.soft_flags),
        soft_flag_penalty=0.5 * len(ranking.soft_flags),
    )
    return ranking.model_copy(
        update={
            "score": score,
            "components": components,
            "effective_weights": weights,
            "signals": signals,
            "basis": basis,
        }
    )
