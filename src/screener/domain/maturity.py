"""Maturity loop: rung scheduling and outcome grading (§6.6).

v0 ships only what cannot be deferred. Labels are the non-renewable asset (§2, principle 7):
a paper enrolled today yields its T+180 measurement in six months, and no later engineering
recovers a label that was never collected. So enrolment and the T+14 rung are in v0, while
calibration reports are not.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime, timedelta
from itertools import pairwise

from screener.domain.models import (
    Outcome,
    OutcomeScale,
    OutcomeSignal,
    RevisitConfig,
    WatchlistEntry,
)


def rung_due_date(cohort_date: date, rung_days: int) -> date:
    return cohort_date + timedelta(days=rung_days)


def age_days(cohort_date: date, now: datetime) -> int:
    return (now.date() - cohort_date).days


def due_rungs(
    entries: Sequence[WatchlistEntry],
    now: datetime,
    cfg: RevisitConfig,
    already_measured: set[tuple[str, int]],
) -> list[tuple[int, WatchlistEntry]]:
    """Which (rung, paper) pairs should be measured today.

    Pure and deterministic: rungs are due-dated from `cohort_date`, so a missed day measures
    the rung *late* rather than losing it — which matters because the host may be asleep
    (§6.6.2). Past `max_lateness_days` the rung is skipped rather than back-filled, because
    a measurement at T+45 is not a T+14 measurement.
    """
    out: list[tuple[int, WatchlistEntry]] = []
    for entry in entries:
        age = age_days(entry.cohort_date, now)
        for rung in cfg.rungs:
            if (entry.arxiv_id, rung) in already_measured:
                continue
            if age < rung:
                continue
            lateness = age - rung
            if lateness > cfg.max_lateness_days.get(rung, rung):
                continue
            out.append((rung, entry))
    out.sort(key=lambda pair: (pair[0], pair[1].arxiv_id))
    return out


def _interpolate(anchors: Mapping[int, int], value: float) -> float:
    """Map a raw count onto 0-10 using written anchors, linearly between them.

    Anchors are config, not code: they drift as a field grows, and re-anchoring must be a
    config change recorded in `config_hash` (§6.6.3). Integer keys are the 0-10 score; the
    values are the counts that earn them.
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


def grade(
    signal: OutcomeSignal, scale: OutcomeScale, now: datetime, *, rung_days: int = 14
) -> Outcome:
    """Turn raw counts into a 0-10 `matured_impact` (§6.6.3).

    Missing signal is not zero signal: a paper with no linked repo is graded on the
    components it has, with weights renormalised over what was actually observed. That is
    the same rule `scoring.effective_weights` applies to unobservable rubric dimensions.
    """
    present: dict[str, float] = {}
    if signal.stars is not None:
        present["stars"] = _interpolate(scale.stars, signal.stars)
    if signal.hf_upvotes is not None:
        present["hf_upvotes"] = _interpolate(scale.hf_upvotes, signal.hf_upvotes)
    if signal.code_url:
        present["code_release"] = 10.0
    if signal.revisions:
        present["revisions"] = min(10.0, float(signal.revisions) * 2.0)

    if not present:
        return Outcome(
            arxiv_id=signal.arxiv_id,
            rung_days=rung_days,
            actual_age_days=rung_days,
            status="missed",
            measured_at=now,
            signal=signal,
            matured_impact=None,
            components_present=[],
        )

    configured = {k: v for k, v in scale.weights.items() if k in present}
    total = sum(configured.values()) or 1.0
    matured = sum(present[k] * (w / total) for k, w in configured.items())

    return Outcome(
        arxiv_id=signal.arxiv_id,
        rung_days=rung_days,
        actual_age_days=rung_days,
        status="measured",
        measured_at=now,
        signal=signal,
        matured_impact=round(matured, 3),
        components_present=sorted(present),
    )


def enrol_band(score: float | None, *, delivered: bool, min_score: float) -> str:
    """The §6.6.1 cohort band. `gate_only` when the paper was never scored."""
    from screener.domain.scoring import score_band

    if score is None:
        return "gate_only"
    return score_band(score, delivered=delivered, min_score=min_score)
