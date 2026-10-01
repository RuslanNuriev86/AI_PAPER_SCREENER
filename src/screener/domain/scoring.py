"""The ranker and deterministic selection (§6.5, §6.4). Pure functions; no I/O.

The invariant this module exists to protect: LLMs produce *features*, Python makes
*decisions*. Nothing here calls a model or a database.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import cast

from screener.domain.models import Ranking, Selection
from screener.domain.types import RUBRIC_WEIGHTS

SOFT_FLAG_PENALTY = 0.5


def effective_weights(observable: dict[str, float]) -> dict[str, float]:
    """§6.1 weights restricted to the observed dimensions, renormalised to sum to 1.0."""
    weights = cast("dict[str, float]", RUBRIC_WEIGHTS)
    base = {dim: weights[dim] for dim in observable}
    total = sum(base.values())
    if total <= 0:
        raise ValueError("no observable dimensions to weight")
    return {dim: w / total for dim, w in base.items()}


def composite(
    quality: dict[str, float],
    measured: dict[str, float] | None = None,
    soft_flag_count: int = 0,
) -> tuple[float, dict[str, float], dict[str, float]]:
    """Combine the two halves of the rubric (§6.1, §6.5).

    Returns (score, components, effective_weights). Weights are renormalised over the dimensions
    actually observed, so an absent measured signal shifts weight to the judged half instead of
    scoring zero — which matters because measurement shows citations and venue are *normally*
    absent at T+14 (§6.1.0), and a paper must not be punished for a signal that does not exist.

    Hard flags do not appear here: they set `disposition='gated'`, which excludes the paper
    outright, so penalising them as well would double-count.
    """
    observed = {**quality, **(measured or {})}
    if not observed:
        raise ValueError("no observable dimensions to weight")
    weights = effective_weights(observed)
    components = {dim: weights[dim] * value for dim, value in observed.items()}
    score = sum(components.values()) - SOFT_FLAG_PENALTY * soft_flag_count
    return score, components, weights


def split_halves(
    quality: dict[str, float], measured: dict[str, float] | None
) -> tuple[float | None, float | None, float, float]:
    """Each half's own 0-10 mean, plus the share of total weight each actually carried.

    Reported separately so the digest can say `7.8q + 6.2m` rather than one blended number: the
    first is an opinion about text, the second is a reading from an external source.
    """
    measured = measured or {}
    weights = cast("dict[str, float]", RUBRIC_WEIGHTS)
    q_weights = {dim: weights[dim] for dim in quality}
    m_weights = {dim: weights[dim] for dim in measured}
    total = sum(q_weights.values()) + sum(m_weights.values())
    if total <= 0:
        return None, None, 0.0, 0.0
    q_share = sum(q_weights.values()) / total
    m_share = sum(m_weights.values()) / total
    q_mean = sum(quality.values()) / len(quality) if quality else None
    m_mean = sum(measured.values()) / len(measured) if measured else None
    return q_mean, m_mean, q_share, m_share


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _over(cap: int, values: Sequence[str], picks: Sequence[Ranking], attr: str) -> bool:
    counts: dict[str, int] = {}
    for p in picks:
        for v in getattr(p, attr) or []:
            counts[v] = counts.get(v, 0) + 1
    return any(counts.get(v, 0) >= cap for v in values)


def _too_similar(candidate: Ranking, picks: Sequence[Ranking], threshold: float) -> bool:
    blob = candidate.review.text_blob()
    return any(_jaccard(blob, p.review.text_blob()) > threshold for p in picks)


def select(ranked: Sequence[Ranking], cfg: Selection) -> list[Ranking]:
    """Greedy, diversified selection (§6.4).

    Gate-stage rejections never reach here at all; review-stage rejections are excluded by
    the `disposition` check. New papers always win ties, and diversification is what keeps
    the digest from reading as five variants of one idea.
    """
    kept = [r for r in ranked if r.score >= cfg.min_score and r.disposition == "eligible"]
    kept.sort(key=lambda r: r.score, reverse=True)

    picks: list[Ranking] = []
    for r in kept:
        if len(picks) == cfg.max_papers:
            break
        headline = r.score >= cfg.headline_score
        if not headline and _over(cfg.per_topic_cap, list(r.topics), picks, "topics"):
            continue
        if not headline and r.lab is not None and _over(cfg.per_lab_cap, [r.lab], picks, "lab"):
            continue
        if _too_similar(r, picks, cfg.max_tag_jaccard):
            continue
        picks.append(r)
    return picks


def score_band(score: float, *, delivered: bool, min_score: float) -> str:
    """Cohort band for the maturity loop (§6.6.1)."""
    if delivered:
        return "delivered"
    if score >= min_score:
        return "above_min"
    if score >= 4.0:
        return "mid"
    return "low"
