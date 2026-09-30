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


def observable_dimensions(scores: object) -> dict[str, float]:
    """The observable subset of a `Scores` object.

    `pedigree` and `early_signal` are excluded when None: an unmeasured signal must not
    silently become a low score (§6.5).
    """
    data: dict[str, object] = scores.model_dump()  # type: ignore[attr-defined]
    # Absence is `None`, not "belongs to the enrichment set": a v1.5 run with a working
    # Enricher *does* observe pedigree and early_signal, and excluding them by name would
    # silently drop two dimensions from every score.
    return {k: float(v) for k, v in data.items() if v is not None}  # type: ignore[arg-type]


def effective_weights(observable: dict[str, float]) -> dict[str, float]:
    """§6.1 weights restricted to the observable dimensions, renormalised to sum to 1.0.

    At v1 this spans six dimensions; from v1.5, eight. The result is stored on the Ranking
    as `effective_weights` so a past score stays explainable after the weights change.
    """
    weights = cast("dict[str, float]", RUBRIC_WEIGHTS)
    base = {dim: weights[dim] for dim in observable}
    total = sum(base.values())
    if total <= 0:
        raise ValueError("no observable dimensions to weight")
    return {dim: w / total for dim, w in base.items()}


def composite(
    observable: dict[str, float], soft_flag_count: int = 0
) -> tuple[float, dict[str, float], dict[str, float]]:
    """composite = sum(w_i * dim_i) - 0.5 * len(soft_flags).

    Hard flags do not appear: they set `disposition='gated'`, which excludes the paper
    outright, so penalising them here as well would double-count (§6.5).
    """
    weights = effective_weights(observable)
    components = {dim: weights[dim] * value for dim, value in observable.items()}
    score = sum(components.values()) - SOFT_FLAG_PENALTY * soft_flag_count
    return score, components, weights


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
