"""Deterministic layer-1 gate and topic classification (§5).

Pure by construction: no I/O, no repository access, no clock. That is what lets the gate be
tested exhaustively, and it is why the one stateful §6.2 rule ("replaced and already
delivered") lives in the pipeline's `dedupe_revisions` step instead of here.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from screener.domain.models import GateResult, Paper, Profile, RelevanceHint
from screener.domain.types import TOPIC_ANCHORS, TOPICS, HardFlag, Topic

#: Below this many words an abstract cannot support a technical judgment (§5 hard reject).
MIN_ABSTRACT_WORDS = 120


def _compile(patterns: Iterable[str]) -> list[re.Pattern[str]]:
    out: list[re.Pattern[str]] = []
    for p in patterns:
        try:
            out.append(re.compile(p, re.IGNORECASE))
        except re.error:
            # A malformed profile pattern must not take down the run; it is surfaced by
            # `screener doctor` instead. Failing open is right here: the gate is a filter,
            # and losing a pattern costs precision, not correctness.
            continue
    return out


def _matches(text: str, patterns: Sequence[re.Pattern[str]]) -> list[str]:
    return [m.group(0) for pat in patterns if (m := pat.search(text))]


def relevance_hint(paper: Paper, profile: Profile) -> RelevanceHint:
    """Layer 1 scoring (§5).

    Strong terms are worth 2, weak terms 1, and exclusions subtract 1. Exclusions are
    deliberately *subtractive rather than absolute*: a paper with two strong agentic terms
    survives one exclusion, because triaging a few extra papers is cheaper than dropping a
    good one.
    """
    text = f"{paper.title}\n{paper.abstract}"
    strong = _matches(text, _compile(profile.strong_terms))
    weak = _matches(text, _compile(profile.weak_terms))
    excluded = _matches(text, _compile(profile.exclude_patterns))
    score = 2.0 * len(strong) + 1.0 * len(weak) - 1.0 * len(excluded)
    return RelevanceHint(score=score, matched=strong + weak, excluded=excluded)


_SURVEY_RE = re.compile(r"\b(survey|systematic review|literature review|taxonomy of)\b", re.I)


def _is_pure_survey(paper: Paper) -> bool:
    return bool(_SURVEY_RE.search(f"{paper.title} {paper.abstract}"))


def gate(paper: Paper, profile: Profile) -> GateResult:
    """Decide whether a paper enters the pipeline.

    Order follows §5: category prefilter, hard rejects, then the positive/exclusion balance.
    Delivery state is deliberately not consulted — see the module docstring.
    """
    categories = set(paper.categories) | (
        {paper.primary_category} if paper.primary_category else set()
    )
    if not categories & set(profile.categories):
        return GateResult(
            paper=paper,
            keep=False,
            reason=HardFlag.NOT_AGENTIC,
            hint=RelevanceHint(score=-1.0, excluded=["category prefilter"]),
        )

    words = len(paper.abstract.split())
    if words < MIN_ABSTRACT_WORDS:
        return GateResult(
            paper=paper,
            keep=False,
            reason=HardFlag.NO_TECHNICAL_CONTRIBUTION,
            hint=RelevanceHint(excluded=[f"abstract < {MIN_ABSTRACT_WORDS} words"]),
        )

    hint = relevance_hint(paper, profile)

    # Surveys are gated from the *daily* digest outright, however strong the signal, because
    # §6.2 allows them only in the dedicated weekly recap. The reason is recorded rather than
    # dropped so the recap mode (and the footer stats) can still find them — and it is checked
    # before the score threshold so a survey is never mislabelled as merely "not agentic",
    # which would lose exactly the information the recap needs.
    if _is_pure_survey(paper):
        return GateResult(paper=paper, keep=False, reason=HardFlag.PURE_SURVEY, hint=hint)

    # The threshold is profile config, not a constant: one weak term ("agent") scoring 1.0
    # must not be enough, or layer 1 passes ~half of cs.AI and the LLM tier pays for it.
    if hint.score < profile.min_gate_score:
        return GateResult(paper=paper, keep=False, reason=HardFlag.NOT_AGENTIC, hint=hint)

    return GateResult(paper=paper, keep=True, hint=hint)


#: Anchors at least this long match on a shared prefix, so plurals line up.
_PREFIX_MATCH_LEN = 6


def _anchor_matches(anchor: str, words: frozenset[str], raw: str) -> bool:
    """Does one tag refer to a topic, via this anchor?

    Three rules, each earning its place:

    * a **phrase** anchor ("post-training") is matched against the raw tag, since splitting on
      punctuation destroys it;
    * a **long** anchor matches a tag word sharing its first `_PREFIX_MATCH_LEN` characters,
      which lines up "benchmark"/"benchmarks" and "protocol"/"protocols";
    * a **short** anchor ("rl", "gui", "mcp") must match a whole word. Substring matching here
      is what made "eval" fire on "ret**rieval**" and "rl" on "wo**rl**dly".
    """
    if "-" in anchor or " " in anchor:
        return anchor in raw
    if len(anchor) >= _PREFIX_MATCH_LEN:
        head = anchor[:_PREFIX_MATCH_LEN]
        return any(len(w) >= _PREFIX_MATCH_LEN and w[:_PREFIX_MATCH_LEN] == head for w in words)
    return anchor in words


def normalise_topics(tags: Sequence[str]) -> list[Topic]:
    """Map free-form review tags onto the closed Topic vocabulary (§6.4).

    Matching is on the explicit anchors in `types.TOPIC_ANCHORS`, never on generic domain
    words: a paper tagged "multi-hop-qa" is not thereby about multi-agent coordination. Getting
    this wrong is not cosmetic — it collapsed six of six live papers into one topic, which
    turned `per_topic_cap` from a diversification rule into a global cap on the digest.

    A tag that matches nothing is dropped from `topics` but kept in `Ranking.tags` for
    similarity, so an unmatched paper carries no topic rather than a wrong one.
    """
    out: list[Topic] = []
    for tag in tags:
        raw = tag.lower().replace("_", "-").replace("/", "-")
        words = frozenset(w for w in re.split(r"[^a-z0-9]+", raw) if w)
        for topic in TOPICS:
            spelled = (topic.lower(), topic.lower().replace(" ", "-").replace(" & ", "-and-"))
            hit = any(name in raw for name in spelled) or any(
                _anchor_matches(a, words, raw) for a in TOPIC_ANCHORS[topic]
            )
            if hit and topic not in out:
                out.append(topic)
    return out
