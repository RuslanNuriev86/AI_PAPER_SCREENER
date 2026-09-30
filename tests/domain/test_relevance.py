"""The gate is pure, so it can be tested exhaustively (§13.3). These are the highest-value
tests in the project: the design claims "agent" is ~70% wrong as a keyword, and this is where
that claim either holds or does not.
"""

from __future__ import annotations

import pytest

from screener.domain.models import Profile
from screener.domain.relevance import MIN_ABSTRACT_WORDS, gate, normalise_topics, relevance_hint
from screener.domain.types import HardFlag
from tests.factories import make_paper, pad


def test_keeps_a_paper_with_strong_agentic_signal(profile: Profile) -> None:
    result = gate(make_paper(), profile)
    assert result.keep
    assert result.reason is None
    assert "LLM agent" in result.hint.matched
    assert result.hint.score > 0


def test_rejects_a_category_outside_the_prefilter(profile: Profile) -> None:
    paper = make_paper(categories=["astro-ph.GA"], title="agentic tool use in galaxies")
    result = gate(paper, profile)
    assert not result.keep
    assert result.reason is HardFlag.NOT_AGENTIC


def test_rejects_a_short_abstract(profile: Profile) -> None:
    paper = make_paper(abstract="agentic " * 10)
    result = gate(paper, profile)
    assert not result.keep
    assert result.reason is HardFlag.NO_TECHNICAL_CONTRIBUTION
    assert len(paper.abstract.split()) < MIN_ABSTRACT_WORDS


def test_exclusions_are_subtractive_not_absolute(profile: Profile) -> None:
    """Two strong agentic terms survive one exclusion — the deliberate §5 asymmetry.

    Dropping a good paper costs more than triaging an extra one, so an exclusion cannot veto
    a paper that has real positive signal.
    """
    paper = make_paper(
        title="LLM agent tool use in an agent-based model of markets",
        abstract=pad(
            "We study LLM agent tool use with a multi-agent LLM framework and an agent memory "
            "module. Our agent benchmark covers planning and reflection."
        ),
    )
    result = gate(paper, profile)
    assert result.hint.excluded, "the exclusion should have matched"
    assert result.keep, "positive signal should outweigh a single exclusion"
    assert result.hint.score > 0


def test_an_exclusion_can_still_reject_when_signal_is_weak() -> None:
    """The other side of the asymmetry: one weak term and one exclusion nets zero, so reject.

    Uses its own minimal profile so the arithmetic is explicit — with the full fixture profile
    the padding contributes extra weak-term hits and the boundary under test is obscured.
    """
    minimal = Profile(
        categories=["cs.AI"],
        strong_terms=[],
        weak_terms=["agent"],
        exclude_patterns=["agent-based model"],
    )
    paper = make_paper(
        title="An agent-based model of epidemic spread",
        abstract=pad("We present an agent-based model of epidemic spread."),
    )
    result = gate(paper, minimal)
    # one weak hit (+1) minus one exclusion (-1) == 0, and 0 is not > 0
    assert result.hint.score == 0.0
    assert not result.keep
    assert result.reason is HardFlag.NOT_AGENTIC


def test_classical_mas_language_is_penalised(profile: Profile) -> None:
    """The negative examples the design says are mandatory in the gate (§16 decision 3)."""
    paper = make_paper(
        title="Multi-agent LLM coordination for a traffic signal control system",
        abstract=pad(
            "We deploy a multi-agent system for traffic signal control where each agent plans "
            "independently."
        ),
    )
    hint = relevance_hint(paper, profile)
    assert hint.excluded, "traffic + multi-agent should match an exclude pattern"


def test_survey_needs_strong_signal_to_survive(profile: Profile) -> None:
    survey = make_paper(
        title="A survey of agent memory", abstract=pad("We survey memory mechanisms.")
    )
    result = gate(survey, profile)
    assert not result.keep
    assert result.reason is HardFlag.PURE_SURVEY


def test_gate_is_deterministic(profile: Profile) -> None:
    """Pure means same input, same GateResult — this is what makes replay meaningful."""
    paper = make_paper()
    first, second = gate(paper, profile), gate(paper, profile)
    assert first.model_dump() == second.model_dump()


def test_rejection_always_carries_a_reason(profile: Profile) -> None:
    """§6.2 promises a gated paper is 'recorded with a reason'."""
    for paper in (
        make_paper(categories=["astro-ph.GA"]),
        make_paper(abstract="too short to judge"),  # deliberately below the floor
    ):
        result = gate(paper, profile)
        assert not result.keep and result.reason is not None


def test_malformed_exclude_pattern_does_not_crash_the_gate() -> None:
    """A bad regex in config must cost precision, not the whole run."""
    profile = Profile(
        categories=["cs.AI"], strong_terms=["LLM agent"], exclude_patterns=["[unclosed"]
    )
    result = gate(make_paper(), profile)
    assert result.keep


# --- topic normalisation ------------------------------------------------------------------


def test_topics_normalise_vague_model_tags() -> None:
    """Models emit "benchmarks"; the vocabulary says "evaluation & benchmarks"."""
    topics = normalise_topics(["benchmarks", "memory"])
    assert "evaluation & benchmarks" in topics
    assert "memory & context" in topics


def test_unknown_tags_never_reach_the_cap() -> None:
    assert normalise_topics(["quantum chromodynamics"]) == []


@pytest.mark.parametrize(
    ("tag", "expected"),
    [
        ("benchmark", "evaluation & benchmarks"),
        ("benchmarks", "evaluation & benchmarks"),
        ("coordination", "multi-agent coordination"),
        ("memory", "memory & context"),
        ("computer-use-agents", "computer use"),
        ("gui-agent", "computer use"),
        ("misalignment", "safety & oversight"),
        ("post-training", "agentic RL"),
        ("protocol", "infrastructure/protocols"),
        ("mcp", "infrastructure/protocols"),
    ],
)
def test_anchors_route_tags_to_the_right_topic(tag: str, expected: str) -> None:
    assert normalise_topics([tag]) == [expected]


@pytest.mark.parametrize(
    "tag",
    [
        "multi-hop-qa",  # "multi" must not imply multi-agent coordination
        "worldly",  # "rl" must not match inside a word
        "retrieval-augmented-generation",  # "eval" must not match inside "retrieval"
        "multi-agent",  # ambiguous on its own; no anchor present
        "quantum-chromodynamics",
    ],
)
def test_generic_words_do_not_route_tags(tag: str) -> None:
    """Regression: collapsing every paper into one topic turns per_topic_cap into a global cap.

    On live data six of six papers mapped to "multi-agent coordination" because "multi" and
    "agent" matched everything, which would have limited the whole digest to two items.
    """
    assert normalise_topics([tag]) == []


def test_a_paper_may_legitimately_map_to_several_topics() -> None:
    topics = normalise_topics(["multi-agent", "safety-verification", "benchmark"])
    assert "safety & oversight" in topics
    assert "evaluation & benchmarks" in topics
