"""Test data factories. Importable as `tests.factories` so every test shares one paper shape."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from screener.domain.models import Paper

NOW = datetime(2025, 10, 1, 12, 0, tzinfo=UTC)

# Deliberately longer than the gate's 120-word minimum (MIN_ABSTRACT_WORDS): a shorter
# fixture makes every gate test fail for the wrong reason.
ABSTRACT = (
    "We present a method for LLM agent tool use in long-horizon tasks. Our approach trains a "
    "process reward model over agent trajectories collected from four different agent scaffolds, "
    "and uses it both to rerank candidate tool-call plans and as a dense reinforcement learning "
    "signal. We evaluate on WebArena and two additional web benchmarks, reporting gains of 18 "
    "points over an outcome-only baseline. We release code and the labelled trajectory dataset. "
    "The process reward model is trained with a step-level objective that assigns credit to "
    "individual actions rather than to the final outcome, which lets us compare scaffolds on a "
    "common scale instead of relying on end-to-end success rates. We ablate the labelling "
    "procedure, the number of trajectories, and the choice of teacher model, and find that the "
    "step-level signal is responsible for most of the improvement while the teacher choice "
    "matters little once the reward model has enough labelled trajectories. We additionally "
    "report error bars over five seeds and describe the failure modes we observed, including "
    "reward hacking on a subset of the benchmark tasks."
)


def make_paper(
    arxiv_id: str = "2509.18422",
    version: int = 1,
    *,
    title: str = "AgentRM: Process Reward Models for Long-Horizon Agent Trajectories",
    abstract: str = ABSTRACT,
    categories: list[str] | None = None,
    comment: str | None = None,
    submitted_at: datetime | None = None,
) -> Paper:
    return Paper(
        arxiv_id=arxiv_id,
        version=version,
        title=title,
        abstract=abstract,
        authors=["A. Author", "B. Author"],
        categories=categories or ["cs.LG", "cs.AI"],
        primary_category=(categories or ["cs.LG"])[0],
        # Inside the T+14 cohort window by default (§5.2): the digest never rates day-0 papers,
        # so a fixture that is one day old would be filtered out before the gate.
        submitted_at=submitted_at or (NOW - timedelta(days=15)),
        abs_url=f"https://arxiv.org/abs/{arxiv_id}",
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
        comment=comment,
        first_seen_at=NOW,
    )


#: Words every custom abstract must reach. The gate rejects anything shorter, so a test that
#: writes its own abstract must pad it or it will fail for the wrong reason.
SAFE_ABSTRACT_WORDS = 160


def pad(text: str, *, min_words: int = SAFE_ABSTRACT_WORDS) -> str:
    """Extend `text` with neutral filler until it clears the gate's length floor.

    The filler is deliberately content-free: a test that supplies a short abstract is testing
    the *length* rule, and incidental keywords in the padding would change which term patterns
    match and make the test lie about what it verified.
    """
    words = text.split()
    filler = ["additional", "experimental", "detail", "is", "reported", "in", "the", "appendices"]
    i = 0
    while len(words) < min_words:
        words.append(filler[i % len(filler)])
        i += 1
    return " ".join(words)


#: Six genuinely distinct review bodies.
#:
#: The prose must be *substantively* different, not one template with the nouns swapped.
#: `select()` drops items whose summaries collide (token Jaccard over tldr+what+why+caveats),
#: so templated fixture prose makes every multi-pick test fail for a reason it is not about.
#: Real papers do not share 60% of their vocabulary; the fixtures should not either.
DISTINCT_REVIEWS: list[dict[str, str]] = [
    {
        "tldr": "Routes retrieval queries to specialised sub-agents, cutting web navigation steps.",
        "tag": "computer use",
        "what_they_did": "A learned dispatcher assigns each navigation substep to one of four "
        "specialised policies trained on disjoint site clusters.",
        "why_it_matters": "Routing replaces one monolithic policy with auditable, per-site "
        "competence, so failures localise to a single specialist.",
        "caveats": "Site clusters were hand-labelled; drift on unseen domains is untested.",
    },
    {
        "tldr": "Assigns credit to individual planning steps instead of the terminal reward.",
        "tag": "agentic RL",
        "what_they_did": "A learned critic regresses per-step advantage from sparse episode "
        "returns, then reweights the policy gradient accordingly.",
        "why_it_matters": "Step-level credit makes long-horizon planning trainable without "
        "dense human annotation of every intermediate state.",
        "caveats": "Critic variance grows with horizon; results stop at forty steps.",
    },
    {
        "tldr": "Compacts conversation memory into fixed-size summaries with recall guarantees.",
        "tag": "memory & context",
        "what_they_did": "An eviction policy keeps a bounded working set selected by estimated "
        "future utility, rewriting the rest into a rolling summary.",
        "why_it_matters": "Bounded memory with measured recall turns context limits from a hard "
        "cliff into a tunable tradeoff.",
        "caveats": "Recall is measured by an automatic judge, not by human raters.",
    },
    {
        "tldr": "Isolates computer-use agents behind scoped permissions per task.",
        "tag": "safety & oversight",
        "what_they_did": "Each task declares a capability manifest, and a mediation layer "
        "denies any filesystem or network call outside the declared scope.",
        "why_it_matters": "Declared scopes convert an unbounded prompt-injection surface into "
        "an enumerable, testable set of reachable actions.",
        "caveats": "Manifests are author-supplied; a wrong manifest silently over-permits.",
    },
    {
        "tldr": "Detects deadlock in multi-agent handoff graphs before execution.",
        "tag": "multi-agent coordination",
        "what_they_did": "A static analysis over the delegation graph finds cycles that cannot "
        "make progress, and the planner rejects those topologies at construction time.",
        "why_it_matters": "Catching deadlock statically is cheaper than recovering from a "
        "stalled multi-agent rollout at inference time.",
        "caveats": "Assumes handoffs are declared up front, which dynamic agents violate.",
    },
    {
        "tldr": "Chooses tools under partial observability with calibrated abstention.",
        "tag": "evaluation & benchmarks",
        "what_they_did": "A selector estimates whether the available tools can satisfy the "
        "request and abstains when the estimate falls below a tuned threshold.",
        "why_it_matters": "Calibrated abstention makes a wrong-tool call visible instead of "
        "silently producing a confident, incorrect answer.",
        "caveats": "The abstention threshold is tuned per benchmark and does not transfer.",
    },
]


def review_body(index: int) -> dict[str, str]:
    """A distinct review body, cycled by index."""
    return DISTINCT_REVIEWS[index % len(DISTINCT_REVIEWS)]


def rubric_scores(value: float = 8.0) -> dict[str, float]:
    return {
        "relevance": value,
        "novelty": value,
        "rigor": value,
        "evidence_strength": value,
        "impact_forecast": value,
        "reproducibility": value,
    }
