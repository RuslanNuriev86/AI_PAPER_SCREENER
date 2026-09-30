"""LLM adapter contract tests (§11).

The request body is asserted, not assumed. DeepSeek enables thinking mode by default and
**thinking mode silently ignores `temperature`** — it does not error, it does nothing — so a
refactor that drops the `thinking` field would quietly make every score non-repeatable while
all existing tests still passed. That is the bug this file exists to prevent.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from screener.adapters.llm import DEFAULT_BASE_URL, FakeLLM, OpenAILikeLLM, extract_json
from screener.domain.models import Prompt, Review

VALID_REVIEW = {
    "tldr": "Routes retrieval to specialised sub-agents, cutting navigation steps.",
    "what_they_did": "A dispatcher assigns each substep to one of four specialised policies.",
    "why_it_matters": "Routing replaces one monolithic policy with auditable competence.",
    "caveats": "Site clusters were hand-labelled; drift on unseen domains is untested.",
    "lenses": ["method"],
    "tags": ["evaluation & benchmarks"],
    "evidence_quotes": [],
    "scores": {
        "relevance": 7,
        "novelty": 7,
        "rigor": 6,
        "evidence_strength": 6,
        "impact_forecast": 7,
        "reproducibility": 5,
    },
    "soft_flags": [],
    "hard_flag": None,
}


class _Resp:
    status_code = 200

    def __init__(self, content: str, usage: dict[str, int] | None = None) -> None:
        self._content = content
        self._usage = usage or {"prompt_tokens": 1200, "completion_tokens": 300}

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return {
            "choices": [{"message": {"content": self._content}}],
            "usage": self._usage,
        }


def _capture(llm: OpenAILikeLLM, responses: list[_Resp]) -> dict[str, Any]:
    """Replace the transport so the outgoing body can be inspected without a network call."""
    seen: dict[str, Any] = {"bodies": [], "urls": []}
    queue = list(responses)

    async def fake_post(url: str, **kw: Any) -> _Resp:
        seen["urls"].append(url)
        seen["bodies"].append(kw.get("json"))
        return queue.pop(0) if queue else responses[-1]

    llm._client.post = fake_post  # type: ignore[method-assign]
    return seen


async def _call(llm: OpenAILikeLLM, temperature: float = 0.0) -> Review:
    try:
        return await llm.parse(
            model="deepseek-flash",
            prompt=Prompt(name="review", version="v0", body="Return one JSON object."),
            payload="TITLE: t",
            schema=Review,
            temperature=temperature,
        )
    finally:
        await llm.aclose()


# --- request shape ----------------------------------------------------------------------


async def test_default_base_url_is_deepseek_openai_format() -> None:
    assert DEFAULT_BASE_URL == "https://api.deepseek.com"
    llm = OpenAILikeLLM("k")
    seen = _capture(llm, [_Resp(json.dumps(VALID_REVIEW))])
    await _call(llm)
    assert seen["urls"] == ["https://api.deepseek.com/chat/completions"]


async def test_thinking_is_explicitly_disabled_and_temperature_is_sent() -> None:
    """The load-bearing assertion: without `thinking: disabled`, temperature is ignored."""
    llm = OpenAILikeLLM("k", thinking=False)
    seen = _capture(llm, [_Resp(json.dumps(VALID_REVIEW))])
    await _call(llm, temperature=0.0)
    body = seen["bodies"][0]
    assert body["thinking"] == {"type": "disabled"}
    assert body["temperature"] == 0.0
    assert body["model"] == "deepseek-flash"
    assert body["response_format"] == {"type": "json_object"}


async def test_thinking_mode_omits_temperature_instead_of_pretending_it_applies() -> None:
    llm = OpenAILikeLLM("k", thinking=True)
    seen = _capture(llm, [_Resp(json.dumps(VALID_REVIEW))])
    await _call(llm, temperature=0.0)
    body = seen["bodies"][0]
    assert body["thinking"] == {"type": "enabled"}
    assert "temperature" not in body, (
        "sending temperature in thinking mode would mislead a reader into thinking it applies"
    )


# --- parsing ----------------------------------------------------------------------------


def test_extract_json_handles_fences_and_prose() -> None:
    assert extract_json('{"a": 1}') == '{"a": 1}'
    assert extract_json('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert extract_json('Here you go:\n{"a": 1}\nHope that helps.') == '{"a": 1}'


def test_extract_json_passes_through_unparseable_text() -> None:
    """Callers validate with Pydantic, so this must not raise here."""
    assert extract_json("not json at all") == "not json at all"


async def test_parses_a_well_formed_response_into_the_schema() -> None:
    llm = OpenAILikeLLM("k")
    _capture(llm, [_Resp(json.dumps(VALID_REVIEW))])
    review = await _call(llm)
    assert isinstance(review, Review)
    assert review.lenses == ["method"]
    assert review.scores.impact_forecast == 7


async def test_malformed_json_is_retried_once_then_succeeds() -> None:
    """§12.1: one repair retry with the validation error appended, then the paper drops."""
    llm = OpenAILikeLLM("k")
    seen = _capture(llm, [_Resp("{}"), _Resp(json.dumps(VALID_REVIEW))])
    review = await _call(llm)
    assert review.tldr.startswith("Routes retrieval")
    assert len(seen["bodies"]) == 2, "a second attempt should have been made"
    repair = seen["bodies"][1]["messages"][-1]["content"]
    assert "failed validation" in repair


async def test_two_malformed_responses_raise_so_the_paper_is_dropped() -> None:
    llm = OpenAILikeLLM("k")
    _capture(llm, [_Resp("{}"), _Resp("{}")])
    with pytest.raises(RuntimeError, match="failed validation twice"):
        await _call(llm)


async def test_cost_estimate_uses_reported_usage() -> None:
    llm = OpenAILikeLLM("k")
    entry = llm.cost_of(
        "deepseek-flash", {"usage": {"prompt_tokens": 1_000_000, "completion_tokens": 0}}
    )
    # deepseek-flash peak cache-miss input is $0.30 / 1M tokens.
    assert entry.usd == pytest.approx(0.30, abs=1e-6)
    assert entry.input_tokens == 1_000_000


# --- the deterministic fake ---------------------------------------------------------------


async def test_fake_llm_returns_valid_reviews_without_network() -> None:
    """Used by tests; never by a live run (build_deps refuses without a credential)."""
    llm = FakeLLM()
    review = await llm.parse(
        model="m",
        prompt=Prompt(name="review", version="v0", body="b"),
        payload="TITLE: Something Distinct\n",
        schema=Review,
    )
    assert isinstance(review, Review)
    assert "Something Distinct" in review.tldr
