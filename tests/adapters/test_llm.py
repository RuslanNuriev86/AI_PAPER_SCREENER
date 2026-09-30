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


# --- chat-id discovery (doctor) -----------------------------------------------------------
# `TELEGRAM_CHAT_ID` is not look-up-able: it is whatever Telegram reports once the bot has
# received something. These tests pin the extraction, since it is the one setup step that
# cannot be done without a live round-trip.


def test_extract_chat_ids_covers_dm_group_and_channel() -> None:
    from screener.doctor import extract_chat_ids

    updates = [
        {"message": {"chat": {"id": 12345, "type": "private", "first_name": "Ada"}}},
        {"message": {"chat": {"id": -1001234567890, "type": "supergroup", "title": "Agents"}}},
        {"channel_post": {"chat": {"id": -1009876543210, "type": "channel", "title": "Feed"}}},
        {
            "my_chat_member": {
                "chat": {"id": -1005555555555, "type": "group", "title": "Added Before Typing"}
            }
        },
    ]
    ids = dict(extract_chat_ids(updates))
    assert ids["12345"] == "private (Ada)"
    assert ids["-1001234567890"] == "supergroup (Agents)"
    assert ids["-1009876543210"] == "channel (Feed)"
    # my_chat_member arrives as soon as the bot is added, before anyone types anything.
    assert ids["-1005555555555"] == "group (Added Before Typing)"


def test_extract_chat_ids_deduplicates_and_ignores_junk() -> None:
    from screener.doctor import extract_chat_ids

    updates = [
        {"message": {"chat": {"id": 7, "type": "private"}}},
        {"message": {"chat": {"id": 7, "type": "private"}}},
        {"message": {}},  # no chat
        {"edited_message": "not a dict"},  # malformed
        {"callback_query": {"id": "x"}},  # not a chat-bearing update
    ]
    assert extract_chat_ids(updates) == [("7", "private")]


def test_extract_chat_ids_returns_empty_when_the_bot_has_heard_nothing() -> None:
    from screener.doctor import extract_chat_ids

    assert extract_chat_ids([]) == []


# --- doctor's HTTP probe classification ---------------------------------------------------
# A blanket "status < 500 is fine" once reported arXiv HTTP 400 as an OK line, hiding a defect
# in the probe's own URL. These tests pin the distinction between "server refused us" and
# "we sent a bad request".


class _StubResp:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


class _StubClient:
    def __init__(self, status_code: int) -> None:
        self._code = status_code

    async def get(self, url: str) -> _StubResp:
        return _StubResp(self._code)


async def _probe(code: int):  # type: ignore[no-untyped-def]
    from screener.doctor import Report, _check_http

    r = Report()
    await _check_http(r, _StubClient(code), "x", "https://example.test")  # type: ignore[arg-type]
    return r.lines[0]


async def test_probe_2xx_is_ok() -> None:
    assert (await _probe(200)).startswith("[  ok  ]")


async def test_probe_401_is_ok_because_it_proves_reachability() -> None:
    line = await _probe(401)
    assert line.startswith("[  ok  ]")
    assert "auth checked separately" in line


async def test_probe_400_is_a_failure_not_an_ok() -> None:
    """This exact status was reported as OK while the probe URL was malformed."""
    line = await _probe(400)
    assert line.startswith("[ fail ]")
    assert "request is wrong" in line


async def test_probe_429_and_5xx_are_warnings() -> None:
    assert (await _probe(429)).startswith("[ warn ]")
    assert (await _probe(503)).startswith("[ warn ]")


def test_arxiv_probe_url_includes_a_search_query() -> None:
    """arXiv 400s on a query without `search_query`, so the probe must send one."""
    from screener.config import load_settings
    from screener.doctor import _arxiv_probe_url

    url = _arxiv_probe_url(load_settings("config"))
    assert "search_query=" in url
    assert "max_results=1" in url


def test_arxiv_probe_url_falls_back_when_no_categories_are_configured() -> None:
    from screener.config import Settings
    from screener.doctor import _arxiv_probe_url

    assert "search_query=cat:cs.AI" in _arxiv_probe_url(Settings())


# --- prose bounds are truncation, not rejection -------------------------------------------
# The first live run dropped a real paper because `what_they_did` came back ~430 chars against
# a 420 limit. A cosmetic overrun must not cost a paper; semantic constraints stay strict.


def test_prose_over_the_limit_is_truncated_not_rejected() -> None:
    long_mechanism = "They train a process reward model. " * 30
    review = Review(**{**VALID_REVIEW, "what_they_did": long_mechanism})
    assert len(review.what_they_did) <= 420
    assert review.what_they_did.endswith((".", "…"))


def test_all_four_prose_fields_are_bounded() -> None:
    review = Review(
        **{
            **VALID_REVIEW,
            "tldr": "x" * 900,
            "what_they_did": "y" * 900,
            "why_it_matters": "z" * 900,
            "caveats": "w" * 900,
        }
    )
    assert len(review.tldr) <= 220
    assert len(review.what_they_did) <= 420
    assert len(review.why_it_matters) <= 420
    assert len(review.caveats) <= 240


def test_short_prose_is_left_alone() -> None:
    review = Review(**VALID_REVIEW)
    assert review.what_they_did == VALID_REVIEW["what_they_did"]
    assert not review.what_they_did.endswith("…")


def test_semantic_errors_are_still_hard_failures() -> None:
    """Truncation applies to length only; a bad enum must still be rejected."""
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        Review(**{**VALID_REVIEW, "hard_flag": "definitely_not_a_flag"})
    with pytest.raises(pydantic.ValidationError):
        Review(**{**VALID_REVIEW, "scores": {**VALID_REVIEW["scores"], "relevance": 99}})


def test_truncate_prose_never_exceeds_the_limit() -> None:
    """Sweep the boundary cases, which is where the first version was wrong.

    Slicing to `limit` and *then* appending an ellipsis returns limit + 1, which still fails
    `max_length`. That shipped once and cost a real paper on the first live run; the original
    tests missed it because their inputs never placed a word or sentence boundary at the cut.
    """
    from screener.domain.models import truncate_prose

    for limit in (10, 20, 240, 420):
        for suffix in ("", " x", " end.", " more words here", ".", " "):
            for n in range(limit - 3, limit + 40):
                text = "a" * n + suffix
                out = truncate_prose(text, limit)
                assert len(out) <= limit, f"limit={limit} n={n} suffix={suffix!r} -> {len(out)}"


def test_truncate_prose_prefers_a_sentence_boundary() -> None:
    from screener.domain.models import truncate_prose

    text = "First sentence about agents. Second sentence that will not fit at all."
    out = truncate_prose(text, 45)
    assert len(out) <= 45
    assert out.endswith(".")
    assert "Second" not in out


def test_truncate_prose_marks_a_hard_cut() -> None:
    from screener.domain.models import truncate_prose

    out = truncate_prose("z" * 500, 100)
    assert len(out) <= 100
    assert out.endswith("…")
