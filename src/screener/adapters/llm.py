"""LLM adapter: structured output over an OpenAI-compatible chat-completions API (§11).

One implementation serves every provider in the dependency budget, because the `LLM` port is
the only shape the pipeline knows. The generic `parse[T]` signature is PEP 695 syntax and is
the reason the project requires Python >= 3.12.
"""

from __future__ import annotations

import json
import re
from typing import Any

import httpx
import structlog
from pydantic import BaseModel, ValidationError

from screener.domain.models import LedgerEntry, Prompt
from screener.ledger import estimate

log = structlog.get_logger(__name__)

#: OpenAI-shaped chat-completions endpoint. DeepSeek serves one at this root;
#: the Anthropic-shaped endpoint is https://api.deepseek.com/anthropic instead.
DEFAULT_BASE_URL = "https://api.deepseek.com"

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json(text: str) -> str:
    """Pull a JSON object out of a model response.

    Models wrap JSON in prose or fences even when told not to, so this is defensive: first
    try the whole string, then a fenced block, then the outermost brace pair.
    """
    stripped = text.strip()
    try:
        json.loads(stripped)
        return stripped
    except json.JSONDecodeError:
        pass
    if m := _FENCE_RE.search(stripped):
        return m.group(1).strip()
    start, end = stripped.find("{"), stripped.rfind("}")
    if start != -1 and end > start:
        return stripped[start : end + 1]
    return stripped


class OpenAILikeLLM:
    """Implements `ports.LLM`."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 120.0,
        thinking: bool = False,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        #: DeepSeek enables thinking mode by default, and thinking mode **ignores
        #: `temperature`** — it does not error, it silently does nothing. Scoring has to be
        #: repeatable (the rubric is meant to be auditable), so the default is off. See
        #: DESIGN.md §7.
        self.thinking = thinking
        self._client = httpx.AsyncClient(
            timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}"},
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> OpenAILikeLLM:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def parse[T: BaseModel](
        self,
        *,
        model: str,
        prompt: Prompt,
        payload: str,
        schema: type[T],
        temperature: float = 0.0,
    ) -> T:
        """One structured call. Retries once with the validation error appended.

        The repair retry is a §12.1 requirement: a malformed generation is a per-paper
        failure, not a run failure, and the second failure drops the paper.
        """
        messages: list[dict[str, str]] = [
            {"role": "system", "content": prompt.body},
            {"role": "user", "content": payload},
        ]
        last_error: Exception | None = None
        for attempt in range(2):
            body: dict[str, Any] = {
                "model": model,
                "messages": messages,
                # JSON output is supported by deepseek-flash and is what makes the Pydantic
                # validation below the only parse step.
                "response_format": {"type": "json_object"},
                # Non-thinking mode honours temperature; thinking mode ignores it entirely.
                "thinking": {"type": "enabled" if self.thinking else "disabled"},
            }
            if not self.thinking:
                body["temperature"] = temperature
            resp = await self._client.post(f"{self.base_url}/chat/completions", json=body)
            resp.raise_for_status()
            data = resp.json()
            content = str(data["choices"][0]["message"]["content"])
            try:
                return schema.model_validate_json(extract_json(content))
            except (ValidationError, json.JSONDecodeError, ValueError) as exc:
                last_error = exc
                log.warning("llm.invalid_json", attempt=attempt, schema=schema.__name__)
                messages = [
                    *messages,
                    {"role": "assistant", "content": content},
                    {
                        "role": "user",
                        "content": f"That failed validation:\n{exc}\nReturn corrected JSON only.",
                    },
                ]
        raise RuntimeError(f"{schema.__name__} failed validation twice: {last_error}")

    def cost_of(self, model: str, response: dict[str, Any]) -> LedgerEntry:
        usage = response.get("usage") or {}
        inp = int(usage.get("prompt_tokens") or 0)
        out = int(usage.get("completion_tokens") or 0)
        return LedgerEntry(
            stage="review",
            model=model,
            input_tokens=inp,
            output_tokens=out,
            usd=estimate(model, inp, out),
        )


class FakeLLM:
    """Deterministic stand-in for tests and `--dry-run` with no credentials.

    Never used in a live run: `build_deps` raises if no API key is configured, so a missing
    key fails loudly at startup rather than producing a plausible digest from fake scores.
    """

    def __init__(self, *, scores: dict[str, float] | None = None) -> None:
        self.calls: list[str] = []
        self._scores = scores or {
            "relevance": 7.0,
            "novelty": 7.0,
            "rigor": 6.0,
            "evidence_strength": 6.0,
            "impact_forecast": 7.0,
            "reproducibility": 5.0,
        }

    async def parse[T: BaseModel](
        self, *, model: str, prompt: Prompt, payload: str, schema: type[T], temperature: float = 0.0
    ) -> T:
        self.calls.append(payload[:200])
        title = payload.split("\n", 1)[0][:80]
        return schema.model_validate(
            {
                "tldr": f"Deterministic placeholder summary for {title}"[:220],
                "what_they_did": "Placeholder: the mechanism would be described here."[:420],
                "why_it_matters": "Placeholder: the significance would be argued here."[:420],
                "caveats": "Placeholder: the honest weakness would go here."[:240],
                "lenses": ["method"],
                "tags": ["evaluation & benchmarks"],
                "evidence_quotes": [],
                "scores": self._scores,
                "soft_flags": [],
                "hard_flag": None,
            }
        )
