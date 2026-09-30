"""Shared fixtures. Every test runs without network and without a real database."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from screener.adapters.sqlite_repo import SqliteRepository
from screener.domain.models import Profile


def _normalise_proxy_env() -> None:
    """Drop NO_PROXY entries in a form httpx cannot parse.

    httpx parses every NO_PROXY entry while *constructing* a client, so a single bracketed
    IPv6 literal ("[::1]") raises InvalidURL and no HTTP client can exist in the process. That
    is a host misconfiguration, not product behaviour — `screener doctor` detects and reports
    it — so tests normalise it rather than asserting on whichever host they run on.
    """
    for var in ("NO_PROXY", "no_proxy"):
        raw = os.environ.get(var)
        if not raw:
            continue
        fixed = [
            e for e in (part.strip() for part in raw.split(",")) if e and not e.startswith("[")
        ]
        os.environ[var] = ",".join(fixed)


# Runs at import, before any test constructs an httpx client.
_normalise_proxy_env()

NOW = datetime(2025, 10, 1, 12, 0, tzinfo=UTC)


@pytest.fixture
def profile() -> Profile:
    """A small but realistic profile: strong terms, weak terms, and real exclusions."""
    return Profile(
        name="test profile",
        categories=["cs.AI", "cs.CL", "cs.LG", "cs.MA"],
        strong_terms=["LLM agent", "agentic", "tool use", "agent benchmark", "trajectory"],
        weak_terms=["agent", "planning", "reflection"],
        exclude_patterns=[
            "agent-based model",
            "(power grid|traffic|epidemic|market).{0,40}multi-agent",
            "multi-agent.{0,40}(power grid|traffic|epidemic|market)",
            r"\b(nanoparticle|pharmacological|biological) agent",
        ],
        boost_topics=[
            "evaluation & benchmarks",
            "multi-agent coordination",
            "memory & context",
            "computer use",
            "safety & oversight",
            "agentic RL",
            "infrastructure/protocols",
        ],
        lookback_days=5,
    )


@pytest.fixture
def repo(tmp_path: Path):
    r = SqliteRepository(tmp_path / "test.db")
    r.migrate()
    yield r
    r.close()


@pytest.fixture
def now() -> datetime:
    return NOW


@pytest.fixture(autouse=True)
def _isolated_outbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the outbox at a per-test directory.

    An earlier version of the suite resolved the outbox relative to the working directory, so a
    pipeline test found a real stranded digest in `./outbox`, "sent" it through a fake notifier,
    and deleted it. Tests must not be able to touch anything outside their tmp dir.
    """
    monkeypatch.setenv("SCREENER_OUTBOX", str(tmp_path / "outbox"))
