"""Composition root (§2 principle 5): the only place that knows which adapters exist.

Nothing below this layer imports a vendor SDK, and every test substitutes a fake here.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime

import structlog

from screener.adapters.arxiv import ArxivSource
from screener.adapters.heartbeat import HttpHeartbeat, NullHeartbeat
from screener.adapters.llm import OpenAILikeLLM
from screener.adapters.probe import GithubStarsProbe, OutcomeProbe
from screener.adapters.sqlite_repo import SqliteRepository
from screener.adapters.telegram import TelegramNotifier
from screener.config import Settings
from screener.ledger import Ledger


class ConfigError(RuntimeError):
    """Raised at startup, never mid-run: a missing credential must fail loudly and early."""


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)


@dataclass
class Deps:
    settings: Settings
    repo: SqliteRepository
    source: ArxivSource
    llm: OpenAILikeLLM
    notifier: TelegramNotifier
    heartbeat: HttpHeartbeat | NullHeartbeat
    probe: OutcomeProbe
    ledger: Ledger

    async def aclose(self) -> None:
        await self.source.aclose()
        await self.llm.aclose()
        await self.notifier.aclose()
        await self.heartbeat.aclose()
        await self.probe.aclose()
        self.repo.close()


def configure_logging(*, json_logs: bool = True, level: str = "INFO") -> None:
    logging.basicConfig(format="%(message)s", level=getattr(logging, level.upper(), logging.INFO))
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.JSONRenderer() if json_logs else structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        cache_logger_on_first_use=True,
    )


@asynccontextmanager
async def build_deps(cfg: Settings, *, require_network: bool = True) -> AsyncIterator[Deps]:
    """Build every adapter, or fail loudly.

    `require_network=False` (used by `doctor` and by tests) tolerates missing credentials so
    configuration problems can be *reported* rather than only *crashed on*.
    """
    configure_logging()
    repo = SqliteRepository(cfg.screener_db)
    repo.migrate()

    if require_network and not cfg.llm_api_key:
        repo.close()
        raise ConfigError(
            "No LLM credential: set DEEPSEEK_API_KEY (or LLM_API_KEY) in .env. A live run "
            "must not fall back to fake scores — that would produce a plausible digest from "
            "numbers nobody computed."
        )
    if require_network and (not cfg.telegram_bot_token or not cfg.telegram_chat_id):
        repo.close()
        raise ConfigError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must both be set for a live run")

    llm = OpenAILikeLLM(
        cfg.llm_api_key or "unused", base_url=cfg.llm_base_url, thinking=cfg.llm_thinking
    )
    source = ArxivSource(cfg.contact_email)
    notifier = TelegramNotifier(cfg.telegram_bot_token or "unused", cfg.telegram_chat_id or "0")
    heartbeat = HttpHeartbeat(cfg.heartbeat_url) if cfg.heartbeat_url else NullHeartbeat()
    probe = OutcomeProbe(GithubStarsProbe(cfg.github_token or None))

    deps = Deps(
        settings=cfg,
        repo=repo,
        source=source,
        llm=llm,
        notifier=notifier,
        heartbeat=heartbeat,
        probe=probe,
        ledger=Ledger(cfg.screener_budget_usd),
    )
    try:
        yield deps
    finally:
        await deps.aclose()
