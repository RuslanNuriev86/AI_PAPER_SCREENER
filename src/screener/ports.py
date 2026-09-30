"""Ports (§11). Seven protocols; every adapter is swappable and no vendor type crosses.

`Ledger` is deliberately **not** here: it has no external system behind it and no adapter,
so it is a pure domain class in `screener.ledger`. Calling it a port would imply a swap-in
implementation that does not and should not exist.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from screener.domain.models import (
    Assessment,
    CalibrationReport,
    GateResult,
    LedgerEntry,
    Outcome,
    OutcomeSignal,
    Paper,
    Profile,
    Prompt,
    Ranking,
    RevisitRun,
    RunStats,
    WatchlistEntry,
)
from screener.domain.types import (
    DeliveryKind,
    MessageId,
    PaperKey,
    Rung,
    RunId,
    Stage,
    Status,
)


@runtime_checkable
class Clock(Protocol):
    """Injected so the rung ladder is testable without waiting 14 days (§13.3)."""

    def now(self) -> datetime: ...


@runtime_checkable
class PaperSource(Protocol):
    async def fetch(self, since: datetime, until: datetime, profile: Profile) -> list[Paper]: ...


@runtime_checkable
class Enricher(Protocol):
    async def enrich(self, papers: Sequence[Paper]) -> Mapping[str, object]: ...


@runtime_checkable
class ImpactProbe(Protocol):
    """Measures what happened to a paper at a fixed age. Never on the delivery path."""

    async def measure(
        self, papers: Sequence[Paper], rung: Rung, now: datetime
    ) -> Mapping[str, OutcomeSignal]: ...


@runtime_checkable
class LLM(Protocol):
    async def parse[T: BaseModel](
        self,
        *,
        model: str,
        prompt: Prompt,
        payload: str,
        schema: type[T],
        temperature: float = 0.0,
    ) -> T: ...


@runtime_checkable
class Notifier(Protocol):
    async def send(self, chunks: Sequence[str]) -> list[MessageId]: ...  # one id per chunk


@runtime_checkable
class Heartbeat(Protocol):
    """Dead-man's-switch ping (§12.2). Absence is the signal, so this fires on failure too."""

    async def ping(self, *, ok: bool, detail: str = "") -> None: ...


@runtime_checkable
class Repository(Protocol):
    # --- lifecycle: the runs row exists before anything FKs to it (§10) ---
    def begin_run(self, now: datetime, config_hash: str, mode: str) -> RunId: ...
    def finish_run(
        self, run_id: RunId, status: Status, stats: RunStats, cost_usd: float
    ) -> None: ...
    def begin_revisit(self, now: datetime) -> str: ...
    def finish_revisit(self, run: RevisitRun) -> None: ...

    # --- delivery path ---
    def seen(self, keys: Sequence[PaperKey]) -> set[PaperKey]: ...  # == gate_results (§10)
    def save_papers(self, papers: Sequence[Paper]) -> None: ...
    def save_gate_results(self, run_id: RunId, results: Sequence[GateResult]) -> None: ...
    def delivered_versions(self, arxiv_ids: Sequence[str]) -> Mapping[str, set[int]]: ...
    def latest_version(self, arxiv_id: str) -> int: ...
    def save_assessment(self, run_id: RunId, a: Assessment) -> None: ...
    def save_rankings(self, run_id: RunId, ranked: Sequence[Ranking]) -> None: ...
    def record_delivery(
        self,
        run_id: RunId,
        picks: Sequence[Ranking],
        kind: DeliveryKind,
        message_of: Mapping[str, str],
    ) -> None: ...
    def ledger_add(self, run_id: RunId, entry: LedgerEntry) -> None: ...
    def run_cost(self, run_id: RunId) -> float: ...

    # --- maturity loop (§6.6) — separate command, same store ---
    def enrol(self, entries: Sequence[WatchlistEntry]) -> None: ...
    def watchlist(self) -> list[WatchlistEntry]: ...
    def measured_rungs(self) -> set[tuple[str, int]]: ...
    def save_outcomes(self, outcomes: Sequence[Outcome]) -> None: ...
    def papers_by_key(self, keys: Sequence[PaperKey]) -> dict[PaperKey, Paper]: ...
    def save_calibration(self, report: CalibrationReport) -> None: ...
    def get_state(self, key: str) -> str | None: ...
    def set_state(self, key: str, value: str) -> None: ...


@runtime_checkable
class FeedbackSource(Protocol):
    """Replies to the bot's own messages. Text only — reactions cannot fire in a DM (§8.3)."""

    async def poll(self) -> list[tuple[MessageId, str, datetime]]: ...


__all__ = [
    "LLM",
    "Clock",
    "Enricher",
    "FeedbackSource",
    "Heartbeat",
    "ImpactProbe",
    "Notifier",
    "PaperSource",
    "Repository",
    "Rung",
    "Stage",
    "Status",
    "date",
]
