"""Heartbeat adapter — dead-man's switch (§12.2, §12.1).

The external monitor (healthchecks.io, Uptime Kuma) alerts on *absence*, so the ping fires
on success, on `empty`, and on failure alike. Pinging only on success would make an outage
indistinguishable from "the job is running fine" — which is exactly the silent-degradation
failure mode §1.1 exists to prevent.
"""

from __future__ import annotations

import httpx
import structlog

log = structlog.get_logger(__name__)


class HttpHeartbeat:
    """Implements `ports.Heartbeat`."""

    def __init__(self, url: str | None, *, timeout: float = 10.0) -> None:
        self.url = url
        self._client = httpx.AsyncClient(timeout=timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def ping(self, *, ok: bool, detail: str = "") -> None:
        if not self.url:
            log.debug("heartbeat.skipped", reason="no HEARTBEAT_URL configured")
            return
        target = self.url if ok else f"{self.url.rstrip('/')}/fail"
        try:
            await self._client.get(target, params={"detail": detail[:200]} if detail else None)
        except httpx.HTTPError as exc:
            # A failing heartbeat must never take down the run it is reporting on.
            log.warning("heartbeat.ping_failed", error=str(exc), target=target)


class NullHeartbeat:
    """Used in tests and dry runs."""

    async def ping(self, *, ok: bool, detail: str = "") -> None:
        return None

    async def aclose(self) -> None:
        return None
