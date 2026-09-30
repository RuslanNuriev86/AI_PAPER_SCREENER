"""Outcome probes for the maturity loop (§6.6.2).

v0 measures the **T+14 rung only**, and only the two signals that exist at that age: GitHub
stars on the repo named by `papers.code_url`, and HF Daily upvotes. Citations are near-zero
at two weeks and venue acceptance cannot exist yet, so probing them here would add cost and
produce noise.

Probes are best-effort and per-source: a dead source drops one component and the grade
renormalises, never zeroes (§6.6.3). This adapter is never on the delivery path.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import datetime

import httpx
import structlog

from screener.domain.models import OutcomeSignal, Paper
from screener.domain.types import Rung

log = structlog.get_logger(__name__)

_GITHUB_RE = re.compile(r"github\.com/([^/]+)/([^/#?\s]+)", re.I)


class GithubStarsProbe:
    """Star counts via the GitHub REST API.

    Unauthenticated access is 60 requests/hour, which a daily cohort exceeds on its own —
    so a token is required in production and `screener doctor` validates it (§12.3).
    """

    def __init__(self, token: str | None = None, *, timeout: float = 20.0) -> None:
        self.token = token
        headers = {"Accept": "application/vnd.github+json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self._client = httpx.AsyncClient(timeout=timeout, headers=headers)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> GithubStarsProbe:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def stars(self, url: str | None) -> int | None:
        if not url:
            return None
        m = _GITHUB_RE.search(url)
        if not m:
            return None
        owner, repo = m.group(1), m.group(2).removesuffix(".git")
        try:
            resp = await self._client.get(f"https://api.github.com/repos/{owner}/{repo}")
            if resp.status_code == 404:
                # A deleted or renamed repo is not zero stars, it is *no measurement*.
                return None
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            log.warning("probe.github_failed", repo=f"{owner}/{repo}", error=str(exc))
            return None
        return int(resp.json().get("stargazers_count") or 0)


class OutcomeProbe:
    """Implements `ports.ImpactProbe`, composing the individual probes."""

    def __init__(
        self, github: GithubStarsProbe, *, hf_upvotes: Mapping[str, int] | None = None
    ) -> None:
        self.github = github
        self._hf = dict(hf_upvotes or {})

    async def aclose(self) -> None:
        await self.github.aclose()

    async def __aenter__(self) -> OutcomeProbe:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def measure(
        self, papers: Sequence[Paper], rung: Rung, now: datetime
    ) -> Mapping[str, OutcomeSignal]:
        out: dict[str, OutcomeSignal] = {}
        sources_ok: list[str] = []
        for p in papers:
            stars = await self.github.stars(p.code_url)
            hf = self._hf.get(p.arxiv_id)
            ok: list[str] = []
            if stars is not None:
                ok.append("github")
            if hf is not None:
                ok.append("hf_daily")
            out[p.arxiv_id] = OutcomeSignal(
                arxiv_id=p.arxiv_id,
                stars=stars,
                hf_upvotes=hf,
                code_url=p.code_url,
                sources_ok=ok,
                raw={"rung": rung, "measured_at": now.isoformat()},
            )
            sources_ok.extend(ok)
        log.info("probe.measured", papers=len(out), sources=sorted(set(sources_ok)))
        return out
