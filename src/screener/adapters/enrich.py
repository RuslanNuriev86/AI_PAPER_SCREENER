"""Enrichment: the measured half of the rating (§6.1). Core path, not an optional tier.

Three sources, each reading a different signal, each best-effort. What matters is that the
values are *read*, with the age they were read at, and that a source which fails leaves its
dimension absent rather than zero (§6.5).

Measurement drove two design decisions here (§6.1.0):

* **OpenAlex rather than Semantic Scholar** for citations and venue. S2's unauthenticated batch
  endpoint returned 429 on the first attempt; OpenAlex answered 200 and indexes arXiv by its
  `10.48550/arXiv.*` DOI.
* **the repo-age guard.** One sampled cohort contained a 34,432-star repository — a paper linking
  to a pre-existing popular project, not that paper's own traction. `repo_created_days_before_paper`
  is recorded so `signals.map_signals` can refuse to credit stars that predate the work.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import datetime

import httpx
import structlog

from screener.domain.models import Enrichment, Paper

log = structlog.get_logger(__name__)

OPENALEX = "https://api.openalex.org/works"
GITHUB_API = "https://api.github.com/repos"
GITHUB_RE = re.compile(r"github\.com/([^/]+)/([^/#?\s]+)", re.IGNORECASE)

#: OpenAlex accepts up to 50 OR-ed filter values per request.
OPENALEX_BATCH = 40


class Enricher:
    """Implements `ports.Enricher`."""

    def __init__(
        self,
        contact_email: str,
        *,
        github_token: str | None = None,
        timeout: float = 40.0,
    ) -> None:
        self.contact_email = contact_email
        headers = {"User-Agent": f"agent-papers-daily/0.1 (+mailto:{contact_email})"}
        self._http = httpx.AsyncClient(timeout=timeout, headers=headers)
        self._github_token = github_token
        self.sources_ok: dict[str, int] = {}

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> Enricher:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def enrich(self, papers: Sequence[Paper]) -> Mapping[str, Enrichment]:
        out: dict[str, Enrichment] = {
            p.arxiv_id: Enrichment(arxiv_id=p.arxiv_id, version=p.version) for p in papers
        }
        self.sources_ok = {}

        await self._citations(papers, out)
        await self._repos(papers, out)

        for arxiv_id, e in out.items():
            out[arxiv_id] = e.model_copy(update={"sources_ok": sorted(set(e.sources_ok))})
        log.info(
            "enrich.done",
            papers=len(out),
            with_citations=sum(1 for e in out.values() if e.citations is not None),
            with_stars=sum(1 for e in out.values() if e.stars is not None),
            with_venue=sum(1 for e in out.values() if e.venue),
            sources=self.sources_ok,
        )
        return out

    async def _citations(self, papers: Sequence[Paper], out: dict[str, Enrichment]) -> None:
        """Citations and venue via OpenAlex, in batches keyed by arXiv DOI."""
        by_doi: dict[str, str] = {
            f"10.48550/arxiv.{p.arxiv_id}".lower(): p.arxiv_id for p in papers
        }
        dois = list(by_doi)
        for start in range(0, len(dois), OPENALEX_BATCH):
            chunk = dois[start : start + OPENALEX_BATCH]
            try:
                resp = await self._http.get(
                    OPENALEX,
                    params={
                        "filter": "doi:" + "|".join(chunk),
                        "per-page": OPENALEX_BATCH,
                        "select": "doi,cited_by_count,primary_location,type",
                        "mailto": self.contact_email,
                    },
                )
                if resp.status_code != 200:
                    log.warning("enrich.openalex_status", status=resp.status_code)
                    continue
                results = resp.json().get("results") or []
            except httpx.HTTPError as exc:
                log.warning("enrich.openalex_failed", error=str(exc)[:120])
                continue
            self.sources_ok["openalex"] = self.sources_ok.get("openalex", 0) + len(results)
            for work in results:
                doi = str(work.get("doi") or "").lower().replace("https://doi.org/", "")
                arxiv_id = by_doi.get(doi)
                if arxiv_id is None:
                    continue
                source = (work.get("primary_location") or {}).get("source") or {}
                name = source.get("display_name")
                # "arXiv (Cornell University)" is the preprint record, not a venue — reporting it
                # as one would fabricate an acceptance.
                venue = None if not name or "arxiv" in str(name).lower() else str(name)
                current = out[arxiv_id]
                out[arxiv_id] = current.model_copy(
                    update={
                        "citations": int(work.get("cited_by_count") or 0),
                        "venue": venue,
                        "sources_ok": [*current.sources_ok, "openalex"],
                    }
                )

    async def _repos(self, papers: Sequence[Paper], out: dict[str, Enrichment]) -> None:
        """Stars per linked repo, plus how long the repo predates the paper."""
        headers = {"Authorization": f"Bearer {self._github_token}"} if self._github_token else {}
        for paper in papers:
            match = GITHUB_RE.search(paper.code_url or "")
            if match is None:
                continue
            owner, repo = match.group(1), match.group(2).removesuffix(".git")
            try:
                resp = await self._http.get(f"{GITHUB_API}/{owner}/{repo}", headers=headers)
                if resp.status_code != 200:
                    # A deleted or renamed repo is *no measurement*, not zero stars.
                    continue
                data = resp.json()
            except httpx.HTTPError as exc:
                log.warning("enrich.github_failed", repo=f"{owner}/{repo}", error=str(exc)[:100])
                continue
            self.sources_ok["github"] = self.sources_ok.get("github", 0) + 1
            created = data.get("created_at")
            days_before: int | None = None
            if created:
                try:
                    created_at = datetime.fromisoformat(str(created).replace("Z", "+00:00"))
                    days_before = (paper.submitted_at - created_at).days
                except ValueError:
                    days_before = None
            current = out[paper.arxiv_id]
            out[paper.arxiv_id] = current.model_copy(
                update={
                    "stars": int(data.get("stargazers_count") or 0),
                    "repo_url": f"https://github.com/{owner}/{repo}",
                    "repo_created_days_before_paper": days_before,
                    "sources_ok": [*current.sources_ok, "github"],
                }
            )
