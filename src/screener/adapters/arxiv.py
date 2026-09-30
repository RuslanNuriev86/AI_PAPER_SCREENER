"""arXiv Atom API client (§4 stage 1, §14).

Two ToU constraints drive the shape of this class, both verified against
https://info.arxiv.org/help/api/tou.html:

* "make no more than one request every three seconds", and
* "limit requests to a single connection at a time".

The second is the one that is easy to violate by accident: the ~8 daily queries are issued
**sequentially on one client**, never via `asyncio.gather`. Parallelising them would be a
terms violation, not just a slowdown.
"""

from __future__ import annotations

import asyncio
import re
from datetime import UTC, datetime
from typing import Any

import feedparser  # type: ignore[import-untyped]
import httpx
import structlog

from screener.domain.models import Paper, Profile

log = structlog.get_logger(__name__)

BASE = "https://export.arxiv.org/api/query"
MIN_INTERVAL_S = 3.5  # ToU says 3; a small margin absorbs clock jitter
MAX_RESULTS_PER_SLICE = 2000

#: arXiv ids are 'YYMM.NNNNN' (new style) or 'archive/YYMMNNN' (old style).
_ID_RE = re.compile(r"abs/([^v]+?)(?:v(\d+))?$")


class ArxivSource:
    """Implements `ports.PaperSource`."""

    def __init__(self, contact_email: str, *, timeout: float = 30.0, page_size: int = 100) -> None:
        self.contact_email = contact_email
        self.page_size = page_size
        self._last_call: float = 0.0
        self._lock = asyncio.Lock()
        self._client = httpx.AsyncClient(
            timeout=timeout,
            headers={
                # arXiv asks for a descriptive User-Agent including a contact address.
                "User-Agent": f"agent-papers-daily/0.1 (+mailto:{contact_email})",
            },
            limits=httpx.Limits(max_connections=1, max_keepalive_connections=1),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> ArxivSource:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def _throttle(self) -> None:
        async with self._lock:
            loop = asyncio.get_running_loop()
            elapsed = loop.time() - self._last_call
            if elapsed < MIN_INTERVAL_S:
                await asyncio.sleep(MIN_INTERVAL_S - elapsed)
            self._last_call = asyncio.get_running_loop().time()

    async def fetch(self, since: datetime, until: datetime, profile: Profile) -> list[Paper]:
        """Fetch papers announced in [since, until) across the profile's categories.

        Queries are sequential by construction (see the module docstring). Pagination stops
        when a slice returns fewer than `page_size` entries.
        """
        seen: dict[tuple[str, int], Paper] = {}
        for category in profile.categories:
            start = 0
            while start < MAX_RESULTS_PER_SLICE:
                entries = await self._page(category, since, until, start)
                for entry in entries:
                    paper = self._to_paper(entry)
                    if paper is not None:
                        seen[paper.key] = paper
                if len(entries) < self.page_size:
                    break
                start += self.page_size
        papers = list(seen.values())
        log.info(
            "arxiv.fetched",
            count=len(papers),
            categories=len(profile.categories),
            since=since.isoformat(),
            until=until.isoformat(),
        )
        return papers

    async def _page(self, category: str, since: datetime, until: datetime, start: int) -> list[Any]:
        params: dict[str, str | int] = {
            "search_query": (
                f"cat:{category} AND submittedDate:[{_stamp(since)} TO {_stamp(until)}]"
            ),
            "start": start,
            "max_results": self.page_size,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        for attempt in range(3):
            await self._throttle()
            try:
                resp = await self._client.get(BASE, params=params)
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                wait = 2**attempt
                log.warning("arxiv.page_failed", category=category, attempt=attempt, error=str(exc))
                if attempt == 2:
                    raise
                await asyncio.sleep(wait)
                continue
            parsed = feedparser.parse(resp.text)
            return list(parsed.entries)
        return []

    def _to_paper(self, entry: Any) -> Paper | None:
        link = str(entry.get("id", ""))
        m = _ID_RE.search(link)
        if not m:
            return None
        arxiv_id = m.group(1)
        version = int(m.group(2) or 1)
        categories = [str(t.get("term", "")) for t in entry.get("tags", []) or []]
        published = _parse_dt(entry.get("published")) or datetime.now(UTC)
        updated = _parse_dt(entry.get("updated"))
        comment = entry.get("arxiv_comment") or None
        return Paper(
            arxiv_id=arxiv_id,
            version=version,
            title=" ".join(str(entry.get("title", "")).split()),
            abstract=" ".join(str(entry.get("summary", "")).split()),
            authors=[str(a.get("name", "")) for a in entry.get("authors", []) or []],
            categories=categories,
            primary_category=str(entry.get("arxiv_primary_category", {}).get("term", ""))
            if isinstance(entry.get("arxiv_primary_category"), dict)
            else (categories[0] if categories else ""),
            submitted_at=published,
            updated_at=updated,
            abs_url=f"https://arxiv.org/abs/{arxiv_id}",
            pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
            comment=str(comment) if comment else None,
            code_url=_extract_code_url(comment, str(entry.get("summary", ""))),
            first_seen_at=datetime.now(UTC),
        )


def _stamp(dt: datetime) -> str:
    return dt.astimezone(UTC).strftime("%Y%m%d%H%M")


def _parse_dt(value: object) -> datetime | None:
    if not value:
        return None
    text = str(value)
    try:
        return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    except ValueError:
        return None


_URL_RE = re.compile(
    r"https?://(?:www\.)?(github\.com|gitlab\.com|huggingface\.co)/[^\s)\]]+", re.I
)


def _extract_code_url(comment: object, abstract: str) -> str | None:
    """Resolve a repo URL from the comment field or abstract.

    This runs at *fetch* time and populates `papers.code_url`, which is what the T+14 probe
    reads for stars. It deliberately does not depend on the v1.5 `Enricher`, so the v0 rung
    works without enrichment (§6.6.2).
    """
    for haystack in (str(comment or ""), abstract):
        m = _URL_RE.search(haystack)
        if m:
            return m.group(0).rstrip(".,;")
    return None
