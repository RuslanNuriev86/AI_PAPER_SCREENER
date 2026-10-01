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
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import feedparser  # type: ignore[import-untyped]
import httpx
import structlog

from screener.domain.models import Paper, Profile

log = structlog.get_logger(__name__)

BASE = "https://export.arxiv.org/api/query"
MIN_INTERVAL_S = 3.5  # ToU says 3; a small margin absorbs clock jitter
MAX_INTERVAL_S = 60.0  # ceiling for adaptive backoff once arXiv says 429

#: ToU: "max_results <= 2000 per slice". Requesting large pages is the single biggest lever on
#: request count: at 100 per page a 5-day window across 8 categories costs ~30 requests, at 1000
#: it costs ~8. arXiv rate-limits by request, so this is the difference between one digest and a
#: 429 storm.
PAGE_SIZE = 1000
MAX_RESULTS_PER_SLICE = 2000
MAX_ATTEMPTS = 3

#: Stop querying new categories after this many seconds and use what has arrived. A digest built
#: from six categories is worth having; one that never finishes is not.
FETCH_BUDGET_S = 150.0

#: Consecutive category failures after which the source is treated as *down* rather than slow.
#: Without this, eight categories x three attempts x a read timeout is a quarter of an hour
#: before the run gives up, when the answer was clear after the first one.
SYSTEMIC_FAILURE_THRESHOLD = 2


class SourceUnavailable(RuntimeError):
    """The paper source could not be reached at all.

    Distinct from "the source returned nothing": §12.1 treats those differently, and only this
    one is allowed to abort a run — with a one-line notice, not a traceback.
    """


#: arXiv ids are 'YYMM.NNNNN' (new style) or 'archive/YYMMNNN' (old style).
_ID_RE = re.compile(r"abs/([^v]+?)(?:v(\d+))?$")


class ArxivSource:
    """Implements `ports.PaperSource`."""

    def __init__(
        self,
        contact_email: str,
        *,
        timeout: float = 60.0,
        page_size: int = PAGE_SIZE,
        cache_dir: str | Path | None = None,
        budget_s: float = FETCH_BUDGET_S,
    ) -> None:
        self.contact_email = contact_email
        self.page_size = page_size
        self.budget_s = budget_s
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.failed_categories: list[str] = []
        self.cache_hits = 0
        self._interval = MIN_INTERVAL_S
        self._last_call: float = 0.0
        self._lock = asyncio.Lock()
        self._client = httpx.AsyncClient(
            # A read timeout is what actually fires behind a slow proxy: the connection is
            # established and the *body* stalls. A separate, shorter connect timeout keeps a
            # dead host from consuming the whole read budget.
            timeout=httpx.Timeout(timeout, connect=15.0),
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
        """Space requests by the current interval, which grows when arXiv pushes back."""
        async with self._lock:
            loop = asyncio.get_running_loop()
            elapsed = loop.time() - self._last_call
            if elapsed < self._interval:
                await asyncio.sleep(self._interval - elapsed)
            self._last_call = asyncio.get_running_loop().time()

    def _slow_down(self) -> None:
        """429 means "you are going too fast" — believe it and widen the gap."""
        self._interval = min(self._interval * 2, MAX_INTERVAL_S)

    def _speed_up(self) -> None:
        self._interval = max(MIN_INTERVAL_S, self._interval / 2)

    # -- response cache ------------------------------------------------------------------
    # The ToU asks for it explicitly: "no need to call more than once a day — please cache".
    # Beyond being polite it is what stops a day of re-runs from turning into a 429 storm.

    def _cache_path(self, category: str, since: datetime, until: datetime) -> Path | None:
        if self.cache_dir is None:
            return None
        key = f"{category}_{since.date().isoformat()}_{until.date().isoformat()}"
        digest = hashlib.sha256(key.encode()).hexdigest()[:16]
        return self.cache_dir / f"{category.replace('/', '_')}-{digest}.json"

    def _cache_read(self, category: str, since: datetime, until: datetime) -> list[Paper] | None:
        path = self._cache_path(category, since, until)
        if path is None or not path.exists():
            return None
        # Valid for the calendar day it was written: `updated` only changes at midnight.
        if datetime.fromtimestamp(path.stat().st_mtime, UTC).date() != datetime.now(UTC).date():
            return None
        try:
            raw = json.loads(path.read_text())
        except (OSError, ValueError):
            return None
        return [Paper.model_validate(item) for item in raw]

    def _cache_write(
        self, category: str, since: datetime, until: datetime, papers: list[Paper]
    ) -> None:
        path = self._cache_path(category, since, until)
        if path is None or not papers:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps([p.model_dump(mode="json") for p in papers]))
        except OSError as exc:
            log.warning("arxiv.cache_write_failed", error=str(exc)[:120])

    async def fetch(self, since: datetime, until: datetime, profile: Profile) -> list[Paper]:
        """Fetch papers announced in [since, until) across the profile's categories.

        Sequential by construction: the ToU requires one connection at a time (see the module
        docstring). Three properties are deliberately non-negotiable here, because the first
        version had none of them and a single flaky afternoon produced a ten-minute run that
        ended in a traceback:

        * **cached per category per day.** The ToU asks for it ("no need to call more than once
          a day"), and it is what stops re-runs from becoming a rate-limit ban.
        * **partial coverage is a success.** One dead category must not lose the seven that
          worked.
        * **bounded.** After `budget_s` the fetch stops querying and returns what it has. A
          digest from six categories beats one that never finishes.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.budget_s
        seen: dict[tuple[str, int], Paper] = {}
        self.failed_categories = []
        self.cache_hits = 0
        budget_exhausted = False

        for category in profile.categories:
            cached = self._cache_read(category, since, until)
            if cached is not None:
                self.cache_hits += 1
                for paper in cached:
                    seen[paper.key] = paper
                continue

            if loop.time() >= deadline:
                budget_exhausted = True
                log.warning("arxiv.budget_exhausted", unqueried=len(profile.categories))
                break

            try:
                papers = await self._category(category, since, until)
            except httpx.HTTPError as exc:
                # One dead category must not lose the seven that worked. Partial coverage is a
                # better digest than none, and the failure is recorded for the run stats.
                self.failed_categories.append(category)
                log.warning(
                    "arxiv.category_failed",
                    category=category,
                    error_type=type(exc).__name__,
                    error=str(exc) or repr(exc),
                )
                if not seen and len(self.failed_categories) >= SYSTEMIC_FAILURE_THRESHOLD:
                    raise SourceUnavailable(
                        f"{len(self.failed_categories)} categories failed with no results "
                        f"({type(exc).__name__}); treating the source as unavailable"
                    ) from exc
                continue

            self._cache_write(category, since, until, papers)
            for paper in papers:
                seen[paper.key] = paper

        papers = list(seen.values())
        if not papers:
            raise SourceUnavailable(
                "no papers retrieved"
                + (
                    f" ({len(self.failed_categories)} categories failed)"
                    if self.failed_categories
                    else ""
                )
            )
        log.info(
            "arxiv.fetched",
            count=len(papers),
            categories=len(profile.categories),
            failed_categories=len(self.failed_categories),
            cache_hits=self.cache_hits,
            budget_exhausted=budget_exhausted,
            since=since.isoformat(),
            until=until.isoformat(),
        )
        return papers

    async def _category(self, category: str, since: datetime, until: datetime) -> list[Paper]:
        """Fetch one category, paginating with large pages. Raises on transport failure."""
        out: list[Paper] = []
        start = 0
        while start < MAX_RESULTS_PER_SLICE:
            entries = await self._page(category, since, until, start)
            for entry in entries:
                paper = self._to_paper(entry)
                if paper is not None:
                    out.append(paper)
            if len(entries) < self.page_size:
                break
            start += self.page_size
        return out

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
        last: Exception | None = None
        for attempt in range(MAX_ATTEMPTS):
            await self._throttle()
            try:
                resp = await self._client.get(BASE, params=params)
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                last = exc
                # `str(exc)` is EMPTY for httpx timeouts, which made the original log line
                # useless ("error": ""). Always log the type, and fall back to repr.
                log.warning(
                    "arxiv.page_failed",
                    category=category,
                    start=start,
                    attempt=attempt,
                    interval=round(self._interval, 1),
                    error_type=type(exc).__name__,
                    error=str(exc) or repr(exc),
                )
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status == 429:
                    self._slow_down()
                    retry_after = _retry_after(exc)
                    wait = retry_after if retry_after is not None else self._interval
                    await asyncio.sleep(wait)
                elif attempt < MAX_ATTEMPTS - 1:
                    await asyncio.sleep(2**attempt)
                continue
            self._speed_up()
            parsed = feedparser.parse(resp.text)
            return list(parsed.entries)
        assert last is not None
        raise last

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


def _retry_after(exc: httpx.HTTPError) -> float | None:
    """Honour a Retry-After header when arXiv sends one."""
    response = getattr(exc, "response", None)
    if response is None:
        return None
    raw = response.headers.get("Retry-After")
    if not raw:
        return None
    try:
        return min(float(raw), MAX_INTERVAL_S)
    except ValueError:
        return None
