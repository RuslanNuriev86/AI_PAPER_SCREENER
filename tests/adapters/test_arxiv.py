"""arXiv adapter behaviour under an outage (§12.1, §14).

The first real outage produced a ten-minute run and a 200-line traceback for a condition the
design says to handle with a one-line notice. These tests pin the three properties that fix it.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from screener.adapters.arxiv import (
    PAGE_SIZE,
    ArxivSource,
    SourceUnavailable,
)

NOW = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)


def _source(tmp_path: Path) -> ArxivSource:
    return ArxivSource("test@example.com", cache_dir=tmp_path / "arxiv", budget_s=5.0)


def _boom(*_a: object, **_kw: object) -> None:
    raise httpx.ConnectError("down")


def test_large_pages_keep_the_request_count_small() -> None:
    """ToU allows up to 2000 per slice; asking for 100 per page wasted ~30 requests a run."""
    assert PAGE_SIZE >= 500


async def test_a_single_dead_category_does_not_lose_the_others(tmp_path, profile) -> None:
    src = _source(tmp_path)
    calls = {"n": 0}

    async def fake_category(category: str, since: datetime, until: datetime):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        if category == "cs.CL":
            raise httpx.ConnectError("dead category")
        from tests.factories import make_paper

        return [make_paper(arxiv_id=f"2509.{calls['n']:05d}", categories=[category])]

    src._category = fake_category  # type: ignore[method-assign]
    try:
        papers = await src.fetch(NOW - timedelta(days=5), NOW, profile)
    finally:
        await src.aclose()
    assert papers, "the seven working categories must still produce a digest"
    assert src.failed_categories == ["cs.CL"]


async def test_a_total_outage_raises_source_unavailable_fast(tmp_path, profile) -> None:
    """The circuit breaker exists so this takes seconds, not a quarter of an hour."""
    src = _source(tmp_path)
    src._category = _boom  # type: ignore[method-assign]
    try:
        with pytest.raises(SourceUnavailable):
            await src.fetch(NOW - timedelta(days=5), NOW, profile)
    finally:
        await src.aclose()


async def test_the_fetch_stops_at_its_budget_and_returns_what_it_has(tmp_path, profile) -> None:
    """A digest from six categories beats one that never finishes."""
    src = _source(tmp_path)
    src.budget_s = 0.0  # already exhausted on entry

    async def never_called(*_a: object, **_kw: object):  # type: ignore[no-untyped-def]
        raise AssertionError("the budget should have stopped this")

    src._category = never_called  # type: ignore[method-assign]
    try:
        with pytest.raises(SourceUnavailable):
            await src.fetch(NOW - timedelta(days=5), NOW, profile)
    finally:
        await src.aclose()


async def test_the_cache_makes_a_second_fetch_free(tmp_path, profile) -> None:
    """The ToU asks for this explicitly, and it is what stops re-runs becoming a ban."""
    from tests.factories import make_paper

    src = _source(tmp_path)
    requests = {"n": 0}

    async def fake_category(category: str, since: datetime, until: datetime):  # type: ignore[no-untyped-def]
        requests["n"] += 1
        return [make_paper(arxiv_id=f"2509.{requests['n']:05d}", categories=[category])]

    src._category = fake_category  # type: ignore[method-assign]
    try:
        first = await src.fetch(NOW - timedelta(days=5), NOW, profile)
        made = requests["n"]
        second = await src.fetch(NOW - timedelta(days=5), NOW, profile)
    finally:
        await src.aclose()
    assert made > 0
    assert len(first) == len(second)
    assert requests["n"] == made, "the second fetch must make no requests at all"
    assert src.cache_hits > 0


async def test_a_stale_cache_entry_is_not_used(tmp_path, profile) -> None:
    """`updated` only changes at midnight, so a cache is good for the day it was written."""
    import os

    from tests.factories import make_paper

    src = _source(tmp_path)
    requests = {"n": 0}

    async def fake_category(category: str, since: datetime, until: datetime):  # type: ignore[no-untyped-def]
        requests["n"] += 1
        return [make_paper(arxiv_id="2509.00001", categories=[category])]

    src._category = fake_category  # type: ignore[method-assign]
    try:
        await src.fetch(NOW - timedelta(days=5), NOW, profile)
        after_first = requests["n"]
        cached = list((tmp_path / "arxiv").glob("*.json"))
        assert cached
        old = (datetime.now(UTC) - timedelta(days=2)).timestamp()
        for path in cached:
            os.utime(path, (old, old))

        await src.fetch(NOW - timedelta(days=5), NOW, profile)
    finally:
        await src.aclose()

    # The stale entry must have been ignored rather than served. Asserting on `_cache_read`
    # afterwards would be wrong: the second fetch legitimately rewrites the cache.
    assert requests["n"] > after_first, "a stale cache entry must not be served"
    assert src.cache_hits == 0
