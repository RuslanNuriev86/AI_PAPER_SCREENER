"""Enrichment provenance and persistence (§6.1).

Two bugs motivate these tests, both of which looked healthy in the log while storing nothing:

* `_citations`/`_repos` updated the readings but never recorded *which source* produced them, so
  every enrichment carried an empty `sources_ok`. The run filtered on exactly that field and
  persisted zero rows while `enrich.done` cheerfully reported 7 papers with citations.
* `save_enrichment` hardcoded `version=1`, but `enrichment` is keyed including the version, so a
  revised paper's signals were filed against its first version.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from screener.adapters.enrich import Enricher
from screener.adapters.sqlite_repo import SqliteRepository
from screener.domain.models import Enrichment, Paper

NOW = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)


def _paper(version: int = 1, code_url: str | None = None) -> Paper:
    return Paper(
        arxiv_id="2609.11111",
        version=version,
        title="T",
        abstract="A",
        authors=["x"],
        categories=["cs.AI"],
        primary_category="cs.AI",
        submitted_at=NOW - timedelta(days=14),
        updated_at=NOW - timedelta(days=14),
        abs_url="u",
        pdf_url="u",
        code_url=code_url,
        first_seen_at=NOW,
    )


def _enricher(handler: object) -> Enricher:
    e = Enricher("test@example.com")
    e._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))  # type: ignore[arg-type]
    return e


def _openalex(citations: int = 3, source: str | None = None) -> dict:
    return {
        "results": [
            {
                "doi": "https://doi.org/10.48550/arxiv.2609.11111",
                "cited_by_count": citations,
                "primary_location": {"source": {"display_name": source} if source else None},
            }
        ]
    }


async def test_each_paper_records_the_source_that_produced_its_signals() -> None:
    """The regression that made persistence silently store nothing."""
    e = _enricher(lambda request: httpx.Response(200, json=_openalex()))
    try:
        out = await e.enrich([_paper()])
    finally:
        await e._http.aclose()
    assert out["2609.11111"].sources_ok == ["openalex"], "provenance must be per paper"
    assert out["2609.11111"].citations == 3


async def test_the_arxiv_preprint_record_is_not_reported_as_a_venue() -> None:
    """OpenAlex reports the preprint itself as a source; calling that a venue fabricates one."""
    e = _enricher(
        lambda request: httpx.Response(200, json=_openalex(source="arXiv (Cornell University)"))
    )
    try:
        out = await e.enrich([_paper()])
    finally:
        await e._http.aclose()
    assert out["2609.11111"].venue is None


async def test_a_real_venue_is_kept() -> None:
    e = _enricher(lambda request: httpx.Response(200, json=_openalex(source="NeurIPS 2026")))
    try:
        out = await e.enrich([_paper()])
    finally:
        await e._http.aclose()
    assert out["2609.11111"].venue == "NeurIPS 2026"


async def test_a_repo_records_its_age_relative_to_the_paper() -> None:
    """Without this the 34,432-star pre-existing project would look like the paper's traction."""
    created = (NOW - timedelta(days=300)).strftime("%Y-%m-%dT%H:%M:%SZ")

    def handler(request: httpx.Request) -> httpx.Response:
        if "api.github.com" in str(request.url):
            return httpx.Response(200, json={"stargazers_count": 34432, "created_at": created})
        return httpx.Response(200, json=_openalex())

    paper = _paper(code_url="https://github.com/huge/project")
    e = _enricher(handler)
    try:
        out = await e.enrich([paper])
    finally:
        await e._http.aclose()
    got = out["2609.11111"]
    assert got.stars == 34432
    # The paper is 14 days old and the repo 300, so the repo predates it by 286 days.
    assert got.repo_created_days_before_paper == pytest.approx(286, abs=2)
    assert "github" in got.sources_ok


async def test_a_dead_github_repo_is_no_measurement_not_zero_stars() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if "api.github.com" in str(request.url):
            return httpx.Response(404, json={})
        return httpx.Response(200, json=_openalex())

    e = _enricher(handler)
    try:
        out = await e.enrich([_paper(code_url="https://github.com/gone/repo")])
    finally:
        await e._http.aclose()
    assert out["2609.11111"].stars is None, "a missing repo is absent, not zero stars"
    assert "github" not in out["2609.11111"].sources_ok


def test_signals_are_stored_against_the_papers_own_version(repo: SqliteRepository) -> None:
    """`enrichment` is keyed including the version; defaulting it misfiled revised papers."""
    # `enrichment` has a foreign key to `papers(arxiv_id, version)`, so v3 must exist first.
    repo.save_papers([_paper(version=3)])
    repo.save_enrichment(
        [Enrichment(arxiv_id="2609.11111", version=3, citations=4, sources_ok=["openalex"])],
        NOW,
    )
    rows = repo._conn.execute("SELECT arxiv_id, version, payload FROM enrichment").fetchall()
    assert len(rows) == 1
    assert rows[0]["version"] == 3, "the paper's real version, not 1"
    assert json.loads(rows[0]["payload"])["citations"] == 4
