"""The web UI (§17) and the reader-feedback storage it reads.

The point of these tests is that the UI is a *read-only* view of real stored data. A page that
renders 200 while silently showing nothing is the failure mode worth guarding against, so the
reaction views are exercised with rows actually present.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from screener.adapters.sqlite_repo import SqliteRepository
from screener.web import queries
from screener.web.app import create_app

NOW = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)


def _seed(path: Path) -> None:
    """A minimal but realistic database: one run, one delivered paper, two reactions."""
    repo = SqliteRepository(str(path))
    repo.migrate()
    conn = repo._conn  # the UI only reads; the test writes directly for brevity
    conn.execute(
        "INSERT INTO runs (run_id, started_at, finished_at, status, mode, config_hash, stats,"
        " cost_usd) VALUES (?,?,?,?,?,?,?,?)",
        (
            "run-1",
            NOW.isoformat(),
            NOW.isoformat(),
            "ok",
            "daily",
            "hash",
            json.dumps(
                {
                    "stage_counts": {
                        "fetched": 900,
                        "fresh": 120,
                        "gated_keep": 40,
                        "ranked": 16,
                        "review_failures": 0,
                        "picked": 2,
                        "enriched": 2,
                    },
                    "stage_seconds": {},
                    "gated_by_reason": {},
                    "cost_usd": 0.02,
                    "notes": [],
                }
            ),
            0.02,
        ),
    )
    conn.execute(
        "INSERT INTO papers (arxiv_id, version, title, abstract, authors, categories,"
        " primary_category, submitted_at, updated_at, abs_url, pdf_url, first_seen_at)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            "2609.11111",
            1,
            "A Very Promising Agent Paper",
            "An abstract about agents.",
            json.dumps(["A. Author"]),
            json.dumps(["cs.AI"]),
            "cs.AI",
            "2026-09-17T00:00:00+00:00",
            "2026-09-17T00:00:00+00:00",
            "https://arxiv.org/abs/2609.11111",
            "https://arxiv.org/pdf/2609.11111",
            NOW.isoformat(),
        ),
    )
    conn.execute(
        "INSERT INTO rankings (run_id, arxiv_id, version, score, components, effective_weights,"
        " soft_flags, disposition, topics, tags, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (
            "run-1",
            "2609.11111",
            1,
            7.25,
            json.dumps({"relevance": 1.2, "novelty": 1.5, "repo_signal": 0.5}),
            json.dumps({"relevance": 0.2, "novelty": 0.25, "repo_signal": 0.1}),
            json.dumps([]),
            "eligible",
            json.dumps(["safety & oversight"]),
            json.dumps([]),
            NOW.isoformat(),
        ),
    )
    conn.execute(
        "INSERT INTO assessments (arxiv_id, version, run_id, stage, prompt_version, model,"
        " created_at, payload, review, flags, cost_usd) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (
            "2609.11111",
            1,
            "run-1",
            "review",
            "v0",
            "deepseek-flash",
            NOW.isoformat(),
            json.dumps(
                {
                    "relevance": 6,
                    "novelty": 6,
                    "rigor": 6,
                    "evidence_strength": 6,
                    "reproducibility": 6,
                }
            ),
            json.dumps(
                {
                    "tldr": "It does a thing.",
                    "what_they_did": "w",
                    "why_it_matters": "y",
                    "caveats": "c",
                    "lenses": ["method"],
                    "tags": ["benchmark"],
                }
            ),
            json.dumps([]),
            0.001,
        ),
    )
    conn.execute(
        "INSERT INTO deliveries (run_id, arxiv_id, version, kind, rank, score, message_id,"
        " sent_at) VALUES (?,?,?,?,?,?,?,?)",
        ("run-1", "2609.11111", 1, "digest", 1, 7.25, "77", NOW.isoformat()),
    )
    # two different readers react with the same emoji: the v1 primary key could not hold both.
    for uid, name, value in ((11, "ada", "👍"), (22, "bob", "👍")):
        conn.execute(
            "INSERT INTO feedback (message_id, run_id, arxiv_id, kind, value, tg_user_id,"
            " tg_user_name, created_at) VALUES (?,?,?,?,?,?,?,?)",
            ("77", "run-1", "2609.11111", "reaction", value, uid, name, NOW.isoformat()),
        )
    conn.execute(
        "INSERT INTO feedback (message_id, run_id, arxiv_id, kind, value, tg_user_id,"
        " tg_user_name, created_at) VALUES (?,?,?,?,?,?,?,?)",
        ("77", "run-1", "2609.11111", "rating", "fire", 11, "ada", NOW.isoformat()),
    )
    conn.commit()
    repo.close()


@pytest.fixture()
def seeded(tmp_path: Path) -> Path:
    path = tmp_path / "web.db"
    _seed(path)
    return path


@pytest.fixture()
def client(seeded: Path) -> TestClient:
    return TestClient(create_app(seeded))


ROUTES = [
    "/",
    "/days",
    "/papers",
    "/reactions",
    "/top?by=score",
    "/top?by=users",
    "/day/2026-10-01",
    "/paper/2609.11111",
    "/healthz",
]


@pytest.mark.parametrize("route", ROUTES)
def test_every_route_renders(client: TestClient, route: str) -> None:
    response = client.get(route)
    assert response.status_code == 200, route


def test_the_database_is_never_written_to(seeded: Path) -> None:
    """Browsing must not be able to corrupt the store the pipeline writes to."""
    conn = queries.connect(seeded)
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("INSERT INTO feedback (message_id) VALUES ('x')")
    finally:
        conn.close()


def test_two_readers_can_react_with_the_same_emoji(seeded: Path) -> None:
    """The v1 key (message_id, arxiv_id, kind, value) collapsed these into one row."""
    conn = queries.connect(seeded)
    try:
        summary = queries.reactions_for(conn, "2609.11111")
    finally:
        conn.close()
    assert summary["emoji"] == {"👍": 2}
    assert summary["total"] == 2
    assert sorted(summary["people"]) == ["ada", "bob"]
    assert summary["score"] == 2


def test_the_reactions_page_lists_the_rows(client: TestClient) -> None:
    body = client.get("/reactions?start=2026-01-01&end=2026-12-31").text
    assert "ada" in body and "bob" in body
    assert "👍" in body
    assert "A Very Promising Agent Paper" in body


def test_top_by_users_ranks_on_reader_reactions(client: TestClient) -> None:
    body = client.get("/top?by=users&start=2026-01-01&end=2026-12-31").text
    assert "A Very Promising Agent Paper" in body
    assert "+2" in body, "two 👍 must weigh two, not one"


def test_top_by_score_ranks_on_our_score(client: TestClient) -> None:
    body = client.get("/top?by=score&start=2026-01-01&end=2026-12-31").text
    assert "A Very Promising Agent Paper" in body
    assert "7.25" in body


def test_the_day_view_shows_the_delivered_item(client: TestClient) -> None:
    body = client.get("/day/2026-10-01").text
    assert "A Very Promising Agent Paper" in body
    assert "It does a thing." in body
    assert "message" in body and "77" in body


def test_the_paper_page_shows_the_rating_arithmetic(client: TestClient) -> None:
    body = client.get("/paper/2609.11111").text
    assert "novelty" in body
    assert "0.250" in body, "the weight must be shown, not just the contribution"
    assert "Measured signals" in body


def test_a_bad_date_is_rejected_not_crashed(client: TestClient) -> None:
    assert client.get("/day/not-a-date").status_code == 400


def test_an_unknown_paper_is_404(client: TestClient) -> None:
    assert client.get("/paper/9999.99999").status_code == 404


def test_the_period_filter_actually_filters(client: TestClient) -> None:
    """A range that excludes the delivery must not show the paper."""
    body = client.get("/papers?start=2020-01-01&end=2020-01-31").text
    assert "A Very Promising Agent Paper" not in body
    body = client.get("/papers?start=2026-01-01&end=2026-12-31").text
    assert "A Very Promising Agent Paper" in body


def test_the_search_box_filters_on_text(client: TestClient) -> None:
    assert "A Very Promising" in client.get("/papers?q=promising").text
    assert "A Very Promising" not in client.get("/papers?q=zzzznotfound").text


def test_an_unmigrated_database_says_so_rather_than_failing_oddly(tmp_path: Path) -> None:
    """A user_version=0 database has no `feedback.tg_user_id`; the error must say what to do."""
    path = tmp_path / "old.db"
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE feedback (message_id TEXT)")
    conn.commit()
    conn.close()
    with pytest.raises(RuntimeError, match="schema version"):
        create_app(path)


def test_a_paper_reviewed_by_several_runs_is_not_duplicated_in_the_day_view(tmp_path: Path) -> None:
    """The day view must show the review the digest was built from, exactly once.

    Re-arming and re-running reviews the same paper again under a new `run_id`, and the schema
    permits that (the assessments key includes `run_id`). Joining assessments without a run
    filter therefore multiplied the item — in the live database one paper appeared three times.
    """
    path = tmp_path / "reassessed.db"
    _seed(path)
    conn = sqlite3.connect(str(path))
    conn.execute(
        "INSERT INTO runs (run_id, started_at, finished_at, status, mode, config_hash, stats,"
        " cost_usd) VALUES ('run-2', ?, ?, 'ok', 'daily', 'h2', '{}', 0.01)",
        (NOW.isoformat(), NOW.isoformat()),
    )
    conn.execute(
        "INSERT INTO rankings (run_id, arxiv_id, version, score, components, effective_weights,"
        " soft_flags, disposition, topics, tags, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (
            "run-2",
            "2609.11111",
            1,
            9.0,
            json.dumps({}),
            json.dumps({}),
            json.dumps([]),
            "eligible",
            json.dumps([]),
            json.dumps([]),
            NOW.isoformat(),
        ),
    )
    conn.execute(
        "INSERT INTO assessments (arxiv_id, version, run_id, stage, prompt_version, model,"
        " created_at, payload, review, flags, cost_usd) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (
            "2609.11111",
            1,
            "run-2",
            "review",
            "v0",
            "deepseek-flash",
            NOW.isoformat(),
            json.dumps(
                {
                    "relevance": 9,
                    "novelty": 9,
                    "rigor": 9,
                    "evidence_strength": 9,
                    "reproducibility": 9,
                }
            ),
            json.dumps(
                {
                    "tldr": "A LATER review that was never delivered.",
                    "what_they_did": "w",
                    "why_it_matters": "y",
                    "caveats": "c",
                    "lenses": [],
                    "tags": [],
                }
            ),
            json.dumps([]),
            0.002,
        ),
    )
    conn.commit()
    conn.close()

    body = TestClient(create_app(path)).get("/day/2026-10-01").text
    assert body.count('<article class="paper">') == 1, "one delivered paper, one article"
    assert body.count("A Very Promising Agent Paper") == 1
    assert "It does a thing." in body, "the delivered run's own review"
    assert "A LATER review that was never delivered." not in body


def test_top_by_score_also_uses_the_delivered_runs_review(tmp_path: Path) -> None:
    """The same run-scoping matters everywhere an assessment is joined for display."""
    path = tmp_path / "top.db"
    _seed(path)
    conn = sqlite3.connect(str(path))
    conn.execute(
        "INSERT INTO runs (run_id, started_at, finished_at, status, mode, config_hash, stats,"
        " cost_usd) VALUES ('run-2', ?, ?, 'ok', 'daily', 'h2', '{}', 0.01)",
        (NOW.isoformat(), NOW.isoformat()),
    )
    conn.execute(
        "INSERT INTO rankings (run_id, arxiv_id, version, score, components, effective_weights,"
        " soft_flags, disposition, topics, tags, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (
            "run-2",
            "2609.11111",
            1,
            9.0,
            json.dumps({}),
            json.dumps({}),
            json.dumps([]),
            "eligible",
            json.dumps([]),
            json.dumps([]),
            NOW.isoformat(),
        ),
    )
    conn.execute(
        "INSERT INTO assessments (arxiv_id, version, run_id, stage, prompt_version, model,"
        " created_at, payload, review, flags, cost_usd) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (
            "2609.11111",
            1,
            "run-2",
            "review",
            "v0",
            "deepseek-flash",
            NOW.isoformat(),
            json.dumps(
                {
                    "relevance": 9,
                    "novelty": 9,
                    "rigor": 9,
                    "evidence_strength": 9,
                    "reproducibility": 9,
                }
            ),
            json.dumps(
                {
                    "tldr": "A LATER review that was never delivered.",
                    "what_they_did": "w",
                    "why_it_matters": "y",
                    "caveats": "c",
                    "lenses": [],
                    "tags": [],
                }
            ),
            json.dumps([]),
            0.002,
        ),
    )
    conn.commit()
    conn.close()

    body = TestClient(create_app(path)).get("/top?by=score&start=2026-01-01&end=2026-12-31").text
    assert body.count("A Very Promising Agent Paper") == 1
    assert "It does a thing." in body
