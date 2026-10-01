"""Read-only queries for the web UI.

Opened with SQLite's `mode=ro` URI, so the UI **cannot** write to the database even by accident:
browsing must never be able to corrupt the store the pipeline depends on.

Two details that would otherwise produce quietly wrong pages:

* **Dates come from `substr(ts, 1, 10)`, not `date(ts)`.** Timestamps are stored with the local
  UTC offset (`2026-10-01T11:00:15+03:00`). SQLite's `date()` normalises to UTC, so a run at
  02:00 local would be grouped under the *previous* day. The string prefix is the local date as
  recorded, which is what a reader means by "that day".
* **Papers are joined on the newest version.** A paper can appear at several versions, and joining
  on `arxiv_id` alone multiplies every aggregate.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

#: A reaction the reader can leave, and what it is worth when ranking by readers.
#: Kept in one place so the digest, the poller and this UI cannot disagree about a 👎.
REACTION_WEIGHTS: dict[str, int] = {
    "👍": 1,
    "🔥": 2,
    "❤": 2,
    "🎉": 1,
    "up": 1,
    "fire": 2,
    "👎": -1,
    "down": -1,
    "💩": -2,
}

_NEWEST_VERSION = "(SELECT MAX(version) FROM papers WHERE arxiv_id = p.arxiv_id)"


def connect(db_path: str | Path) -> sqlite3.Connection:
    """Open the database read-only. Writes fail loudly rather than silently succeeding."""
    path = Path(db_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"no database at {path}")
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


#: The schema this UI's SQL depends on. `feedback.tg_user_id` arrives in migration 002, and
#: without a guard an unmigrated database fails as "no such column: tg_user_id" from deep inside
#: a template — which reads like a UI bug rather than a database that needs migrating.
REQUIRED_SCHEMA_VERSION = 2


def require_schema(conn: sqlite3.Connection) -> None:
    """Fail loudly and actionably if the database predates the migrations this UI needs."""
    version = int(conn.execute("PRAGMA user_version").fetchone()[0])
    if version < REQUIRED_SCHEMA_VERSION:
        raise RuntimeError(
            f"screener.db is at schema version {version}, but the web UI needs "
            f"{REQUIRED_SCHEMA_VERSION}. Run any pipeline command first (for example "
            f"`screener stats`) to apply migrations."
        )


def default_range() -> tuple[str, str]:
    """The last 30 days, as `YYYY-MM-DD` strings."""
    today = date.today()
    return (today - timedelta(days=29)).isoformat(), today.isoformat()


def _rows(cur: sqlite3.Cursor) -> list[dict[str, Any]]:
    return [dict(r) for r in cur.fetchall()]


def _review_of(raw: object) -> dict[str, Any]:
    """The stored `Review` JSON, or an empty dict — a page must not 500 on a missing review."""
    if not raw:
        return {}
    try:
        parsed = json.loads(str(raw))
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _json_of(raw: object) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(str(raw))
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


@dataclass
class Overview:
    """Everything the dashboard's headline numbers need."""

    papers: int = 0
    gated: int = 0
    delivered: int = 0
    runs: int = 0
    cost_usd: float = 0.0
    reactions: int = 0
    reaction_users: int = 0
    replies: int = 0
    ratings_by_user: dict[str, dict[str, int]] = field(default_factory=dict)
    emoji_totals: dict[str, int] = field(default_factory=dict)
    status_counts: dict[str, int] = field(default_factory=dict)


def overview(conn: sqlite3.Connection) -> Overview:
    o = Overview()
    o.papers = int(conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0])
    o.gated = int(conn.execute("SELECT COUNT(*) FROM gate_results").fetchone()[0])
    o.delivered = int(
        conn.execute("SELECT COUNT(*) FROM deliveries WHERE message_id IS NOT NULL").fetchone()[0]
    )
    o.runs = int(conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0])
    cost = conn.execute("SELECT COALESCE(SUM(cost_usd), 0) FROM runs").fetchone()[0]
    o.cost_usd = float(cost or 0.0)

    for row in conn.execute("SELECT status, COUNT(*) c FROM runs GROUP BY status"):
        o.status_counts[str(row["status"])] = int(row["c"])

    for row in conn.execute("SELECT kind, COUNT(*) c FROM feedback GROUP BY kind"):
        if str(row["kind"]) == "reaction":
            o.reactions = int(row["c"])
        else:
            o.replies += int(row["c"])

    o.reaction_users = int(
        conn.execute(
            "SELECT COUNT(DISTINCT tg_user_id) FROM feedback"
            " WHERE kind='reaction' AND tg_user_id > 0"
        ).fetchone()[0]
    )
    for row in conn.execute(
        "SELECT value, COUNT(*) c FROM feedback WHERE kind='reaction' GROUP BY value"
        " ORDER BY c DESC"
    ):
        o.emoji_totals[str(row["value"])] = int(row["c"])

    for row in conn.execute(
        "SELECT COALESCE(tg_user_name, 'user ' || tg_user_id) name,"
        " SUM(CASE WHEN kind='reaction' THEN 1 ELSE 0 END) reactions,"
        " SUM(CASE WHEN kind<>'reaction' THEN 1 ELSE 0 END) replies,"
        " SUM(CASE WHEN value IN ('👍','🔥','❤','up','fire') THEN 1"
        "          WHEN value IN ('👎','down') THEN -1 ELSE 0 END) score"
        " FROM feedback GROUP BY tg_user_id, name HAVING reactions + replies > 0"
        " ORDER BY score DESC, reactions DESC"
    ):
        o.ratings_by_user[str(row["name"])] = {
            "reactions": int(row["reactions"]),
            "replies": int(row["replies"]),
            "score": int(row["score"] or 0),
        }
    return o


def funnel(conn: sqlite3.Connection, limit: int = 30) -> list[dict[str, Any]]:
    """Per-run funnel, read from the stored `RunStats`, newest first."""
    out: list[dict[str, Any]] = []
    for row in conn.execute(
        "SELECT run_id, started_at, finished_at, status, mode, stats, cost_usd FROM runs"
        " ORDER BY started_at DESC LIMIT ?",
        (limit,),
    ):
        stats = _json_of(row["stats"])
        counts = stats.get("stage_counts") or {}
        out.append(
            {
                "run_id": str(row["run_id"])[:8],
                "day": str(row["started_at"])[:10],
                "started_at": row["started_at"],
                "status": row["status"],
                "mode": row["mode"],
                "cost_usd": float(row["cost_usd"] or 0),
                "fetched": counts.get("fetched", 0),
                "fresh": counts.get("fresh", 0),
                "gated": counts.get("gated_keep", counts.get("kept", 0)),
                "ranked": counts.get("ranked", 0),
                "review_failures": counts.get("review_failures", 0),
                "picked": counts.get("picked", 0),
                "enriched": counts.get("enriched", 0),
                "notes": stats.get("notes") or [],
            }
        )
    return out


def days(conn: sqlite3.Connection, limit: int = 60) -> list[dict[str, Any]]:
    """One row per day that produced a digest, with what was sent and what it cost."""
    return _rows(
        conn.execute(
            """
            SELECT substr(d.sent_at, 1, 10)                AS day,
                   COUNT(*)                                AS papers,
                   COUNT(DISTINCT d.message_id)            AS messages,
                   ROUND(AVG(d.score), 2)                  AS mean_score,
                   ROUND(MAX(d.score), 2)                  AS best_score,
                   (SELECT COALESCE(SUM(r.cost_usd), 0) FROM runs r
                     WHERE substr(r.started_at, 1, 10) = substr(d.sent_at, 1, 10)) AS cost_usd,
                   (SELECT COUNT(*) FROM feedback f
                     WHERE f.arxiv_id IN (SELECT arxiv_id FROM deliveries d2
                                           WHERE substr(d2.sent_at,1,10) = substr(d.sent_at,1,10))
                   )                                       AS reactions
            FROM deliveries d
            WHERE d.message_id IS NOT NULL
            GROUP BY day
            ORDER BY day DESC
            LIMIT ?
            """,
            (limit,),
        )
    )


def day_detail(conn: sqlite3.Connection, day: str) -> dict[str, Any]:
    """Everything a single day produced: what was sent, and what the runs did."""
    runs = funnel_for_day(conn, day)
    items = _rows(
        conn.execute(
            f"""
            SELECT d.arxiv_id, d.version, d.rank, d.score, d.message_id, d.sent_at,
                   p.title, p.abstract, p.authors, p.categories, p.abs_url, p.pdf_url,
                   p.comment, p.code_url, p.submitted_at,
                   k.disposition, k.soft_flags, k.topics, k.components, k.effective_weights,
                   a.review, a.model, a.payload
            FROM deliveries d
            JOIN papers p ON p.arxiv_id = d.arxiv_id AND p.version = {_NEWEST_VERSION}
            LEFT JOIN rankings k ON k.arxiv_id = d.arxiv_id AND k.version = d.version
                 AND k.run_id = d.run_id
            LEFT JOIN assessments a ON a.arxiv_id = d.arxiv_id AND a.version = d.version
                 AND a.run_id = d.run_id AND a.stage = 'review'
            WHERE d.message_id IS NOT NULL AND substr(d.sent_at, 1, 10) = ?
            ORDER BY d.rank
            """,
            (day,),
        )
    )
    items = _dedupe_papers(items)
    for item in items:
        item["review"] = _review_of(item.get("review"))
        item["soft_flags"] = _list_of(item.get("soft_flags"))
        item["topics"] = _list_of(item.get("topics"))
        item["components"] = _json_of(item.get("components"))
        item["effective_weights"] = _json_of(item.get("effective_weights"))
        # The judged dimension scores live in the assessment payload as JSON; the template needs
        # a mapping, not the raw string it was stored as.
        item["payload"] = _json_of(item.get("payload"))
        item["reactions"] = reactions_for(conn, str(item["arxiv_id"]))
    return {"day": day, "runs": runs, "items": items}


def _dedupe_papers(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per paper, keeping its most recent delivery.

    Re-arming a day and running again sends the same paper in a second digest, which is correct
    historically but reads as a duplicate in a "what went out that day" view. The newest delivery
    wins, so the page shows one row per paper; `deliveries` keeps the full history for anyone who
    needs it.
    """
    newest: dict[tuple[str, int], dict[str, Any]] = {}
    for item in items:
        key = (str(item["arxiv_id"]), int(item.get("version") or 1))
        seen = newest.get(key)
        if seen is None or str(item.get("sent_at") or "") > str(seen.get("sent_at") or ""):
            newest[key] = item
    return sorted(newest.values(), key=lambda i: (str(i.get("sent_at") or ""), i.get("rank") or 0))


def _list_of(raw: object) -> list[Any]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    try:
        parsed = json.loads(str(raw))
    except ValueError:
        return []
    return parsed if isinstance(parsed, list) else []


def funnel_for_day(conn: sqlite3.Connection, day: str) -> list[dict[str, Any]]:
    return [r for r in funnel(conn, limit=200) if r["day"] == day]


def reactions_for(conn: sqlite3.Connection, arxiv_id: str) -> dict[str, Any]:
    """Per-paper reaction summary: the raw emoji, the net score, and who reacted."""
    emoji: dict[str, int] = {}
    people: list[str] = []
    for row in conn.execute(
        "SELECT value, COUNT(*) c FROM feedback WHERE arxiv_id=? AND kind='reaction'"
        " GROUP BY value ORDER BY c DESC",
        (arxiv_id,),
    ):
        emoji[str(row["value"])] = int(row["c"])
    for row in conn.execute(
        "SELECT DISTINCT COALESCE(tg_user_name, 'user ' || tg_user_id) name FROM feedback"
        " WHERE arxiv_id=? AND kind='reaction' ORDER BY name",
        (arxiv_id,),
    ):
        people.append(str(row["name"]))
    return {
        "emoji": emoji,
        "score": sum(REACTION_WEIGHTS.get(k, 0) * v for k, v in emoji.items()),
        "total": sum(emoji.values()),
        "people": people,
    }


def top_by_score(
    conn: sqlite3.Connection, start: str, end: str, limit: int = 25
) -> list[dict[str, Any]]:
    """Papers *we* rated highest in a period — the screener's own ranking."""
    items = _rows(
        conn.execute(
            f"""
            SELECT d.arxiv_id, d.version, d.rank, d.score, substr(d.sent_at,1,10) AS day,
                   d.message_id, p.title, p.categories, p.abs_url, p.submitted_at,
                   k.soft_flags, k.topics, a.review
            FROM deliveries d
            JOIN papers p ON p.arxiv_id = d.arxiv_id AND p.version = {_NEWEST_VERSION}
            LEFT JOIN rankings k ON k.arxiv_id = d.arxiv_id AND k.version = d.version
                 AND k.run_id = d.run_id
            LEFT JOIN assessments a ON a.arxiv_id = d.arxiv_id AND a.version = d.version
                 AND a.run_id = d.run_id AND a.stage = 'review'
            WHERE d.message_id IS NOT NULL
              AND substr(d.sent_at, 1, 10) BETWEEN ? AND ?
            ORDER BY d.score DESC
            LIMIT ?
            """,
            (start, end, limit),
        )
    )
    items = _dedupe_papers(items)
    for item in items:
        item["review"] = _review_of(item.get("review"))
        item["soft_flags"] = _list_of(item.get("soft_flags"))
        item["reactions"] = reactions_for(conn, str(item["arxiv_id"]))
    return items


def top_by_users(
    conn: sqlite3.Connection, start: str, end: str, limit: int = 25
) -> list[dict[str, Any]]:
    """Papers *readers* rated highest in a period.

    Weighted by `REACTION_WEIGHTS` rather than counted, so one 🔥 is not the same as one 👍, and
    a 👎 subtracts. `users` counts distinct people, because three reactions from one enthusiastic
    reader is not the same signal as three readers agreeing.
    """
    # `CASE value WHEN ? THEN ?` — the subject matters. Without it SQLite evaluates the emoji
    # itself as a boolean, every branch is false, and every reader score silently becomes 0.
    case = " ".join("WHEN ? THEN ?" for _ in REACTION_WEIGHTS)
    case_params: list[Any] = [
        value for emoji, weight in REACTION_WEIGHTS.items() for value in (emoji, weight)
    ]
    items = _rows(
        conn.execute(
            f"""
            SELECT f.arxiv_id,
                   MAX(p.title)                                  AS title,
                   MAX(p.abs_url)                                AS abs_url,
                   MAX(p.categories)                             AS categories,
                   COUNT(*)                                      AS reactions,
                   COUNT(DISTINCT f.tg_user_id)                  AS users,
                   SUM(CASE value {case} ELSE 0 END)             AS user_score,
                   GROUP_CONCAT(DISTINCT f.value)                AS emoji,
                   MIN(substr(f.created_at, 1, 10))              AS first_seen,
                   MAX(substr(f.created_at, 1, 10))              AS last_seen
            FROM feedback f
            JOIN papers p ON p.arxiv_id = f.arxiv_id AND p.version = {_NEWEST_VERSION}
            WHERE f.kind = 'reaction' AND substr(f.created_at, 1, 10) BETWEEN ? AND ?
            GROUP BY f.arxiv_id
            ORDER BY user_score DESC, users DESC, reactions DESC
            LIMIT ?
            """,
            (*case_params, start, end, limit),
        )
    )
    for item in items:
        item["emoji"] = sorted(str(item.get("emoji") or "").split(","))
        item["delivered"] = conn.execute(
            "SELECT COUNT(*) FROM deliveries WHERE arxiv_id=? AND message_id IS NOT NULL",
            (item["arxiv_id"],),
        ).fetchone()[0]
    return items


def reactions(
    conn: sqlite3.Connection, start: str, end: str, limit: int = 500
) -> list[dict[str, Any]]:
    """Every piece of reader feedback in a period, newest first."""
    return _rows(
        conn.execute(
            f"""
            SELECT f.message_id, f.arxiv_id, f.kind, f.value, f.tg_user_id, f.tg_user_name,
                   f.created_at, substr(f.created_at,1,10) AS day,
                   p.title, p.abs_url,
                   (SELECT substr(d.sent_at,1,10) FROM deliveries d
                     WHERE d.message_id = f.message_id LIMIT 1) AS delivered_on
            FROM feedback f
            JOIN papers p ON p.arxiv_id = f.arxiv_id AND p.version = {_NEWEST_VERSION}
            WHERE substr(f.created_at, 1, 10) BETWEEN ? AND ?
            ORDER BY f.created_at DESC
            LIMIT ?
            """,
            (start, end, limit),
        )
    )


def papers(
    conn: sqlite3.Connection, start: str, end: str, query: str = "", limit: int = 100
) -> list[dict[str, Any]]:
    """Delivered and ranked papers in a period, optionally filtered by a text query."""
    like = f"%{query.strip()}%"
    where = (
        "AND (p.title LIKE ? OR p.abstract LIKE ? OR d.arxiv_id LIKE ?)" if query.strip() else ""
    )
    params: list[Any] = [start, end]
    if query.strip():
        params += [like, like, like]
    params.append(limit)
    items = _rows(
        conn.execute(
            f"""
            SELECT d.arxiv_id, d.version, d.rank, d.score, substr(d.sent_at,1,10) AS day,
                   d.message_id, p.title, p.categories, p.abs_url, p.submitted_at, p.code_url,
                   k.disposition, k.soft_flags, k.topics, a.review
            FROM deliveries d
            JOIN papers p ON p.arxiv_id = d.arxiv_id AND p.version = {_NEWEST_VERSION}
            LEFT JOIN rankings k ON k.arxiv_id = d.arxiv_id AND k.version = d.version
                 AND k.run_id = d.run_id
            LEFT JOIN assessments a ON a.arxiv_id = d.arxiv_id AND a.version = d.version
                 AND a.run_id = d.run_id AND a.stage = 'review'
            WHERE d.message_id IS NOT NULL
              AND substr(d.sent_at, 1, 10) BETWEEN ? AND ?
              {where}
            ORDER BY d.sent_at DESC, d.rank
            LIMIT ?
            """,
            params,
        )
    )
    items = _dedupe_papers(items)
    for item in items:
        item["review"] = _review_of(item.get("review"))
        item["soft_flags"] = _list_of(item.get("soft_flags"))
        item["topics"] = _list_of(item.get("topics"))
    return items


def paper_detail(conn: sqlite3.Connection, arxiv_id: str) -> dict[str, Any] | None:
    """One paper's full history: rating arithmetic, measured signals, delivery and reactions."""
    row = conn.execute(
        f"SELECT * FROM papers p WHERE p.arxiv_id = ? AND p.version = {_NEWEST_VERSION}",
        (arxiv_id,),
    ).fetchone()
    if row is None:
        return None
    paper = dict(row)

    ranking = conn.execute(
        "SELECT * FROM rankings WHERE arxiv_id=? ORDER BY created_at DESC LIMIT 1", (arxiv_id,)
    ).fetchone()
    assessment = conn.execute(
        "SELECT * FROM assessments WHERE arxiv_id=? AND stage='review'"
        " ORDER BY created_at DESC LIMIT 1",
        (arxiv_id,),
    ).fetchone()
    enrichment = conn.execute(
        # Scoped to the version being displayed: a revision can have its own signals.
        "SELECT * FROM enrichment WHERE arxiv_id=? AND version=? ORDER BY fetched_at DESC LIMIT 1",
        (arxiv_id, int(paper["version"])),
    ).fetchone()
    delivery = conn.execute(
        "SELECT * FROM deliveries WHERE arxiv_id=? AND message_id IS NOT NULL"
        " ORDER BY sent_at DESC LIMIT 1",
        (arxiv_id,),
    ).fetchone()
    watch = conn.execute(
        "SELECT * FROM watchlist WHERE arxiv_id=? ORDER BY cohort_date DESC LIMIT 1", (arxiv_id,)
    ).fetchone()
    outcomes = _rows(
        conn.execute("SELECT * FROM outcomes WHERE arxiv_id=? ORDER BY rung_days", (arxiv_id,))
    )

    return {
        "paper": paper,
        "ranking": dict(ranking) if ranking else None,
        "assessment": dict(assessment) if assessment else None,
        "review": _review_of(assessment["review"]) if assessment else {},
        "scores": _json_of(assessment["payload"]) if assessment else {},
        "components": _json_of(ranking["components"]) if ranking else {},
        "weights": _json_of(ranking["effective_weights"]) if ranking else {},
        "soft_flags": _list_of(ranking["soft_flags"]) if ranking else [],
        "topics": _list_of(ranking["topics"]) if ranking else [],
        "signals": _json_of(enrichment["payload"]) if enrichment else {},
        "enrichment_sources": _json_of(enrichment["payload"]).get("sources_ok", [])
        if enrichment
        else [],
        "delivery": dict(delivery) if delivery else None,
        "watchlist": dict(watch) if watch else None,
        "outcomes": outcomes,
        "reactions": reactions_for(conn, arxiv_id),
    }
