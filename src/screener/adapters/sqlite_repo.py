"""SQLite repository (§10). stdlib `sqlite3`, plain .sql migrations, no ORM.

The two invariants this adapter is responsible for:

* **the `runs` row exists before anything references it.** `begin_run` inserts it with
  `status='running'`; every later write can carry a non-null `run_id` without an ordering
  hazard. The earlier design called `record_delivery` before ever calling `record_run`,
  which would have violated the FK on the first real send.
* **the seen-set is `gate_results`, keyed `(arxiv_id, version)`.** `seen()` returns what is
  already known rather than what is new, so a caller that inverts it wrongly looks empty
  instead of looking fresh — the safe direction.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from importlib import resources
from pathlib import Path

from screener.domain.models import (
    Assessment,
    CalibrationReport,
    GateResult,
    LedgerEntry,
    Outcome,
    Paper,
    Ranking,
    RevisitRun,
    RunStats,
    WatchlistEntry,
)
from screener.domain.types import (
    DeliveryKind,
    PaperKey,
    RunId,
    Status,
)

SCHEMA_VERSION = 1


def _json(value: object) -> str:
    dumper = getattr(value, "model_dump_json", None)
    if callable(dumper):
        return str(dumper())
    return json.dumps(value, default=str)


def _loads(raw: str | None) -> object:
    return json.loads(raw) if raw else None


def _load_str_list(raw: str | None) -> list[str]:
    """JSON array column -> list[str]. Typed so callers do not have to guess at `object`."""
    loaded = _loads(raw)
    if not isinstance(loaded, list):
        return []
    return [str(item) for item in loaded]


class SqliteRepository:
    """Implements `ports.Repository`."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._conn = sqlite3.connect(self.path, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._ledger_seq: dict[str, int] = {}

    # -- lifecycle ---------------------------------------------------------------------

    def migrate(self) -> None:
        sql = resources.files("screener.migrations").joinpath("001_init.sql").read_text()
        self._conn.executescript(sql)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> SqliteRepository:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def begin_run(self, now: datetime, config_hash: str, mode: str) -> RunId:
        import uuid

        run_id = RunId(str(uuid.uuid4()))
        self._conn.execute(
            "INSERT INTO runs (run_id, started_at, status, mode, config_hash, stats)"
            " VALUES (?,?,?,?,?,?)",
            (run_id, now.isoformat(), "running", mode, config_hash, "{}"),
        )
        return run_id

    def finish_run(self, run_id: RunId, status: Status, stats: RunStats, cost_usd: float) -> None:
        self._conn.execute(
            "UPDATE runs SET finished_at=?, status=?, stats=?, cost_usd=? WHERE run_id=?",
            (datetime.now().astimezone().isoformat(), status, _json(stats), cost_usd, run_id),
        )

    def begin_revisit(self, now: datetime) -> str:
        import uuid

        revisit_id = str(uuid.uuid4())
        self._conn.execute(
            "INSERT INTO revisits (revisit_id, run_date, due, measured, missed) VALUES (?,?,0,0,0)",
            (revisit_id, now.date().isoformat()),
        )
        return revisit_id

    def finish_revisit(self, run: RevisitRun) -> None:
        self._conn.execute(
            "UPDATE revisits SET due=?, measured=?, missed=?, per_source_errors=?,"
            " cost_usd=?, stats=? WHERE revisit_id=?",
            (
                run.due,
                run.measured,
                run.missed,
                _json(run.per_source_errors),
                run.cost_usd,
                _json(run.stats),
                run.revisit_id,
            ),
        )

    # -- delivery path -----------------------------------------------------------------

    def seen(self, keys: Sequence[PaperKey]) -> set[PaperKey]:
        if not keys:
            return set()
        out: set[PaperKey] = set()
        # Chunked: SQLite's default parameter limit is 999.
        for start in range(0, len(keys), 400):
            chunk = keys[start : start + 400]
            placeholders = ",".join("(?,?)" for _ in chunk)
            params: list[object] = []
            for arxiv_id, version in chunk:
                params += [arxiv_id, version]
            rows = self._conn.execute(
                f"SELECT arxiv_id, version FROM gate_results WHERE (arxiv_id, version)"
                f" IN (VALUES {placeholders})",
                params,
            ).fetchall()
            out |= {(str(r["arxiv_id"]), int(r["version"])) for r in rows}
        return out

    def save_papers(self, papers: Sequence[Paper]) -> None:
        for p in papers:
            self._conn.execute(
                "INSERT INTO papers (arxiv_id, version, title, abstract, authors, categories,"
                " primary_category, submitted_at, updated_at, abs_url, pdf_url, comment,"
                " code_url, first_seen_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
                " ON CONFLICT(arxiv_id, version) DO UPDATE SET"
                "   title=excluded.title, abstract=excluded.abstract,"
                "   categories=excluded.categories, comment=excluded.comment,"
                "   code_url=COALESCE(excluded.code_url, papers.code_url),"
                "   updated_at=COALESCE(papers.updated_at, excluded.updated_at)",
                (
                    p.arxiv_id,
                    p.version,
                    p.title,
                    p.abstract,
                    _json(p.authors),
                    _json(p.categories),
                    p.primary_category,
                    p.submitted_at.isoformat(),
                    p.updated_at.isoformat() if p.updated_at else None,
                    p.abs_url,
                    p.pdf_url,
                    p.comment,
                    p.code_url,
                    (p.first_seen_at or p.submitted_at).isoformat(),
                ),
            )

    def save_gate_results(self, run_id: RunId, results: Sequence[GateResult]) -> None:
        for r in results:
            self._conn.execute(
                "INSERT INTO gate_results (arxiv_id, version, run_id, keep, reason, hint,"
                " created_at) VALUES (?,?,?,?,?,?,?)"
                " ON CONFLICT(arxiv_id, version) DO NOTHING",
                (
                    r.paper.arxiv_id,
                    r.paper.version,
                    run_id,
                    1 if r.keep else 0,
                    r.reason.value if r.reason else None,
                    _json(r.hint),
                    datetime.now().astimezone().isoformat(),
                ),
            )

    def delivered_versions(self, arxiv_ids: Sequence[str]) -> Mapping[str, set[int]]:
        out: dict[str, set[int]] = {}
        if not arxiv_ids:
            return out
        for start in range(0, len(arxiv_ids), 400):
            chunk = arxiv_ids[start : start + 400]
            placeholders = ",".join("?" for _ in chunk)
            rows = self._conn.execute(
                f"SELECT arxiv_id, version FROM deliveries WHERE message_id IS NOT NULL"
                f" AND arxiv_id IN ({placeholders})",
                list(chunk),
            ).fetchall()
            for r in rows:
                out.setdefault(str(r["arxiv_id"]), set()).add(int(r["version"]))
        return out

    def latest_version(self, arxiv_id: str) -> int:
        row = self._conn.execute(
            "SELECT MAX(version) AS v FROM papers WHERE arxiv_id=?", (arxiv_id,)
        ).fetchone()
        return int(row["v"]) if row and row["v"] is not None else 0

    def save_assessment(self, run_id: RunId, a: Assessment) -> None:
        payload = a.triage or a.scores or a.verdict
        flags = {
            "soft": [f.value for f in a.soft_flags],
            "hard": a.hard_flag.value if a.hard_flag else None,
        }
        self._conn.execute(
            "INSERT INTO assessments (arxiv_id, version, run_id, stage, prompt_version, model,"
            " created_at, payload, review, flags, cost_usd) VALUES (?,?,?,?,?,?,?,?,?,?,?)"
            " ON CONFLICT DO NOTHING",
            (
                a.arxiv_id,
                a.version,
                run_id,
                a.stage,
                a.prompt_version,
                a.model,
                a.created_at.isoformat(),
                _json(payload),
                _json(a.review) if a.review else None,
                _json(flags),
                a.cost_usd,
            ),
        )

    def save_rankings(self, run_id: RunId, ranked: Sequence[Ranking]) -> None:
        now = datetime.now().astimezone().isoformat()
        for r in ranked:
            self._conn.execute(
                "INSERT INTO rankings (run_id, arxiv_id, version, score, components,"
                " effective_weights, soft_flags, disposition, hard_flag, topics, lab, tags,"
                " created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)"
                " ON CONFLICT(run_id, arxiv_id, version) DO NOTHING",
                (
                    run_id,
                    r.arxiv_id,
                    r.version,
                    r.score,
                    _json(r.components),
                    _json(r.effective_weights),
                    _json([f.value for f in r.soft_flags]),
                    r.disposition,
                    r.hard_flag.value if r.hard_flag else None,
                    _json(list(r.topics)),
                    r.lab,
                    _json(r.tags),
                    now,
                ),
            )

    def record_delivery(
        self,
        run_id: RunId,
        picks: Sequence[Ranking],
        kind: DeliveryKind,
        message_of: Mapping[str, str],
    ) -> None:
        now = datetime.now().astimezone().isoformat()
        for rank, r in enumerate(picks, start=1):
            self._conn.execute(
                "INSERT INTO deliveries (run_id, arxiv_id, version, kind, rank, score,"
                " message_id, sent_at) VALUES (?,?,?,?,?,?,?,?)"
                " ON CONFLICT(run_id, arxiv_id, version, kind) DO UPDATE SET"
                "   message_id=excluded.message_id, sent_at=excluded.sent_at",
                (
                    run_id,
                    r.arxiv_id,
                    r.version,
                    kind,
                    rank,
                    r.score,
                    message_of.get(r.arxiv_id),
                    now,
                ),
            )

    def ledger_add(self, run_id: RunId, entry: LedgerEntry) -> None:
        seq = self._ledger_seq.get(str(run_id), 0)
        self._ledger_seq[str(run_id)] = seq + 1
        self._conn.execute(
            "INSERT INTO cost_ledger (run_id, seq, created_at, stage, model, input_tokens,"
            " output_tokens, usd) VALUES (?,?,?,?,?,?,?,?)",
            (
                run_id,
                seq,
                datetime.now().astimezone().isoformat(),
                entry.stage,
                entry.model,
                entry.input_tokens,
                entry.output_tokens,
                entry.usd,
            ),
        )

    def run_cost(self, run_id: RunId) -> float:
        row = self._conn.execute(
            "SELECT COALESCE(SUM(usd), 0) AS c FROM cost_ledger WHERE run_id=?", (run_id,)
        ).fetchone()
        return float(row["c"]) if row else 0.0

    # -- maturity loop -----------------------------------------------------------------

    def enrol(self, entries: Sequence[WatchlistEntry]) -> None:
        for e in entries:
            self._conn.execute(
                "INSERT INTO watchlist (arxiv_id, enrolled_version, cohort_date, score_band,"
                " day0_score, day0_impact_forecast, delivered, enrolled_at)"
                " VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(arxiv_id) DO NOTHING",
                (
                    e.arxiv_id,
                    e.enrolled_version,
                    e.cohort_date.isoformat(),
                    e.score_band,
                    e.day0_score,
                    e.day0_impact_forecast,
                    1 if e.delivered else 0,
                    datetime.now().astimezone().isoformat(),
                ),
            )

    def watchlist(self) -> list[WatchlistEntry]:
        self._conn.row_factory = sqlite3.Row
        rows = self._conn.execute(
            "SELECT * FROM watchlist ORDER BY cohort_date, arxiv_id"
        ).fetchall()
        return [
            WatchlistEntry(
                arxiv_id=str(r["arxiv_id"]),
                enrolled_version=int(r["enrolled_version"]),
                cohort_date=datetime.fromisoformat(str(r["cohort_date"])).date()
                if "T" in str(r["cohort_date"])
                else datetime.strptime(str(r["cohort_date"]), "%Y-%m-%d").date(),
                score_band=str(r["score_band"]),  # type: ignore[arg-type]
                day0_score=r["day0_score"],
                day0_impact_forecast=r["day0_impact_forecast"],
                delivered=bool(r["delivered"]),
            )
            for r in rows
        ]

    def measured_rungs(self) -> set[tuple[str, int]]:
        rows = self._conn.execute("SELECT arxiv_id, rung_days FROM outcomes").fetchall()
        return {(str(r["arxiv_id"]), int(r["rung_days"])) for r in rows}

    def save_outcomes(self, outcomes: Sequence[Outcome]) -> None:
        for o in outcomes:
            self._conn.execute(
                "INSERT INTO outcomes (arxiv_id, rung_days, actual_age_days, status,"
                " measured_at, citations, influential_citations, stars, hf_upvotes, venue,"
                " social_mentions, revisions, code_url, matured_impact, components_present,"
                " raw) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
                " ON CONFLICT(arxiv_id, rung_days) DO UPDATE SET"
                "   status=excluded.status, matured_impact=excluded.matured_impact,"
                "   measured_at=excluded.measured_at",
                (
                    o.arxiv_id,
                    o.rung_days,
                    o.actual_age_days,
                    o.status,
                    o.measured_at.isoformat(),
                    o.signal.citations,
                    o.signal.influential_citations,
                    o.signal.stars,
                    o.signal.hf_upvotes,
                    o.signal.venue,
                    o.signal.social_mentions,
                    o.signal.revisions,
                    o.signal.code_url,
                    o.matured_impact,
                    _json(o.components_present),
                    _json(o.signal.raw),
                ),
            )

    def papers_by_key(self, keys: Sequence[PaperKey]) -> dict[PaperKey, Paper]:
        out: dict[PaperKey, Paper] = {}
        for start in range(0, len(keys), 400):
            chunk = keys[start : start + 400]
            placeholders = ",".join("(?,?)" for _ in chunk)
            params: list[object] = []
            for arxiv_id, version in chunk:
                params += [arxiv_id, version]
            rows = self._conn.execute(
                f"SELECT * FROM papers WHERE (arxiv_id, version) IN (VALUES {placeholders})",
                params,
            ).fetchall()
            for r in rows:
                p = _paper_from_row(r)
                out[p.key] = p
        return out

    def save_calibration(self, report: CalibrationReport) -> None:
        self._conn.execute(
            "INSERT INTO calibration (report_date, rung_days, n, spearman,"
            " spearman_within_topic, mae, false_negative_count, payload)"
            " VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(report_date, rung_days) DO UPDATE SET"
            "   n=excluded.n, spearman=excluded.spearman, mae=excluded.mae,"
            "   false_negative_count=excluded.false_negative_count, payload=excluded.payload",
            (
                report.report_date.isoformat(),
                report.rung_days,
                report.n,
                report.spearman,
                None,
                report.mae,
                report.false_negative_count,
                _json(report),
            ),
        )

    def get_state(self, key: str) -> str | None:
        row = self._conn.execute("SELECT value FROM kv_state WHERE key=?", (key,)).fetchone()
        return str(row["value"]) if row else None

    def set_state(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT INTO kv_state (key, value, updated_at) VALUES (?,?,?)"
            " ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
            (key, value, datetime.now().astimezone().isoformat()),
        )

    # -- feedback (§8.3) ---------------------------------------------------------------

    def delivery_by_message(self, message_id: str) -> tuple[str | None, str | None] | None:
        """message_id -> (run_id, arxiv_id).

        Resolves a replied-to message back to the paper it carried. This works only because
        `compose` records which item landed in which chunk (§10): `send` returns one id per
        chunk, so without that map the join is impossible.
        """
        row = self._conn.execute(
            "SELECT run_id, arxiv_id FROM deliveries WHERE message_id=? ORDER BY rank LIMIT 1",
            (message_id,),
        ).fetchone()
        if row is None:
            return None
        return str(row["run_id"]), str(row["arxiv_id"])

    def add_feedback(
        self,
        *,
        message_id: str,
        run_id: str,
        arxiv_id: str,
        kind: str,
        value: str,
        created_at: datetime,
    ) -> None:
        self._conn.execute(
            "INSERT INTO feedback (message_id, run_id, arxiv_id, kind, value, created_at)"
            " VALUES (?,?,?,?,?,?) ON CONFLICT DO NOTHING",
            (message_id, run_id, arxiv_id, kind, value, created_at.isoformat()),
        )

    # -- diagnostics -------------------------------------------------------------------

    def counts(self) -> dict[str, int]:
        """Row count per table, derived from the schema so it cannot go stale.

        Listing tables by hand meant this stayed at 8 entries after the maturity loop added
        four more, so `doctor` reported a schema that no longer existed.
        """
        tables = [
            str(r["name"])
            for r in self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
                " AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
        ]
        out: dict[str, int] = {}
        for table in tables:
            row = self._conn.execute(f"SELECT COUNT(*) AS c FROM {table}").fetchone()
            out[table] = int(row["c"]) if row else 0
        return out


def _paper_from_row(r: sqlite3.Row) -> Paper:
    return Paper(
        arxiv_id=str(r["arxiv_id"]),
        version=int(r["version"]),
        title=str(r["title"]),
        abstract=str(r["abstract"]),
        authors=_load_str_list(str(r["authors"])),
        categories=_load_str_list(str(r["categories"])),
        primary_category=str(r["primary_category"]),
        submitted_at=datetime.fromisoformat(str(r["submitted_at"])),
        updated_at=datetime.fromisoformat(str(r["updated_at"])) if r["updated_at"] else None,
        abs_url=str(r["abs_url"]),
        pdf_url=str(r["pdf_url"]),
        comment=str(r["comment"]) if r["comment"] else None,
        code_url=str(r["code_url"]) if r["code_url"] else None,
        first_seen_at=datetime.fromisoformat(str(r["first_seen_at"])),
    )


def iter_paper_keys(papers: Iterable[Paper]) -> list[PaperKey]:
    return [p.key for p in papers]
