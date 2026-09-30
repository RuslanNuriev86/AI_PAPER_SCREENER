"""`screener replay` and `screener backtest` (§12.2).

Both are offline: no network, no sends. `replay` rebuilds a digest from stored data under
current config; `backtest` re-scores stored rankings under candidate weights and reports
which papers would have been added or dropped. Together they are what makes a weight change
reviewable before it ships (§13.5).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from screener.config import Settings
from screener.domain.models import Ranking, Review, Scores, Selection
from screener.domain.scoring import observable_dimensions, select
from screener.domain.types import RUBRIC_WEIGHTS, RunId


@dataclass
class ReplayResult:
    date: str
    run_ids: list[str]
    ranked: list[Ranking]
    picks: list[Ranking]

    def render(self) -> str:
        lines = [f"replay {self.date}: {len(self.ranked)} ranked, {len(self.picks)} would ship"]
        if not self.ranked:
            return "\n".join([*lines, "  (no stored rankings for that date)"])
        for r in sorted(self.ranked, key=lambda x: -x.score):
            mark = "PICK" if r in self.picks else "    "
            lines.append(f"  {mark} {r.score:5.2f}  {r.arxiv_id}v{r.version}  {r.disposition}")
        return "\n".join(lines)


def _connect(cfg: Settings) -> sqlite3.Connection:
    conn = sqlite3.connect(cfg.screener_db)
    conn.row_factory = sqlite3.Row
    return conn


def _stored_rankings(conn: sqlite3.Connection, where: str, params: list[object]) -> list[Ranking]:
    """Rebuild Ranking objects from the `rankings` + `assessments` tables.

    Only what replay needs is reconstructed: score, components, topics, disposition and the
    Review prose that compose renders. The Assessment is rebuilt as a shell because replay
    never re-scores against a model.
    """
    from screener.domain.models import Assessment

    rows = conn.execute(
        f"SELECT * FROM rankings WHERE {where} ORDER BY score DESC",
        params,
    ).fetchall()
    out: list[Ranking] = []
    for row in rows:
        arxiv_id = str(row["arxiv_id"])
        version = int(row["version"])
        arow = conn.execute(
            "SELECT * FROM assessments WHERE arxiv_id=? AND version=? AND stage='review'"
            " ORDER BY created_at DESC LIMIT 1",
            (arxiv_id, version),
        ).fetchone()
        review = None
        if arow is not None and arow["review"]:
            review = Review.model_validate(json.loads(str(arow["review"])))
        if review is None:
            continue
        assessment = Assessment(
            arxiv_id=arxiv_id,
            version=version,
            run_id=RunId(str(row["run_id"])),
            stage="review",
            prompt_version=str(arow["prompt_version"]) if arow else "unknown",
            model=str(arow["model"]) if arow else "unknown",
            created_at=datetime.fromisoformat(str(arow["created_at"])) if arow else datetime.now(),
            scores=review.scores,
            review=review,
        )
        out.append(
            Ranking(
                arxiv_id=arxiv_id,
                version=version,
                assessment=assessment,
                review=review,
                score=float(row["score"]),
                components=json.loads(str(row["components"])),
                effective_weights=json.loads(str(row["effective_weights"])),
                topics=json.loads(str(row["topics"])),
                disposition=str(row["disposition"]),  # type: ignore[arg-type]
                lab=row["lab"],
                tags=json.loads(str(row["tags"])),
            )
        )
    return out


def replay_run(cfg: Settings, date: str) -> ReplayResult:
    conn = _connect(cfg)
    try:
        try:
            runs = conn.execute(
                "SELECT run_id FROM runs WHERE date(started_at)=? ORDER BY started_at", (date,)
            ).fetchall()
        except sqlite3.OperationalError:
            # A DB that has never been migrated is "nothing stored", not a crash: the CLI turns
            # an empty result into a readable "no stored rankings for that date".
            return ReplayResult(date=date, run_ids=[], ranked=[], picks=[])
        run_ids = [str(r["run_id"]) for r in runs]
        if not run_ids:
            return ReplayResult(date=date, run_ids=[], ranked=[], picks=[])
        placeholders = ",".join("?" for _ in run_ids)
        ranked = _stored_rankings(conn, f"run_id IN ({placeholders})", list(run_ids))
        picks = select(ranked, cfg.selection)
        return ReplayResult(date=date, run_ids=run_ids, ranked=ranked, picks=picks)
    finally:
        conn.close()


def backtest_weights(cfg: Settings, since: str, overrides: dict[str, float] | None) -> str:
    """Re-score stored reviews under candidate weights and diff the picks.

    This is the tool that makes §6.6.4's proposals reviewable: a weight change is only
    shippable once you can see which historical papers it would have added or dropped.
    """
    weights: dict[str, float] = {k: float(v) for k, v in RUBRIC_WEIGHTS.items()}
    if overrides:
        weights.update(overrides)

    conn = _connect(cfg)
    try:
        rows = conn.execute(
            "SELECT * FROM rankings WHERE run_id IN"
            " (SELECT run_id FROM runs WHERE date(started_at) >= ?)",
            (since,),
        ).fetchall()
        before: list[tuple[str, float]] = []
        after: list[tuple[str, float]] = []
        for row in rows:
            arxiv_id = str(row["arxiv_id"])
            current = float(row["score"])
            before.append((arxiv_id, current))
            arow = conn.execute(
                "SELECT payload FROM assessments WHERE arxiv_id=? AND version=?"
                " AND stage='review' ORDER BY created_at DESC LIMIT 1",
                (arxiv_id, int(row["version"])),
            ).fetchone()
            if arow is None or not arow["payload"]:
                after.append((arxiv_id, current))
                continue
            scores = Scores.model_validate(json.loads(str(arow["payload"])))
            observable = observable_dimensions(scores)
            base = {dim: weights[str(dim)] for dim in observable}
            total = sum(base.values()) or 1.0
            rescored = sum(observable[d] * (base[d] / total) for d in observable)
            after.append((arxiv_id, rescored))
    finally:
        conn.close()

    if not before:
        return f"backtest since {since}: no stored rankings in range"

    sel = cfg.selection
    picks_before = {a for a, s in before if s >= sel.min_score}
    picks_after = {a for a, s in after if s >= sel.min_score}
    added = sorted(picks_after - picks_before)
    dropped = sorted(picks_before - picks_after)

    lines = [
        f"backtest since {since}: {len(before)} items re-scored",
        f"  weight overrides: {overrides or 'none'}",
        f"  added   ({len(added)}): {', '.join(added) or '-'}",
        f"  dropped ({len(dropped)}): {', '.join(dropped) or '-'}",
    ]
    return "\n".join(lines)


def selection_from_file(path: Path) -> Selection:
    import yaml

    return Selection.model_validate(yaml.safe_load(path.read_text()) or {})


def recover_to_outbox(cfg: Settings, date: str, outbox: Path | None = None) -> Path | None:
    """Rebuild a digest from stored data and park it for delivery.

    This is the recovery path for a digest that was rendered but could not be sent and whose
    outbox artifact was then lost. By the time it is discovered the papers are usually already
    in the seen-set, so no future run will rebuild it — the *only* remaining source is the
    stored `rankings`/`assessments` for that date, which is exactly what this reads.

    Nothing is sent here: the digest goes to the outbox and the next `screener run` delivers it
    through the normal retry, so delivery stays in one place.
    """
    from screener.adapters.sqlite_repo import SqliteRepository
    from screener.domain.compose import compose
    from screener.pipeline.deliver import OUTBOX, _write_outbox

    conn = _connect(cfg)
    try:
        runs = conn.execute(
            "SELECT run_id, stats, cost_usd FROM runs WHERE date(started_at)=? ORDER BY started_at",
            (date,),
        ).fetchall()
        if not runs:
            return None
        run_ids = [str(r["run_id"]) for r in runs]
        placeholders = ",".join("?" for _ in run_ids)
        ranked = _stored_rankings(conn, f"run_id IN ({placeholders})", list(run_ids))
        if not ranked:
            return None

        stats = json.loads(str(runs[-1]["stats"]) or "{}")
        scanned = int((stats.get("stage_counts") or {}).get("fetched", 0))
        relevant = int((stats.get("stage_counts") or {}).get("gated_kept", 0))
        cost = sum(float(r["cost_usd"] or 0) for r in runs)

        picks = select(ranked, cfg.selection)
        repo = SqliteRepository(cfg.screener_db)
        try:
            papers = repo.papers_by_key([(r.arxiv_id, r.version) for r in picks])
        finally:
            repo.close()
    finally:
        conn.close()

    if not picks:
        return None

    digest = compose(
        picks,
        papers,
        datetime.now(),
        scanned=scanned,
        relevant=relevant,
        cost_usd=cost,
        gated_by_reason={},
        below_threshold=max(0, len(ranked) - len(picks)),
    )
    target = outbox or Path(cfg.screener_outbox or OUTBOX)
    return _write_outbox(digest, datetime.now(), target)


def rearm(cfg: Settings, date: str) -> list[str]:
    """Make a date's reviewed papers fresh again so the next run re-processes them.

    Recovery for the case that actually happened: a digest was reviewed and paid for, the send
    failed, and the outbox artifact was then lost. The papers are in the seen-set, so no future
    run rebuilds them, and there is nothing to replay because the *prose* was only ever in the
    delivered message. Clearing their `gate_results` rows makes the next run treat them as new.

    Only the papers that were actually shortlisted are cleared, not the whole window: the other
    few thousand papers stay seen, so this costs one re-review rather than a re-fetch of a day.

    **Undelivered delivery rows are cleared; successful sends are not.** `dedupe_revisions` would
    otherwise drop a re-armed paper as "already delivered", which would make this command a
    no-op in exactly the case it exists for — while clearing a *successful* send would risk
    breaking the at-most-once guarantee (§10). Recovery is for failed delivery, so that is the
    line.

    Returns the arxiv_ids that were re-armed.
    """
    conn = _connect(cfg)
    try:
        try:
            run_ids = [
                str(r["run_id"])
                for r in conn.execute(
                    "SELECT run_id FROM runs WHERE date(started_at)=?", (date,)
                ).fetchall()
            ]
        except sqlite3.OperationalError:
            return []
        if not run_ids:
            return []
        placeholders = ",".join("?" for _ in run_ids)
        rows = conn.execute(
            f"SELECT DISTINCT arxiv_id, version FROM rankings WHERE run_id IN ({placeholders})",
            run_ids,
        ).fetchall()
        keys = [(str(r["arxiv_id"]), int(r["version"])) for r in rows]
        for arxiv_id, version in keys:
            conn.execute(
                "DELETE FROM gate_results WHERE arxiv_id=? AND version=?", (arxiv_id, version)
            )
            conn.execute(
                "DELETE FROM deliveries WHERE arxiv_id=? AND version=? AND message_id IS NULL",
                (arxiv_id, version),
            )
        conn.commit()
        return [k[0] for k in keys]
    finally:
        conn.close()
