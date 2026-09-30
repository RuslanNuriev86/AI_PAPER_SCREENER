"""`screener eval` and `screener prune` (§12.2, §13.4, §10).

`eval` reports what the maturity loop actually has: rung coverage and the reliability of the
impact forecast once any rung has matured. It is deliberately honest about having nothing to
report — at v0 there are no matured rungs yet, and saying so is correct.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from screener.config import Settings


def _connect(cfg: Settings) -> sqlite3.Connection:
    conn = sqlite3.connect(cfg.screener_db)
    conn.row_factory = sqlite3.Row
    return conn


@dataclass
class Coverage:
    watchlist: int
    delivered: int
    gate_only: int
    measured: int
    by_rung: dict[int, int]

    def render(self, spearman: float | None) -> str:
        lines = [
            "maturity loop",
            f"  enrolled          {self.watchlist}",
            f"  delivered         {self.delivered}",
            f"  gate_only         {self.gate_only}  (recall check: never delivered)",
            f"  rungs measured    {self.measured}",
        ]
        for rung, count in sorted(self.by_rung.items()):
            lines.append(f"    T+{rung:<3}            {count}")
        if self.measured == 0:
            lines.append(
                "  no matured rungs yet — calibration is not computable. This is expected at "
                "v0: enrolment just started, and the T+14 rung needs 14 days of history."
            )
        elif spearman is not None:
            lines.append(f"  impact_forecast rho {spearman:+.3f}  (predicted vs matured)")
        return "\n".join(lines)


def run_eval(cfg: Settings) -> str:
    conn = _connect(cfg)
    try:
        watch = conn.execute("SELECT COUNT(*) c FROM watchlist").fetchone()
        delivered = conn.execute("SELECT COUNT(*) c FROM watchlist WHERE delivered=1").fetchone()
        gate_only = conn.execute(
            "SELECT COUNT(*) c FROM watchlist WHERE score_band='gate_only'"
        ).fetchone()
        rung_rows = conn.execute(
            "SELECT rung_days, COUNT(*) c FROM outcomes WHERE status='measured' GROUP BY rung_days"
        ).fetchall()
        measured = sum(int(r["c"]) for r in rung_rows)
        by_rung = {int(r["rung_days"]): int(r["c"]) for r in rung_rows}
        spearman = _spearman(conn)
    finally:
        conn.close()

    coverage = Coverage(
        watchlist=int(watch["c"]) if watch else 0,
        delivered=int(delivered["c"]) if delivered else 0,
        gate_only=int(gate_only["c"]) if gate_only else 0,
        measured=measured,
        by_rung=by_rung,
    )
    return coverage.render(spearman)


def _spearman(conn: sqlite3.Connection) -> float | None:
    """Rank correlation between predicted impact and matured impact, if both exist.

    Reported only when there are enough pairs to mean anything; a coefficient over four
    papers is noise dressed as a number.
    """
    rows = conn.execute(
        "SELECT w.day0_impact_forecast AS pred, o.matured_impact AS obs"
        " FROM outcomes o JOIN watchlist w ON w.arxiv_id = o.arxiv_id"
        " WHERE o.status='measured' AND o.matured_impact IS NOT NULL"
        "   AND w.day0_impact_forecast IS NOT NULL"
    ).fetchall()
    if len(rows) < 8:
        return None
    preds = [float(r["pred"]) for r in rows]
    obs = [float(r["obs"]) for r in rows]
    return _rank_correlation(preds, obs)


def _rank_correlation(xs: list[float], ys: list[float]) -> float | None:
    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        for rank, idx in enumerate(order):
            out[idx] = float(rank)
        return out

    rx, ry = ranks(xs), ranks(ys)
    n = len(rx)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    den_x = sum((a - mx) ** 2 for a in rx) ** 0.5
    den_y = sum((b - my) ** 2 for b in ry) ** 0.5
    if den_x == 0 or den_y == 0:
        return None
    return float(num / (den_x * den_y))


def run_prune(cfg: Settings, older_than_days: int) -> str:
    """Apply the §10 retention rules.

    Only two things are ever pruned, and both are bulky audit blobs: `enrichment.payload`
    and `outcomes.raw`. Derived integers and `matured_impact` are kept forever, because they
    are the labels (§2, principle 7).
    """
    cutoff = (datetime.now(UTC) - timedelta(days=older_than_days)).isoformat()
    conn = _connect(cfg)
    try:
        e = conn.execute("DELETE FROM enrichment WHERE fetched_at < ?", (cutoff,))
        o = conn.execute(
            "UPDATE outcomes SET raw='{}' WHERE measured_at < ? AND raw != '{}'", (cutoff,)
        )
        conn.commit()
        return (
            f"prune older than {older_than_days}d: "
            f"{e.rowcount} enrichment payloads deleted, {o.rowcount} outcome raw blobs cleared"
        )
    finally:
        conn.close()


def notes_from_feedback(cfg: Settings) -> list[str]:
    """Free-text replies, for the weekly preference extraction (§8.3).

    v0 stores them; the extraction itself is v1. Returned as raw strings rather than parsed
    preferences so nothing silently becomes policy without review.
    """
    conn = _connect(cfg)
    try:
        rows = conn.execute(
            "SELECT value FROM feedback WHERE kind='reply' ORDER BY created_at DESC LIMIT 200"
        ).fetchall()
        return [str(r["value"]) for r in rows]
    finally:
        conn.close()


def calibration_payload(cfg: Settings, rung_days: int) -> str:
    conn = _connect(cfg)
    try:
        rows = conn.execute(
            "SELECT * FROM calibration WHERE rung_days=? ORDER BY report_date DESC LIMIT 1",
            (rung_days,),
        ).fetchall()
        return json.dumps([dict(r) for r in rows])
    finally:
        conn.close()
