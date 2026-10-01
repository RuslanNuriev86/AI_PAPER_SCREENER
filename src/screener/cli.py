"""CLI (§3). One entrypoint; subcommands map to the documented surface.

Commands are thin: they wire dependencies and print a result. All logic lives in `pipeline/`
and `domain/`, so everything here is testable by calling the pipeline directly.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Annotated

import httpx
import typer

from screener.config import CONFIG_DIR, Settings, load_settings
from screener.deps import ConfigError, SystemClock, build_deps, configure_logging
from screener.domain.models import Selection
from screener.pipeline.run import execute as run_pipeline

app = typer.Typer(add_completion=False, help="Agent Papers Daily")

ModeOpt = Annotated[str, typer.Option("--mode", help="daily | weekly")]


def _settings(config_dir: str = str(CONFIG_DIR)) -> Settings:
    return load_settings(config_dir)


# ---------------------------------------------------------------------------------------
# run / dry-run
# ---------------------------------------------------------------------------------------


async def _run(cfg: Settings) -> int:
    async with build_deps(cfg, require_network=not cfg.dry_run) as deps:
        run = await run_pipeline(
            cfg,
            clock=SystemClock(),
            repo=deps.repo,
            source=deps.source,
            llm=deps.llm,
            notifier=deps.notifier,
            heartbeat=deps.heartbeat,
            ledger=deps.ledger,
            enricher=deps.enricher,
        )
    typer.echo(f"status={run.status} spent=${run.stats.cost_usd:.4f}")
    picks = run.digest.items if run.digest else []
    for item in picks:
        typer.echo(f"  {item.arxiv_id}v{item.version} -> message chunk {item.chunk_index}")
    if run.digest is None:
        typer.echo("  (no digest: nothing cleared the gate and threshold)")
    return 0


@app.command()
def run(
    mode: Annotated[str, typer.Option("--mode", help="daily | weekly")] = "daily",
    config_dir: Annotated[str, typer.Option(help="config directory")] = str(CONFIG_DIR),
) -> None:
    """One run: fetch, gate, score, compose, deliver."""
    cfg = _settings(config_dir)
    cfg.screener_mode = "live" if mode == "daily" else "weekly"
    raise typer.Exit(asyncio.run(_run(cfg)))


@app.command("dry-run")
def dry_run(
    config_dir: Annotated[str, typer.Option(help="config directory")] = str(CONFIG_DIR),
) -> None:
    """The whole pipeline with no Telegram send and no spend on the digest path.

    Note this still calls the LLM: a dry run that skipped scoring would not exercise the
    ranker, and the point of a dry run is to see the digest you would have gotten.
    """
    cfg = _settings(config_dir)
    cfg.screener_mode = "dry"
    raise typer.Exit(asyncio.run(_run(cfg)))


# ---------------------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------------------


@app.command()
def doctor(
    config_dir: Annotated[str, typer.Option(help="config directory")] = str(CONFIG_DIR),
) -> None:
    """Validate config, credentials, DB, network and clock. Exits non-zero on a hard failure."""
    from screener.doctor import run_doctor

    cfg = _settings(config_dir)
    report = asyncio.run(run_doctor(cfg))
    for line in report.lines:
        typer.echo(line)
    raise typer.Exit(0 if report.ok else 1)


# ---------------------------------------------------------------------------------------
# revisit
# ---------------------------------------------------------------------------------------


async def _revisit(cfg: Settings) -> int:
    from screener.pipeline.revisit import execute_revisit

    async with build_deps(cfg, require_network=False) as deps:
        result = await execute_revisit(
            cfg,
            clock=SystemClock(),
            repo=deps.repo,
            probe=deps.probe,
            heartbeat=deps.heartbeat,
        )
    typer.echo(
        f"revisit due={result.due} measured={result.measured} missed={result.missed} "
        f"errors={len(result.per_source_errors)}"
    )
    return 0


@app.command()
def revisit(
    config_dir: Annotated[str, typer.Option(help="config directory")] = str(CONFIG_DIR),
) -> None:
    """Measure due outcome rungs. Separate job, off the delivery path (§6.6)."""
    raise typer.Exit(asyncio.run(_revisit(_settings(config_dir))))


# ---------------------------------------------------------------------------------------
# replay / backtest
# ---------------------------------------------------------------------------------------


@app.command()
def replay(
    date: Annotated[str, typer.Option("--date", help="YYYY-MM-DD")],
    selection: Annotated[Path | None, typer.Option(help="override selection.yaml")] = None,
    write_outbox: Annotated[
        bool,
        typer.Option("--write-outbox", help="rebuild the digest and park it for delivery"),
    ] = False,
    config_dir: Annotated[str, typer.Option(help="config directory")] = str(CONFIG_DIR),
) -> None:
    """Re-score a stored run under current (or overridden) config. No network, no sends.

    With `--write-outbox` it also rebuilds the rendered digest and parks it, which recovers a
    digest whose outbox artifact was lost: by then the papers are in the seen-set, so no later
    run would ever rebuild it. The next `screener run` then delivers it.
    """
    from screener.pipeline.replay import recover_to_outbox, replay_run

    cfg = _settings(config_dir)
    if selection is not None:
        import yaml

        cfg.selection = Selection.model_validate(yaml.safe_load(selection.read_text()) or {})
    result = replay_run(cfg, date)
    typer.echo(result.render())
    if write_outbox:
        path = recover_to_outbox(cfg, date)
        if path is None:
            typer.secho("nothing to recover for that date", fg=typer.colors.YELLOW)
            raise typer.Exit(1)
        typer.secho(f"parked for delivery: {path}", fg=typer.colors.GREEN)
        typer.echo("the next `screener run` will send it")
    raise typer.Exit(0)


@app.command()
def backtest(
    since: Annotated[str, typer.Option("--since", help="YYYY-MM-DD")],
    weights: Annotated[Path | None, typer.Option(help="candidate weight JSON")] = None,
    config_dir: Annotated[str, typer.Option(help="config directory")] = str(CONFIG_DIR),
) -> None:
    """Re-score stored rankings under candidate weights; report added/dropped papers."""
    from screener.pipeline.replay import backtest_weights

    cfg = _settings(config_dir)
    overrides = json.loads(weights.read_text()) if weights else None
    typer.echo(backtest_weights(cfg, since, overrides))
    raise typer.Exit(0)


# ---------------------------------------------------------------------------------------
# feedback
# ---------------------------------------------------------------------------------------


async def _feedback(cfg: Settings) -> int:
    from screener.pipeline.feedback import poll_feedback

    async with build_deps(cfg, require_network=False) as deps:
        count = await poll_feedback(deps, cfg)
    typer.echo(f"feedback: {count} new")
    return 0


@app.command()
def web(
    host: Annotated[str, typer.Option(help="interface to bind")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="port to bind")] = 8765,
    config_dir: Annotated[str, typer.Option(help="config directory")] = str(CONFIG_DIR),
    allow_remote: Annotated[
        bool, typer.Option("--allow-remote", help="permit binding a non-loopback interface")
    ] = False,
) -> None:
    """Browse what was found, delivered and rated (§17).

    Read-only and localhost-only by default. There is no authentication, so binding anything but
    a loopback address publishes the entire paper history to the network; that requires saying so
    explicitly with --allow-remote.
    """
    import uvicorn

    from screener.web.app import create_app

    cfg = _settings(config_dir)
    loopback = host in {"127.0.0.1", "::1", "localhost"}
    if not loopback and not allow_remote:
        typer.secho(
            f"refusing to bind {host}: this server has no authentication.\n"
            "Re-run with --allow-remote if the network really is trusted, or reach it over an\n"
            "SSH tunnel:  ssh -L "
            f"{port}:127.0.0.1:{port} <host>",
            fg=typer.colors.RED,
        )
        raise typer.Exit(2)
    typer.secho(
        f"serving http://{host}:{port}  (read-only, {cfg.screener_db})", fg=typer.colors.GREEN
    )
    uvicorn.run(create_app(cfg.screener_db), host=host, port=port, log_level="warning")


@app.command()
def explain(
    arxiv_id: Annotated[str, typer.Argument(help="the paper, e.g. 2609.35909")],
    config_dir: Annotated[str, typer.Option(help="config directory")] = str(CONFIG_DIR),
) -> None:
    """Print exactly why a paper scored what it scored (§7.3).

    Reads stored data, so it works long after the digest went out, and shows both halves
    separately — the judged half is an opinion about text, the measured half is a reading from
    an external source at a known age. They are never blended into one unexplained number.
    """
    import json as _json
    import sqlite3

    cfg = _settings(config_dir)
    conn = sqlite3.connect(cfg.screener_db)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT r.*, u.started_at FROM rankings r JOIN runs u USING(run_id)"
            " WHERE r.arxiv_id=? ORDER BY u.started_at DESC LIMIT 1",
            (arxiv_id,),
        ).fetchone()
        if row is None:
            typer.secho(f"no stored rating for {arxiv_id}", fg=typer.colors.YELLOW)
            raise typer.Exit(1)
        arow = conn.execute(
            "SELECT payload, model, prompt_version FROM assessments"
            " WHERE arxiv_id=? AND stage='review' ORDER BY created_at DESC LIMIT 1",
            (arxiv_id,),
        ).fetchone()
        mrow = conn.execute(
            "SELECT citations, stars, venue, code_url, components_present, status, rung_days,"
            " actual_age_days, matured_impact FROM outcomes WHERE arxiv_id=?"
            " ORDER BY rung_days",
            (arxiv_id,),
        ).fetchall()
        # What the adapters read at the rating point. Distinct from the outcome rungs below,
        # which are later re-measurements: this is the evidence the rating was actually built on.
        srow = conn.execute(
            "SELECT payload, fetched_at FROM enrichment WHERE arxiv_id=?"
            " ORDER BY fetched_at DESC LIMIT 1",
            (arxiv_id,),
        ).fetchone()
    finally:
        conn.close()

    typer.secho(f"{arxiv_id}  composite {row['score']:.2f}  ({row['disposition']})", bold=True)
    typer.echo(
        f"rated {str(row['started_at'])[:10]} by {arow['model'] if arow else '?'}"
        f" prompt {arow['prompt_version'] if arow else '?'}"
    )
    typer.echo()
    typer.echo("judged half (from the text):")
    if arow is not None and arow["payload"]:
        payload = _json.loads(str(arow["payload"]))
        components = _json.loads(str(row["components"]))
        weights = _json.loads(str(row["effective_weights"]))
        for dim, raw in payload.items():
            w = weights.get(dim)
            if w is None:
                continue
            typer.echo(f"  {dim:<18} {raw:>4.1f} x {w:.3f} = {raw * w:>5.2f}")
        contrib = sum(components.get(d, 0.0) for d in payload)
        typer.echo(f"  {'':<18} {'':>4}   {'':>5}   {contrib:>5.2f}")
        # Without this the printed arithmetic does not add up to the composite, which is worse
        # than printing nothing: the reader cannot tell a penalty from a bug.
        flags = _json.loads(str(row["soft_flags"] or "[]"))
        if flags:
            penalty = 0.5 * len(flags)
            typer.echo(
                f"  {'soft flags':<18} {'':>4}   {'':>5}   {-penalty:>5.2f}   ({', '.join(flags)})"
            )
            contrib -= penalty
        typer.secho(f"  {'composite':<18} {'':>4}   {'':>5}   {contrib:>5.2f}", bold=True)
    typer.echo()
    typer.echo("measured half (read from external sources at rating time):")
    if srow is not None and srow["payload"]:
        signals = _json.loads(str(srow["payload"]))

        def show(label: str, value: object) -> None:
            typer.echo(f"  {label:<18} " + ("not measured" if value is None else str(value)))

        show("age (days)", signals.get("age_days"))
        show("stars", signals.get("stars"))
        show("citations", signals.get("citations"))
        show("venue", signals.get("venue"))
        show("repo", signals.get("repo_url"))
        typer.echo("  sources: " + (", ".join(signals.get("sources_ok") or []) or "none"))
    else:
        typer.echo("  nothing stored for this paper")

    typer.echo()
    typer.echo("later re-measurements (outcome rungs):")
    if mrow:
        for m in mrow:
            bits = [
                f"{m['stars']}★" if m["stars"] is not None else "no repo",
                f"{m['citations']} citations" if m["citations"] is not None else "unread",
                m["venue"] or "no venue",
            ]
            typer.echo(
                f"  T+{m['rung_days']:<4} ({m['status']}, measured at "
                f"T+{m['actual_age_days']}): " + " · ".join(bits)
            )
    else:
        typer.echo("  no outcome rungs measured yet")
    typer.echo()
    typer.secho(
        "note: the measured half is never fed back into the day-0 score (§6.6.6), so it "
        "is reported, not blended.",
        fg=typer.colors.BRIGHT_BLACK,
    )
    raise typer.Exit(0)


@app.command()
def rearm(
    date: Annotated[str, typer.Option("--date", help="YYYY-MM-DD")],
    config_dir: Annotated[str, typer.Option(help="config directory")] = str(CONFIG_DIR),
) -> None:
    """Make a date's reviewed papers fresh again so the next run re-processes them.

    Use when a digest was reviewed and paid for but never delivered and its outbox entry is
    gone: the papers are in the seen-set and the review prose was never stored, so nothing can
    be replayed — they have to be reviewed again. Only the shortlisted papers are cleared.
    """
    from screener.pipeline.replay import rearm as rearm_date

    ids = rearm_date(_settings(config_dir), date)
    if not ids:
        typer.secho(f"nothing to re-arm for {date}", fg=typer.colors.YELLOW)
        raise typer.Exit(1)
    typer.secho(f"re-armed {len(ids)} papers for {date}", fg=typer.colors.GREEN)
    for arxiv_id in ids:
        typer.echo(f"  {arxiv_id}")
    typer.echo("the next `screener run` will re-review them")


@app.command()
def feedback(
    config_dir: Annotated[str, typer.Option(help="config directory")] = str(CONFIG_DIR),
) -> None:
    """Poll Telegram for replies (text only) and store them as feedback."""
    raise typer.Exit(asyncio.run(_feedback(_settings(config_dir))))


# ---------------------------------------------------------------------------------------
# eval / prune
# ---------------------------------------------------------------------------------------


@app.command()
def eval(
    config_dir: Annotated[str, typer.Option(help="config directory")] = str(CONFIG_DIR),
) -> None:
    """Run the offline suites and report maturity coverage."""
    from screener.pipeline.evaluate import run_eval

    typer.echo(run_eval(_settings(config_dir)))
    raise typer.Exit(0)


@app.command()
def prune(
    older_than_days: Annotated[int, typer.Option("--older-than-days")] = 180,
    config_dir: Annotated[str, typer.Option(help="config directory")] = str(CONFIG_DIR),
) -> None:
    """Apply the §10 retention rules: enrichment payloads and outcome raw blobs only."""
    from screener.pipeline.evaluate import run_prune

    typer.echo(run_prune(_settings(config_dir), older_than_days))
    raise typer.Exit(0)


# ---------------------------------------------------------------------------------------
# stats (a small addition: v0 needs a way to see whether labels are accumulating)
# ---------------------------------------------------------------------------------------


@app.command()
def stats(
    config_dir: Annotated[str, typer.Option(help="config directory")] = str(CONFIG_DIR),
) -> None:
    """Row counts and rung coverage. The health metric that matters for the loop (§12.2)."""
    cfg = _settings(config_dir)
    from screener.adapters.sqlite_repo import SqliteRepository

    repo = SqliteRepository(cfg.screener_db)
    try:
        repo.migrate()
        for table, count in repo.counts().items():
            typer.echo(f"{table:<14} {count}")
    finally:
        repo.close()
    raise typer.Exit(0)


def main() -> None:
    """Entry point.

    Operational failures (a dead source, a rejected credential) get one readable line, not a
    two-hundred-line rich traceback — the traceback is what the first real outage produced, and
    it buried the only useful fact. Set `SCREENER_DEBUG=1` to get the full trace anyway.
    """
    configure_logging()
    debug = os.environ.get("SCREENER_DEBUG", "").strip() not in {"", "0", "false"}
    try:
        app()
    except ConfigError as exc:
        typer.secho(f"config error: {exc}", fg=typer.colors.RED)
        raise typer.Exit(2) from exc
    except httpx.HTTPError as exc:
        if debug:
            raise
        typer.secho(
            f"network error: {type(exc).__name__}: {exc or 'no detail'}\n"
            "the digest was not sent; re-running is safe (the seen-set prevents duplicates)",
            fg=typer.colors.RED,
        )
        raise typer.Exit(1) from exc
    except Exception as exc:
        if debug:
            raise
        typer.secho(f"error: {type(exc).__name__}: {exc}", fg=typer.colors.RED)
        raise typer.Exit(1) from exc


if __name__ == "__main__":
    main()
