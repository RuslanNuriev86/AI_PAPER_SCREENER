"""CLI (§3). One entrypoint; subcommands map to the documented surface.

Commands are thin: they wire dependencies and print a result. All logic lives in `pipeline/`
and `domain/`, so everything here is testable by calling the pipeline directly.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated

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
    config_dir: Annotated[str, typer.Option(help="config directory")] = str(CONFIG_DIR),
) -> None:
    """Re-score a stored run under current (or overridden) config. No network, no sends."""
    from screener.pipeline.replay import replay_run

    cfg = _settings(config_dir)
    if selection is not None:
        import yaml

        cfg.selection = Selection.model_validate(yaml.safe_load(selection.read_text()) or {})
    result = replay_run(cfg, date)
    typer.echo(result.render())
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
    configure_logging()
    try:
        app()
    except ConfigError as exc:
        typer.secho(f"config error: {exc}", fg=typer.colors.RED)
        raise typer.Exit(2) from exc


if __name__ == "__main__":
    main()
