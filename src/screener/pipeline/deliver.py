"""Delivery with the outbox fallback (§4 stage 8, §9).

On a failed send the rendered digest is written to `outbox/{date}.html` and the heartbeat
fires. The next run **retries it automatically** at the head of the run, labelled
"previously unsent" — nothing is ever "offered" to a human, because there is no human in the
loop (§1.2, *Unattended*).
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import structlog

from screener.domain.models import Digest
from screener.domain.types import MessageId
from screener.ports import Notifier

log = structlog.get_logger(__name__)

OUTBOX = Path("outbox")


class DeliveryFailed(RuntimeError):
    pass


async def deliver(
    notifier: Notifier,
    digest: Digest,
    *,
    now: datetime,
    dry_run: bool,
    outbox: Path = OUTBOX,
) -> list[MessageId]:
    """Send, or persist to the outbox. Never raises — delivery failure is not run failure."""
    if dry_run:
        log.info("deliver.dry_run", chunks=len(digest.chunks))
        return []

    try:
        ids = await notifier.send(digest.chunks)
    except Exception as exc:
        log.error("deliver.failed", error=str(exc))
        _write_outbox(digest, now, outbox)
        raise DeliveryFailed(str(exc)) from exc

    if len(ids) != len(digest.chunks):
        log.warning("deliver.id_count_mismatch", ids=len(ids), chunks=len(digest.chunks))
    log.info("deliver.sent", messages=len(ids))
    return ids


def _write_outbox(digest: Digest, now: datetime, outbox: Path) -> Path:
    outbox.mkdir(parents=True, exist_ok=True)
    path = outbox / f"{now.date().isoformat()}.html"
    path.write_text("\n\n<hr/>\n\n".join(digest.chunks))
    (outbox / f"{now.date().isoformat()}.json").write_text(
        json.dumps(
            {
                "written_at": now.isoformat(),
                "chunks": digest.chunks,
                "items": [i.model_dump() for i in digest.items],
            },
            indent=2,
        )
    )
    log.info("deliver.outboxed", path=str(path))
    return path


#: Separator between message chunks in the rendered `.html` artifact.
_HTML_SEPARATOR = "\n\n<hr/>\n\n"


def pending_outbox(outbox: Path = OUTBOX) -> Path | None:
    """The oldest unsent digest, if any. Retried at the head of the next run.

    Accepts **either** artifact. Keying the retry solely on the `.json` sidecar meant that
    losing that one file stranded the digest forever: the papers are already in the seen-set,
    so no later run would ever rebuild it. The human-readable `.html` is the durable artifact
    and is now sufficient on its own.
    """
    if not outbox.exists():
        return None
    candidates = sorted(outbox.glob("*.json")) or sorted(outbox.glob("*.html"))
    return candidates[0] if candidates else None


def load_outbox(path: Path) -> tuple[list[str], list[tuple[str, int, int]]]:
    """Recover the chunks to send, from either artifact.

    From `.json` the item→chunk map is available; from `.html` it is not, and the chunks are
    reconstructed by splitting on the separator that wrote them. An empty item list is honest:
    it means the digest can be re-sent but not attributed.
    """
    if path.suffix == ".json":
        data = json.loads(path.read_text())
        chunks = [str(c) for c in data["chunks"]]
        items = [
            (str(i["arxiv_id"]), int(i["version"]), int(i["chunk_index"])) for i in data["items"]
        ]
        return chunks, items
    return path.read_text().split(_HTML_SEPARATOR), []


def retire_outbox(path: Path) -> None:
    """Remove the whole digest, both artifacts.

    Retiring only the file that was passed in would leave the sibling behind — and since
    `pending_outbox` now accepts either one, an orphan would be re-sent forever.
    """
    path.unlink(missing_ok=True)
    for suffix in (".json", ".html"):
        path.with_suffix(suffix).unlink(missing_ok=True)


def today_stamp(now: datetime) -> date:
    return now.date()
