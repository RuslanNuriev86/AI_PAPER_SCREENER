"""Feedback capture (§8.3). Text replies only.

Reactions cannot be a live mechanism in a 1:1 chat: `message_reaction` requires the bot to be
an *administrator*, and admin status only exists in groups and channels. The design's answer
is that a 👍 reply **is** a text message, so the reply path we already have captures it.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime

import structlog

from screener.config import Settings
from screener.deps import Deps

log = structlog.get_logger(__name__)

OFFSET_KEY = "telegram.get_updates_offset"

_RATING_RE = re.compile(r"^\s*(👍|👎|🔥)\s*(.*)$", re.S)

_RATING_VALUES = {"👍": "up", "👎": "down", "🔥": "fire"}


def parse_reply(text: str) -> tuple[str, str] | None:
    """Classify a reply as ('rating', value) or ('reply', text).

    A leading emoji rates; anything else is stored verbatim for the weekly preference
    extraction. Emoji-only messages are the low-friction path that replaces reactions.
    """
    stripped = (text or "").strip()
    if not stripped:
        return None
    m = _RATING_RE.match(stripped)
    if m:
        return "rating", _RATING_VALUES[m.group(1)]
    return "reply", stripped


def _message_from_update(update: dict[str, object]) -> dict[str, object] | None:
    msg = update.get("message")
    if not isinstance(msg, dict):
        return None
    if not isinstance(msg.get("reply_to_message"), dict):
        # Only replies are attributable to a paper; a stray message is not feedback.
        return None
    return msg


async def poll_feedback(deps: Deps, cfg: Settings) -> int:
    """Consume new replies exactly once, using the offset stored in `kv_state`."""
    raw_offset = deps.repo.get_state(OFFSET_KEY)
    offset = int(raw_offset) if raw_offset else None
    updates, next_offset = await deps.notifier.get_updates(offset)

    stored = 0
    for update in updates:
        msg = _message_from_update(update)
        if msg is None:
            continue
        text = str(msg.get("text") or "")
        parsed = parse_reply(text)
        if parsed is None:
            continue
        kind, value = parsed
        reply_to = msg.get("reply_to_message")
        assert isinstance(reply_to, dict)
        replied_id = str(reply_to.get("message_id"))
        run_id, arxiv_id = _attribute(deps, replied_id)
        if arxiv_id is None:
            log.info("feedback.unattributed", message_id=replied_id)
            continue
        deps.repo.add_feedback(
            message_id=replied_id,
            run_id=run_id or "",
            arxiv_id=arxiv_id,
            kind=kind,
            value=value,
            created_at=datetime.now(UTC),
        )
        stored += 1

    if next_offset is not None:
        deps.repo.set_state(OFFSET_KEY, str(next_offset))
    log.info("feedback.polled", updates=len(updates), stored=stored)
    return stored


def _attribute(deps: Deps, message_id: str) -> tuple[str | None, str | None]:
    """message_id -> (run_id, arxiv_id) via the deliveries table.

    This is exactly why `compose` returns an item→chunk map: `send` returns one id per
    *chunk*, and without the map there is no way to join a replied-to message back to a
    paper (§10).
    """
    row = deps.repo.delivery_by_message(message_id)
    if row is None:
        return None, None
    return row
