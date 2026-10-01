"""Feedback capture (§8.3). Text replies *and* emoji reactions.

Both paths are live. A reply to a digest message is attributed through `reply_to_message`; a
reaction is attributed through the message it was placed on. Both resolve to a paper via the
`deliveries` table, which is the only record of which chunk carried which item.

One real constraint, measured rather than assumed: Telegram only delivers `message_reaction`
updates when the bot is an **administrator** of the chat. In `ai_papers` the bot is currently a
plain member, so reaction updates do not arrive yet — enabling the subscription is harmless and
they start flowing the moment it is promoted. Text replies work either way, which is why both
mechanisms are kept rather than swapping one for the other.

A reaction must also be *removable*: `new_reaction` is the authoritative current set for that
user, so each update deletes that user's prior reactions on the message before inserting the new
ones. Without that, taking back a 👎 would leave it counted forever.
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
    """Consume new replies and reactions exactly once, using the offset in `kv_state`."""
    raw_offset = deps.repo.get_state(OFFSET_KEY)
    offset = int(raw_offset) if raw_offset else None
    updates, next_offset = await deps.notifier.get_updates(offset)

    stored = 0
    reactions = 0
    for update in updates:
        msg = _message_from_update(update)
        if msg is not None:
            stored += _store_reply(deps, msg)
        reaction = _reaction_from_update(update)
        if reaction is not None:
            reactions += _store_reaction(deps, reaction)

    if next_offset is not None:
        deps.repo.set_state(OFFSET_KEY, str(next_offset))
    log.info("feedback.polled", updates=len(updates), stored=stored, reactions=reactions)
    return stored + reactions


def _user_of(msg: dict[str, object]) -> tuple[int, str | None]:
    """Telegram user id and display name, for per-user aggregates.

    `tg_user_id` is 0 when Telegram omits it, which the schema treats as "unknown" rather than
    letting a NULL defeat the primary key's uniqueness (SQLite considers NULLs distinct).
    """
    # A message carries its author under `from`; a `message_reaction` update carries it under
    # `user`. Reading only `from` attributed every reaction to user 0, so two readers reacting
    # with the same emoji collided on the primary key and one of them vanished.
    sender = msg.get("from") or msg.get("user")
    if not isinstance(sender, dict):
        return 0, None
    uid = sender.get("id")
    name = sender.get("username") or sender.get("first_name")
    return (int(uid) if isinstance(uid, int) else 0), (str(name) if name else None)


def _store_reply(deps: Deps, msg: dict[str, object]) -> int:
    text = str(msg.get("text") or "")
    parsed = parse_reply(text)
    if parsed is None:
        return 0
    kind, value = parsed
    reply_to = msg.get("reply_to_message")
    if not isinstance(reply_to, dict):
        return 0
    replied_id = str(reply_to.get("message_id"))
    run_id, arxiv_id = _attribute(deps, replied_id)
    if arxiv_id is None:
        log.info("feedback.unattributed", message_id=replied_id)
        return 0
    uid, name = _user_of(msg)
    deps.repo.add_feedback(
        message_id=replied_id,
        run_id=run_id or "",
        arxiv_id=arxiv_id,
        kind=kind,
        value=value,
        created_at=datetime.now(UTC),
        tg_user_id=uid,
        tg_user_name=name,
    )
    return 1


def _reaction_from_update(update: dict[str, object]) -> dict[str, object] | None:
    reaction = update.get("message_reaction")
    return reaction if isinstance(reaction, dict) else None


def _emoji_of(reactions: object) -> list[str]:
    """Emoji in a reaction list. Custom emoji fall back to a stable placeholder.

    A custom emoji has no codepoint to show, so it is recorded as `custom:<id>`: the UI can count
    it and label it honestly instead of rendering a blank cell.
    """
    if not isinstance(reactions, list):
        return []
    out: list[str] = []
    for item in reactions:
        if not isinstance(item, dict):
            continue
        if item.get("emoji"):
            out.append(str(item["emoji"]))
        elif item.get("custom_emoji_id"):
            out.append(f"custom:{item['custom_emoji_id']}")
        elif item.get("type"):
            out.append(f"paid:{item['type']}")
    return out


def _store_reaction(deps: Deps, reaction: dict[str, object]) -> int:
    """Replace one user's reactions on one message with the authoritative new set."""
    message_id = str(reaction.get("message_id"))
    run_id, arxiv_id = _attribute(deps, message_id)
    if arxiv_id is None:
        log.info("feedback.unattributed_reaction", message_id=message_id)
        return 0
    uid, name = _user_of(reaction)
    # `new_reaction` is the complete current set, so the old rows must go first — that is what
    # makes removing a reaction actually remove it.
    deps.repo.remove_feedback(message_id, arxiv_id, "reaction", uid)
    emojis = _emoji_of(reaction.get("new_reaction"))
    for emoji in emojis:
        deps.repo.add_feedback(
            message_id=message_id,
            run_id=run_id or "",
            arxiv_id=arxiv_id,
            kind="reaction",
            value=emoji,
            created_at=datetime.now(UTC),
            tg_user_id=uid,
            tg_user_name=name,
        )
    return len(emojis)


def _attribute(deps: Deps, message_id: str) -> tuple[str | None, str | None]:
    """message_id -> (run_id, arxiv_id), or (None, None) when it cannot be known.

    A Telegram message can carry **more than one paper** — `compose` packs items into chunks up to
    the 4096-character limit, and in practice most messages hold two. A reaction is attached to
    the *message*, so when it holds two papers the reaction cannot be attributed to either one.
    This used to return "first by rank", which quietly credited every reaction on a two-paper
    message to whichever paper happened to rank higher.

    Returning nothing is deliberate: a reaction we cannot place is logged and dropped, which is
    recoverable. A reaction filed against the wrong paper is not — it silently distorts the
    reader ranking that `most rated papers by users` is built on.
    """
    papers = deps.repo.papers_in_message(message_id)
    if not papers:
        return None, None
    if len(papers) > 1:
        log.warning(
            "feedback.ambiguous_message",
            message_id=message_id,
            papers=papers,
            hint="one message carried several papers; a reaction cannot be placed (see §8.3)",
        )
        return None, None
    row = deps.repo.delivery_by_message(message_id)
    if row is None:
        return None, None
    return row
