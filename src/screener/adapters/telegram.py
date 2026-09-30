"""Telegram adapter (§9).

Sends with `parse_mode=HTML` and disabled link previews, and polls `getUpdates` for
**`message` updates only**.

Reactions are deliberately absent. `message_reaction` requires the bot to be an
*administrator* of the chat, and admin status is a group/channel-only concept — a bot cannot
be an admin of a 1:1 DM, so zero reaction updates ever arrive in a private chat. Business
Bot mode does not help (the update has no `business_connection_id`), and neither does MTProto
push. The only thing that works is user-session polling, which would mean shipping a second
long-lived Telethon service to read an emoji. A 👍 *reply* is a text message, so the existing
reply path already covers the use case at zero cost (§8.3).
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

import httpx
import structlog

from screener.domain.types import MessageId

log = structlog.get_logger(__name__)

API = "https://api.telegram.org"
CHUNK_LIMIT = 4096
SEND_INTERVAL_S = 1.0  # Telegram allows <= 1 message/second per chat


class TelegramNotifier:
    """Implements `ports.Notifier` and `ports.FeedbackSource`."""

    def __init__(self, token: str, chat_id: str, *, timeout: float = 30.0) -> None:
        self.token = token
        self.chat_id = chat_id
        self._client = httpx.AsyncClient(timeout=timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> TelegramNotifier:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    def _url(self, method: str) -> str:
        return f"{API}/bot{self.token}/{method}"

    async def send(self, chunks: Sequence[str]) -> list[MessageId]:
        """Send each chunk as its own message. Returns one id per chunk, in order."""
        ids: list[MessageId] = []
        for i, chunk in enumerate(chunks):
            if len(chunk) > CHUNK_LIMIT:
                raise AssertionError(f"refusing to send a {len(chunk)}-char chunk")
            if i:
                await asyncio.sleep(SEND_INTERVAL_S)
            payload: dict[str, Any] = {
                "chat_id": self.chat_id,
                "text": chunk,
                "parse_mode": "HTML",
                "link_preview_options": {"is_disabled": True},
            }
            resp = await self._post("sendMessage", payload)
            result = resp.get("result") or {}
            assert isinstance(result, dict)
            ids.append(MessageId(str(result.get("message_id"))))
        return ids

    async def _post(self, method: str, payload: dict[str, Any]) -> dict[str, Any]:
        last: Exception | None = None
        for attempt in range(3):
            try:
                resp = await self._client.post(self._url(method), json=payload)
                if resp.status_code == 429:
                    retry_after = float(
                        (resp.json() or {}).get("parameters", {}).get("retry_after", 1)
                    )
                    log.warning("telegram.rate_limited", retry_after=retry_after)
                    await asyncio.sleep(retry_after)
                    continue
                resp.raise_for_status()
                body = resp.json()
                if not body.get("ok"):
                    raise RuntimeError(f"telegram error: {body.get('description')}")
                return dict(body)
            except (httpx.HTTPError, RuntimeError) as exc:
                last = exc
                await asyncio.sleep(2**attempt)
        raise RuntimeError(f"telegram {method} failed after retries: {last}")

    # -- feedback (§8.3) ---------------------------------------------------------------

    async def get_updates(self, offset: int | None) -> tuple[list[dict[str, Any]], int | None]:
        """Fetch `message` updates since `offset`.

        The caller persists the returned offset in `kv_state`, so each update is consumed
        exactly once even across restarts. Text only — see the module docstring on why
        reactions cannot be a live mechanism in a private chat.
        """
        payload: dict[str, Any] = {
            "allowed_updates": ["message"],  # NOT message_reaction — see module docstring
            "timeout": 0,
        }
        if offset is not None:
            payload["offset"] = offset
        body = await self._post("getUpdates", payload)
        updates: list[dict[str, Any]] = list(body.get("result") or [])
        next_offset = offset
        for upd in updates:
            update_id = upd.get("update_id")
            if isinstance(update_id, int):
                next_offset = update_id + 1
        return updates, next_offset

    async def send_message(self, text: str) -> MessageId:
        resp = await self._post(
            "sendMessage",
            {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": "HTML",
                "link_preview_options": {"is_disabled": True},
            },
        )
        return MessageId(str((resp.get("result") or {}).get("message_id")))

    async def get_me(self) -> dict[str, Any]:
        return dict(await self._post("getMe", {}))
