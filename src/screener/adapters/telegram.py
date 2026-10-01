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
        """Send each chunk as its own message. Returns one id per chunk, in order.

        Every chunk after the first is sent with notifications suppressed. Since §8.4 puts one
        paper in each message, a five-paper digest is five messages; without this the reader's
        phone would buzz five times for one digest. Suppression is per-message, so the digest
        still announces itself once.
        """
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
                "disable_notification": bool(i),
            }
            resp = await self._post(
                "sendMessage", payload, context=f"chunk {i + 1}/{len(chunks)} ({len(chunk)} chars)"
            )
            result = resp.get("result") or {}
            assert isinstance(result, dict)
            ids.append(MessageId(str(result.get("message_id"))))
        return ids

    async def _post(
        self, method: str, payload: dict[str, Any], *, context: str = ""
    ) -> dict[str, Any]:
        """POST one Bot API method.

        Two behaviours that the first version got wrong, both of which turned a diagnosable
        Telegram error into an uninformative one:

        * **the response body is read before any status check.** `raise_for_status()` on a 4xx
          discards the body, and Telegram puts the actual reason in it — "chat not found",
          "can't parse entities", "message is too long". The bare httpx message says none of
          that.
        * **only retryable failures are retried.** A 400 is permanent: retrying it three times
          just delays the error and triples the log noise. 429 and 5xx are retried.
        """
        last: Exception | None = None
        for attempt in range(3):
            try:
                resp = await self._client.post(self._url(method), json=payload)
            except httpx.HTTPError as exc:
                last = exc
                await asyncio.sleep(2**attempt)
                continue

            body: dict[str, Any] = {}
            try:
                parsed = resp.json()
                if isinstance(parsed, dict):
                    body = parsed
            except ValueError:
                pass

            if resp.status_code == 429:
                retry_after = float((body.get("parameters") or {}).get("retry_after", 1))
                log.warning("telegram.rate_limited", retry_after=retry_after, method=method)
                await asyncio.sleep(retry_after)
                continue

            if body.get("ok"):
                return body

            description = str(body.get("description") or resp.reason_phrase or "no description")
            error_code = body.get("error_code", resp.status_code)
            detail = f"telegram {method} failed: {error_code} {description}"
            if context:
                detail += f" [{context}]"

            if resp.status_code >= 500:
                last = RuntimeError(detail)
                log.warning("telegram.server_error", method=method, status=resp.status_code)
                await asyncio.sleep(2**attempt)
                continue

            # A 4xx that is not 429 will not improve with repetition.
            log.error(
                "telegram.permanent_error", method=method, status=resp.status_code, detail=detail
            )
            raise RuntimeError(detail)

        raise RuntimeError(f"telegram {method} failed after retries: {last}")

    # -- feedback (§8.3) ---------------------------------------------------------------

    async def get_updates(self, offset: int | None) -> tuple[list[dict[str, Any]], int | None]:
        """Fetch `message` updates since `offset`.

        The caller persists the returned offset in `kv_state`, so each update is consumed
        exactly once even across restarts. Text only — see the module docstring on why
        reactions cannot be a live mechanism in a private chat.
        """
        payload: dict[str, Any] = {
            # Reaction updates only arrive for an administrator bot, but subscribing costs
            # nothing and means no code change is needed once it is promoted.
            "allowed_updates": ["message", "message_reaction"],
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
