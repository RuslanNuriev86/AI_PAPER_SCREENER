"""Telegram adapter contract tests (§9).

Two behaviours here turned a diagnosable error into an uninformative one on the first live
send: the response body was discarded on a 4xx (throwing away Telegram's `description`), and a
permanent 400 was retried three times. Both are pinned below.
"""

from __future__ import annotations

from typing import Any

import pytest

from screener.adapters.telegram import TelegramNotifier
from screener.domain.types import MessageId


class _Resp:
    def __init__(self, status_code: int, body: Any) -> None:
        self.status_code = status_code
        self._body = body
        self.reason_phrase = "Bad Request"

    def json(self) -> Any:
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


def _client(notifier: TelegramNotifier, responses: list[_Resp]) -> dict[str, int]:
    seen = {"posts": 0}
    queue = list(responses)

    async def fake_post(url: str, **kw: Any) -> _Resp:
        seen["posts"] += 1
        return queue.pop(0) if queue else responses[-1]

    notifier._client.post = fake_post  # type: ignore[method-assign]
    return seen


OK_SEND = _Resp(200, {"ok": True, "result": {"message_id": 42}})


@pytest.fixture(autouse=True)
def _no_sleeping(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove real backoff and rate-limit waits.

    The adapter sleeps between chunks and between retries; leaving that in makes the suite
    slow and timing-dependent without testing anything about the logic.
    """
    import screener.adapters.telegram as tg

    async def instant(_seconds: float) -> None:
        return None

    monkeypatch.setattr(tg.asyncio, "sleep", instant)


async def test_a_successful_send_returns_one_id_per_chunk() -> None:
    n = TelegramNotifier("token", "123")
    _client(n, [OK_SEND, _Resp(200, {"ok": True, "result": {"message_id": 43}})])
    try:
        ids = await n.send(["first", "second"])
    finally:
        await n._client.aclose()
    assert ids == [MessageId("42"), MessageId("43")]


async def test_telegram_description_is_surfaced_not_discarded() -> None:
    """The whole point: "chat not found" must reach the operator, not just "400"."""
    n = TelegramNotifier("token", "123")
    _client(
        n,
        [
            _Resp(
                400, {"ok": False, "error_code": 400, "description": "Bad Request: chat not found"}
            )
        ],
    )
    try:
        with pytest.raises(RuntimeError) as exc:
            await n.send(["hello"])
    finally:
        await n._client.aclose()
    assert "chat not found" in str(exc.value)
    assert "400" in str(exc.value)


async def test_a_permanent_error_is_not_retried() -> None:
    """A 400 will not improve with repetition; retrying only delays and triples the noise."""
    n = TelegramNotifier("token", "123")
    seen = _client(
        n, [_Resp(400, {"ok": False, "error_code": 400, "description": "chat not found"})]
    )
    try:
        with pytest.raises(RuntimeError):
            await n.send(["hello"])
    finally:
        await n._client.aclose()
    assert seen["posts"] == 1


async def test_the_failing_chunk_is_identified() -> None:
    """With a multi-message digest, which chunk failed is the first question asked."""
    n = TelegramNotifier("token", "123")
    _client(
        n,
        [
            OK_SEND,
            _Resp(
                400,
                {
                    "ok": False,
                    "error_code": 400,
                    "description": "Bad Request: can't parse entities",
                },
            ),
        ],
    )
    try:
        with pytest.raises(RuntimeError) as exc:
            await n.send(["first", "second chunk here"])
    finally:
        await n._client.aclose()
    assert "chunk 2/2" in str(exc.value)
    assert "can't parse entities" in str(exc.value)


async def test_a_5xx_is_retried_then_raises() -> None:
    n = TelegramNotifier("token", "123")
    seen = _client(n, [_Resp(502, {"ok": False, "description": "Bad Gateway"})])
    try:
        with pytest.raises(RuntimeError):
            await n.send(["hello"])
    finally:
        await n._client.aclose()
    assert seen["posts"] == 3, "server errors are transient and should be retried"


async def test_an_oversized_chunk_is_refused_before_any_request() -> None:
    n = TelegramNotifier("token", "123")
    seen = _client(n, [OK_SEND])
    try:
        with pytest.raises(AssertionError, match="refusing to send"):
            await n.send(["x" * 5000])
    finally:
        await n._client.aclose()
    assert seen["posts"] == 0, "the 4096 limit must be enforced locally, not discovered via a 400"


async def test_a_non_json_body_does_not_mask_the_status() -> None:
    """Telegram can return an HTML error page behind a proxy; the status must still surface."""
    n = TelegramNotifier("token", "123")
    _client(n, [_Resp(400, ValueError("not json"))])
    try:
        with pytest.raises(RuntimeError) as exc:
            await n.send(["hello"])
    finally:
        await n._client.aclose()
    assert "400" in str(exc.value)


# --- doctor's chat validation -------------------------------------------------------------
# "token + chat id present" only proves two variables are non-empty. The first real send then
# fails with a bare 400 whose reason lives in the response body, so doctor asks Telegram
# directly instead.


class _StubResp:
    def __init__(self, body: Any) -> None:
        self._body = body

    def json(self) -> Any:
        return self._body


async def _chat_check(body: Any) -> list[str]:  # type: ignore[no-untyped-def]
    from screener.config import Settings
    from screener.doctor import Report, _check_telegram_chat

    class _Client:
        async def get(self, url: str, **kw: Any) -> _StubResp:
            return _StubResp(body)

    cfg = Settings(telegram_bot_token="t", telegram_chat_id="123")  # type: ignore[call-arg]
    r = Report()
    await _check_telegram_chat(r, _Client(), cfg)  # type: ignore[arg-type]
    return r.lines


async def test_doctor_accepts_a_reachable_chat() -> None:
    lines = await _chat_check({"ok": True, "result": {"type": "private", "first_name": "Ada"}})
    assert lines[0].startswith("[  ok  ]")
    assert "private" in lines[0]


async def test_doctor_reports_chat_not_found_with_telegrams_own_words() -> None:
    lines = await _chat_check(
        {"ok": False, "error_code": 400, "description": "Bad Request: chat not found"}
    )
    assert lines[0].startswith("[ fail ]")
    assert "chat not found" in lines[0]
    # The actionable hint must be present: this failure is nearly always "you never started
    # the bot", not a typo in the id.
    assert "/start" in lines[0]


async def test_doctor_survives_a_transport_failure() -> None:
    from screener.config import Settings
    from screener.doctor import Report, _check_telegram_chat

    class _Boom:
        async def get(self, url: str, **kw: Any) -> Any:
            raise RuntimeError("network down")

    cfg = Settings(telegram_bot_token="t", telegram_chat_id="1")  # type: ignore[call-arg]
    r = Report()
    await _check_telegram_chat(r, _Boom(), cfg)  # type: ignore[arg-type]
    assert r.lines[0].startswith("[ warn ]")


async def test_doctor_spots_a_group_id_with_the_sign_dropped() -> None:
    """The real-world failure: a group id pasted without its leading minus.

    `5402616139` is not a valid user chat, so Telegram says "chat not found". The correct id is
    the same digits negated, and doctor should say so rather than leaving the operator to
    compare two numbers by eye.
    """
    from screener.config import Settings
    from screener.doctor import Report, _check_telegram_chat

    class _Client:
        async def get(self, url: str, **kw: Any) -> _StubResp:
            if "getChat" in url:
                return _StubResp(
                    {"ok": False, "error_code": 400, "description": "Bad Request: chat not found"}
                )
            return _StubResp(
                {
                    "ok": True,
                    "result": [
                        {
                            "my_chat_member": {
                                "chat": {
                                    "id": -5402616139,
                                    "type": "group",
                                    "title": "ai_papers",
                                }
                            }
                        }
                    ],
                }
            )

    cfg = Settings(telegram_bot_token="t", telegram_chat_id="5402616139")  # type: ignore[call-arg]
    r = Report()
    await _check_telegram_chat(r, _Client(), cfg)  # type: ignore[arg-type]
    joined = "\n".join(r.lines)
    assert "sign dropped" in joined
    assert "TELEGRAM_CHAT_ID=-5402616139" in joined


async def test_doctor_does_not_invent_a_negation_that_is_not_real() -> None:
    """Only suggest `-<id>` when Telegram actually reports that chat."""
    from screener.config import Settings
    from screener.doctor import Report, _check_telegram_chat

    class _Client:
        async def get(self, url: str, **kw: Any) -> _StubResp:
            if "getChat" in url:
                return _StubResp(
                    {"ok": False, "error_code": 400, "description": "Bad Request: chat not found"}
                )
            return _StubResp({"ok": True, "result": []})

    cfg = Settings(telegram_bot_token="t", telegram_chat_id="123")  # type: ignore[call-arg]
    r = Report()
    await _check_telegram_chat(r, _Client(), cfg)  # type: ignore[arg-type]
    assert "sign dropped" not in "\n".join(r.lines)


# --- outbox durability --------------------------------------------------------------------
# The retry originally keyed on the `.json` sidecar alone. Losing that one file stranded a real
# digest permanently, because its papers were already in the seen-set and no later run would
# rebuild it. The rendered `.html` is the durable artifact and must be sufficient.


def _digest():  # type: ignore[no-untyped-def]
    from screener.domain.models import Digest, DigestItem

    return Digest(
        chunks=["first message", "second message"],
        items=[
            DigestItem(arxiv_id="2509.1", version=1, chunk_index=0),
            DigestItem(arxiv_id="2509.2", version=1, chunk_index=1),
        ],
    )


def test_outbox_writes_both_artifacts(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from datetime import UTC, datetime

    from screener.pipeline.deliver import _write_outbox

    _write_outbox(_digest(), datetime(2026, 9, 30, tzinfo=UTC), tmp_path)
    assert (tmp_path / "2026-09-30.html").exists()
    assert (tmp_path / "2026-09-30.json").exists()


def test_pending_outbox_finds_the_html_when_the_sidecar_is_gone(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from datetime import UTC, datetime

    from screener.pipeline.deliver import _write_outbox, load_outbox, pending_outbox

    _write_outbox(_digest(), datetime(2026, 9, 30, tzinfo=UTC), tmp_path)
    (tmp_path / "2026-09-30.json").unlink()

    found = pending_outbox(tmp_path)
    assert found is not None, "a digest must be recoverable from the html alone"
    chunks, items = load_outbox(found)
    assert chunks == ["first message", "second message"]
    assert items == [], "the item map is not recoverable from html, and is reported as such"


def test_load_outbox_reads_the_json_when_present(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from datetime import UTC, datetime

    from screener.pipeline.deliver import _write_outbox, load_outbox, pending_outbox

    _write_outbox(_digest(), datetime(2026, 9, 30, tzinfo=UTC), tmp_path)
    path = pending_outbox(tmp_path)
    assert path is not None and path.suffix == ".json"
    chunks, items = load_outbox(path)
    assert chunks == ["first message", "second message"]
    assert items == [("2509.1", 1, 0), ("2509.2", 1, 1)]


def test_retiring_removes_both_artifacts(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from datetime import UTC, datetime

    from screener.pipeline.deliver import _write_outbox, pending_outbox, retire_outbox

    _write_outbox(_digest(), datetime(2026, 9, 30, tzinfo=UTC), tmp_path)
    path = pending_outbox(tmp_path)
    assert path is not None
    retire_outbox(path)
    assert not list(tmp_path.iterdir()), "an orphan artifact would be re-sent forever"
    assert pending_outbox(tmp_path) is None


def test_retiring_from_the_html_also_removes_the_sidecar(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from datetime import UTC, datetime

    from screener.pipeline.deliver import _write_outbox, retire_outbox

    _write_outbox(_digest(), datetime(2026, 9, 30, tzinfo=UTC), tmp_path)
    retire_outbox(tmp_path / "2026-09-30.html")
    assert not list(tmp_path.iterdir())
