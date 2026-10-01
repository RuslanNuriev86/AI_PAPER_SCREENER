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


# --- reaction capture (§8.3) ---------------------------------------------------------------
# Telegram delivers `message_reaction` only to an administrator bot, so this path cannot be
# checked against the live API yet. It is tested here because the *removal* semantics are easy to
# get subtly wrong: a taken-back 👎 that stays counted would quietly corrupt the reader ranking.


def _reaction_update(message_id: int, uid: int, emojis: list[str]) -> dict[str, object]:
    return {
        "update_id": 1,
        "message_reaction": {
            "chat": {"id": -5402},
            "message_id": message_id,
            "user": {"id": uid, "username": f"u{uid}"},
            "new_reaction": [{"type": "emoji", "emoji": e} for e in emojis],
        },
    }


def test_emoji_extraction_handles_custom_and_paid_reactions() -> None:
    from screener.pipeline.feedback import _emoji_of

    assert _emoji_of([{"type": "emoji", "emoji": "👍"}]) == ["👍"]
    assert _emoji_of([{"type": "custom_emoji", "custom_emoji_id": "123"}]) == ["custom:123"]
    assert _emoji_of([{"type": "paid"}]) == ["paid:paid"]
    assert _emoji_of(None) == []


def test_the_reader_ranking_weights_are_shared_with_the_poller() -> None:
    """One definition of what a 👎 is worth, or the UI and the digest will disagree."""
    from screener.web.queries import REACTION_WEIGHTS

    assert REACTION_WEIGHTS["👍"] > 0
    assert REACTION_WEIGHTS["👎"] < 0
    assert REACTION_WEIGHTS["🔥"] > REACTION_WEIGHTS["👍"], (
        "a fire is a stronger signal than a thumbs up"
    )


async def test_reactions_are_stored_per_user_and_removal_is_honoured(repo) -> None:
    from screener.pipeline.feedback import poll_feedback

    class _Notifier:
        def __init__(self, updates: list[dict[str, object]]) -> None:
            self.updates = updates

        async def get_updates(self, offset: int | None):  # type: ignore[no-untyped-def]
            return self.updates, 99

    repo._conn.execute(
        "INSERT INTO runs (run_id, started_at, status, mode, config_hash, stats, cost_usd)"
        " VALUES ('r1', '2026-10-01T12:00:00+00:00', 'ok', 'daily', 'h', '{}', 0)"
    )
    # `deliveries` has a foreign key to `papers`, so the paper must exist first.
    repo._conn.execute(
        "INSERT INTO papers (arxiv_id, version, title, abstract, authors, categories,"
        " primary_category, submitted_at, updated_at, abs_url, pdf_url, first_seen_at)"
        " VALUES ('2609.11111',1,'T','A','[]','[]','cs.AI','2026-09-17T00:00:00+00:00',"
        "'2026-09-17T00:00:00+00:00','u','u','2026-10-01T12:00:00+00:00')"
    )
    repo._conn.execute(
        "INSERT INTO deliveries (run_id, arxiv_id, version, kind, rank, score, message_id,"
        " sent_at) VALUES ('r1','2609.11111',1,'digest',1,7.0,'77','2026-10-01T12:00:00+00:00')"
    )
    repo._conn.commit()

    class _Deps:
        def __init__(self) -> None:
            self.repo = repo
            self.notifier = _Notifier([])

    deps = _Deps()
    deps.notifier.updates = [_reaction_update(77, 11, ["👍"]), _reaction_update(77, 22, ["👍"])]
    assert await poll_feedback(deps, None) == 2  # type: ignore[arg-type]

    rows = repo._conn.execute(
        "SELECT tg_user_id, value FROM feedback WHERE kind='reaction' ORDER BY tg_user_id"
    ).fetchall()
    assert len(rows) == 2, "two readers giving the same emoji must be two rows, not one"
    assert {r["tg_user_id"] for r in rows} == {11, 22}

    # User 11 takes the reaction back: `new_reaction` is empty, so the row must disappear.
    deps.notifier.updates = [_reaction_update(77, 11, [])]
    await poll_feedback(deps, None)  # type: ignore[arg-type]
    remaining = repo._conn.execute(
        "SELECT tg_user_id FROM feedback WHERE kind='reaction'"
    ).fetchall()
    assert [r["tg_user_id"] for r in remaining] == [22], "a removed reaction must stop counting"


async def test_an_unattributable_reaction_is_ignored(repo) -> None:
    """A reaction on a message we never sent is not feedback about a paper."""
    from screener.pipeline.feedback import poll_feedback

    class _Notifier:
        async def get_updates(self, offset: int | None):  # type: ignore[no-untyped-def]
            return [_reaction_update(999, 11, ["👍"])], 99

    class _Deps:
        def __init__(self) -> None:
            self.repo = repo
            self.notifier = _Notifier()

    assert await poll_feedback(_Deps(), None) == 0  # type: ignore[arg-type]


async def test_a_reaction_on_a_two_paper_message_is_not_guessed(repo) -> None:
    """One message, two papers, one reaction: attribution is genuinely unknowable.

    `compose` packs items into chunks up to Telegram's 4096-character limit, and in the live
    database four of five messages carried two papers. The old code returned "first by rank", so
    every reaction on such a message was credited to whichever paper ranked higher — a silent
    fabrication in the reader ranking. Dropping it is recoverable; mis-filing it is not.
    """
    from screener.pipeline.feedback import poll_feedback

    class _Notifier:
        def __init__(self, updates: list[dict[str, object]]) -> None:
            self.updates = updates

        async def get_updates(self, offset: int | None):  # type: ignore[no-untyped-def]
            return self.updates, 99

    repo._conn.execute(
        "INSERT INTO runs (run_id, started_at, status, mode, config_hash, stats, cost_usd)"
        " VALUES ('r1','2026-10-01T12:00:00+00:00','ok','daily','h','{}',0)"
    )
    for arxiv_id, rank in (("2609.11111", 1), ("2609.22222", 2)):
        repo._conn.execute(
            "INSERT INTO papers (arxiv_id, version, title, abstract, authors, categories,"
            " primary_category, submitted_at, updated_at, abs_url, pdf_url, first_seen_at)"
            " VALUES (?,1,'T','A','[]','[]','cs.AI','2026-09-17T00:00:00+00:00',"
            "'2026-09-17T00:00:00+00:00','u','u','2026-10-01T12:00:00+00:00')",
            (arxiv_id,),
        )
        repo._conn.execute(
            "INSERT INTO deliveries (run_id, arxiv_id, version, kind, rank, score, message_id,"
            " sent_at) VALUES ('r1',?,1,'digest',?,7.0,'77','2026-10-01T12:00:00+00:00')",
            (arxiv_id, rank),
        )
    repo._conn.commit()

    assert repo.papers_in_message("77") == ["2609.11111", "2609.22222"]

    class _Deps:
        def __init__(self) -> None:
            self.repo = repo
            self.notifier = _Notifier([_reaction_update(77, 11, ["🔥"])])

    assert await poll_feedback(_Deps(), None) == 0  # type: ignore[arg-type]
    assert repo._conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0] == 0


async def test_a_reply_to_a_two_paper_message_is_also_not_guessed(repo) -> None:
    """Replies have the same ambiguity as reactions: quoting a message quotes all of it."""
    from screener.pipeline.feedback import poll_feedback

    class _Notifier:
        async def get_updates(self, offset: int | None):  # type: ignore[no-untyped-def]
            return [
                {
                    "update_id": 5,
                    "message": {
                        "text": "👍",
                        "from": {"id": 11, "username": "ada"},
                        "reply_to_message": {"message_id": 77},
                    },
                }
            ], 99

    repo._conn.execute(
        "INSERT INTO runs (run_id, started_at, status, mode, config_hash, stats, cost_usd)"
        " VALUES ('r1','2026-10-01T12:00:00+00:00','ok','daily','h','{}',0)"
    )
    for arxiv_id, rank in (("2609.11111", 1), ("2609.22222", 2)):
        repo._conn.execute(
            "INSERT INTO papers (arxiv_id, version, title, abstract, authors, categories,"
            " primary_category, submitted_at, updated_at, abs_url, pdf_url, first_seen_at)"
            " VALUES (?,1,'T','A','[]','[]','cs.AI','2026-09-17T00:00:00+00:00',"
            "'2026-09-17T00:00:00+00:00','u','u','2026-10-01T12:00:00+00:00')",
            (arxiv_id,),
        )
        repo._conn.execute(
            "INSERT INTO deliveries (run_id, arxiv_id, version, kind, rank, score, message_id,"
            " sent_at) VALUES ('r1',?,1,'digest',?,7.0,'77','2026-10-01T12:00:00+00:00')",
            (arxiv_id, rank),
        )
    repo._conn.commit()

    class _Deps:
        def __init__(self) -> None:
            self.repo = repo
            self.notifier = _Notifier()

    assert await poll_feedback(_Deps(), None) == 0  # type: ignore[arg-type]


async def test_only_the_first_message_of_a_digest_notifies() -> None:
    """One paper per message must not become one phone buzz per paper (§8.4)."""
    n = TelegramNotifier("token", "123")
    payloads: list[dict[str, Any]] = []

    async def fake_post(url: str, **kw: Any) -> _Resp:
        payloads.append(kw["json"])
        return _Resp(200, {"ok": True, "result": {"message_id": len(payloads)}})

    n._client.post = fake_post  # type: ignore[method-assign]
    try:
        await n.send(["paper one", "paper two", "paper three"])
    finally:
        await n._client.aclose()

    assert [p["disable_notification"] for p in payloads] == [False, True, True]
    assert [p["text"] for p in payloads] == ["paper one", "paper two", "paper three"]
