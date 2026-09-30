"""`screener doctor` (§12.2): validate everything that can silently break, before it does.

Checks are ordered cheapest-and-most-local first so a broken .env is reported without waiting
on a network timeout. A missing optional integration is a WARN; a missing hard prerequisite is
a FAIL and the command exits non-zero.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from screener.config import CONFIG_DIR, Settings

OK = "  ok  "
WARN = " warn "
FAIL = " fail "


@dataclass
class Report:
    lines: list[str] = field(default_factory=list)
    failures: int = 0
    warnings: int = 0

    @property
    def ok(self) -> bool:
        return self.failures == 0

    def add(self, level: str, name: str, detail: str = "") -> None:
        if level == FAIL:
            self.failures += 1
        elif level == WARN:
            self.warnings += 1
        self.lines.append(f"[{level}] {name:<22} {detail}")


async def run_doctor(cfg: Settings, *, config_dir: str | Path = CONFIG_DIR) -> Report:
    r = Report()
    r.add(OK, "python", _python_version())
    r.add(OK, "clock", datetime.now(UTC).isoformat(timespec="seconds"))

    # --- config files ---
    d = Path(config_dir)
    for name in ("profile.yaml", "selection.yaml", "revisit.yaml", "outcome_scale.yaml"):
        path = d / name
        r.add(
            OK if path.exists() else WARN,
            f"config/{name}",
            "loaded" if path.exists() else "absent — using built-in defaults",
        )

    cats = cfg.profile.categories
    r.add(
        OK if cats else FAIL,
        "profile.categories",
        f"{len(cats)} categories" if cats else "empty: the gate would reject every paper",
    )
    if not cfg.profile.strong_terms:
        r.add(FAIL, "profile.strong_terms", "empty: the gate can never pass anything")
    else:
        r.add(OK, "profile.strong_terms", f"{len(cfg.profile.strong_terms)} terms")

    bad_maps = _weight_maps_off_by(cfg)
    r.add(
        OK if not bad_maps else FAIL,
        "weights sum",
        "every weight map sums to 1.0" if not bad_maps else f"maps not summing to 1.0: {bad_maps}",
    )

    # --- database ---
    db = Path(cfg.screener_db)
    try:
        db.parent.mkdir(parents=True, exist_ok=True)
        probe = db.parent / ".doctor-write-probe"
        probe.write_text("x")
        probe.unlink()
        r.add(OK, "db writable", str(db))
    except OSError as exc:
        r.add(FAIL, "db writable", f"{db}: {exc}")

    from screener.adapters.sqlite_repo import SqliteRepository

    try:
        repo = SqliteRepository(db)
        repo.migrate()
        counts = repo.counts()
        repo.close()
        r.add(OK, "db schema", f"{len(counts)} tables")
    except Exception as exc:
        r.add(FAIL, "db schema", str(exc)[:120])

    # --- secrets ---
    r.add(OK, "llm provider", f"{cfg.llm_base_url}  model={cfg.llm_deep}")
    r.add(
        OK if cfg.llm_api_key else FAIL,
        "LLM credential",
        "DEEPSEEK_API_KEY present"
        if cfg.llm_api_key
        else "missing DEEPSEEK_API_KEY (or LLM_API_KEY): a live run cannot score anything",
    )
    r.add(
        OK,
        "llm thinking mode",
        "enabled (note: thinking mode ignores temperature, so scoring is not repeatable)"
        if cfg.llm_thinking
        else "disabled (temperature honoured; required for reproducible scores)",
    )
    if cfg.telegram_bot_token and cfg.telegram_chat_id:
        r.add(OK, "telegram config", "token + chat id present")
    elif not cfg.telegram_bot_token:
        r.add(FAIL, "telegram config", "TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID required")
    else:
        # Token but no chat id: the id is not something you can look up, it is whatever
        # Telegram reports once the bot has received something. So fetch it and print it.
        r.add(FAIL, "telegram config", "TELEGRAM_CHAT_ID is unset — see candidates below")
    r.add(
        OK if cfg.contact_email and "@" in cfg.contact_email else WARN,
        "CONTACT_EMAIL",
        cfg.contact_email or "missing — arXiv asks for a contact address in the User-Agent",
    )
    r.add(
        OK if cfg.github_token else WARN,
        "GITHUB_TOKEN",
        "set" if cfg.github_token else "unset: 60 req/h cannot cover a daily T+14 cohort",
    )
    r.add(
        OK if cfg.heartbeat_url else WARN,
        "HEARTBEAT_URL",
        "set" if cfg.heartbeat_url else "unset: no dead-man's switch (§12.2)",
    )

    disk = shutil.disk_usage(Path.cwd())
    r.add(
        OK if disk.free > 100 * 1024 * 1024 else WARN,
        "disk free",
        f"{disk.free // (1024 * 1024)} MB",
    )

    # --- proxy environment ---------------------------------------------------------
    # httpx parses every NO_PROXY entry, and a bracketed IPv6 literal ("[::1]") makes it
    # raise InvalidURL while *constructing* a client — so a host with that entry cannot make
    # any HTTP request at all, and the failure surfaces as an unhandled traceback from
    # somewhere unrelated. Diagnosing that is precisely doctor's job.
    client_error = _httpx_client_error()
    if client_error:
        r.add(
            FAIL,
            "proxy env",
            f"httpx cannot construct a client ({client_error}). Usually a NO_PROXY entry in "
            'an unsupported form — e.g. a bracketed IPv6 literal "[::1]" where "::1" is meant.',
        )
    else:
        r.add(OK, "proxy env", "httpx builds clients from the current environment")

    # --- network (cheap HEAD/GETs, all failures are WARNs so doctor runs offline) ---
    if client_error:
        r.add(WARN, "net", "skipped: no HTTP client can be constructed in this environment")
        return r

    async with httpx.AsyncClient(timeout=12.0) as client:
        await _check_http(r, client, "arxiv", _arxiv_probe_url(cfg))
        await _check_llm_auth(r, client, cfg)
        await _check_http(r, client, "github", "https://api.github.com/rate_limit")
        if cfg.telegram_bot_token:
            await _check_http(
                r,
                client,
                "telegram",
                f"https://api.telegram.org/bot{cfg.telegram_bot_token}/getMe",
            )
            if not cfg.telegram_chat_id:
                await _report_candidate_chat_ids(r, client, cfg.telegram_bot_token)
            else:
                await _check_telegram_chat(r, client, cfg)
    return r


async def _check_telegram_chat(r: Report, client: httpx.AsyncClient, cfg: Settings) -> None:
    """Ask Telegram whether the configured chat actually exists.

    `telegram config: token + chat id present` only proves two variables are non-empty. The
    first real send then fails with a bare "400 Bad Request" whose *reason* — "chat not found"
    is the usual one — only appears in the response body. Checking `getChat` here turns that
    into a labelled FAIL before a digest is ever composed, and it costs one API call.
    """
    try:
        resp = await client.get(
            f"https://api.telegram.org/bot{cfg.telegram_bot_token}/getChat",
            params={"chat_id": cfg.telegram_chat_id},
        )
        body = resp.json()
    except Exception as exc:
        r.add(WARN, "telegram chat", f"could not verify chat: {type(exc).__name__}")
        return

    if body.get("ok"):
        result = body.get("result") or {}
        kind = result.get("type", "?")
        name = result.get("title") or result.get("username") or result.get("first_name") or ""
        r.add(OK, "telegram chat", f"reachable: {kind}{f' ({name})' if name else ''}")
        return

    r.add(
        FAIL,
        "telegram chat",
        f"{body.get('error_code')} {body.get('description')} — TELEGRAM_CHAT_ID="
        f"{cfg.telegram_chat_id} is not a chat this bot can post to. "
        "A bot cannot open a conversation: send the bot /start first, then re-run doctor.",
    )
    # A failed getChat is exactly when the working ids are wanted, so list them here rather
    # than making the operator clear the variable to see candidates.
    candidates = await _report_candidate_chat_ids(r, client, cfg.telegram_bot_token)
    configured = cfg.telegram_chat_id.strip()
    if configured and not configured.startswith("-") and f"-{configured}" in dict(candidates):
        r.add(
            FAIL,
            "telegram chat id",
            f"looks like a group id with the sign dropped: set TELEGRAM_CHAT_ID=-{configured}",
        )


def extract_chat_ids(updates: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Pull (chat_id, label) pairs out of a getUpdates payload.

    Covers every place a chat can appear: a normal `message` (a DM, or a group message if the
    bot is not in privacy mode), a `channel_post` (a channel, where the bot sees posts but no
    messages), and `my_chat_member` (the bot being added — which arrives *before* anyone has to
    type anything, so it is the reliable way to get a group or channel id).

    Calling getUpdates without an `offset` does not consume anything, so this is safe to run
    while `screener feedback` owns the real offset.
    """
    found: dict[str, str] = {}
    for update in updates:
        for key in ("message", "channel_post", "edited_message", "my_chat_member"):
            payload = update.get(key)
            if not isinstance(payload, dict):
                continue
            chat = payload.get("chat")
            if not isinstance(chat, dict) or "id" not in chat:
                continue
            chat_id = str(chat["id"])
            kind = str(chat.get("type") or key)
            name = chat.get("title") or chat.get("username") or chat.get("first_name") or ""
            found.setdefault(chat_id, f"{kind}{f' ({name})' if name else ''}")
    return sorted(found.items())


async def _check_llm_auth(r: Report, client: httpx.AsyncClient, cfg: Settings) -> None:
    """Query the provider *with* the credential.

    The shared probe client is unauthenticated, so it reported the LLM endpoint as
    "HTTP 401 (reachable; auth checked separately)" while nothing anywhere checked the auth —
    a wrong or expired key looked exactly like a good one. Presence of an env var is not
    evidence that the credential works, so this authenticates for real.
    """
    if not cfg.llm_api_key:
        r.add(WARN, "net:llm", "skipped: no credential to authenticate with")
        return
    try:
        resp = await client.get(
            f"{cfg.llm_base_url.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {cfg.llm_api_key}"},
        )
    except httpx.HTTPError as exc:
        r.add(WARN, "net:llm", f"unreachable: {type(exc).__name__}")
        return

    if resp.status_code == 200:
        ids = _model_ids(resp)
        has_model = cfg.llm_deep in ids
        r.add(
            OK if has_model or not ids else WARN,
            "net:llm",
            f"HTTP 200, authenticated; {len(ids)} models"
            + ("" if has_model or not ids else f" — {cfg.llm_deep} is NOT among them"),
        )
    elif resp.status_code in (401, 403):
        r.add(
            FAIL,
            "net:llm",
            f"HTTP {resp.status_code} WITH a credential — the key is rejected as invalid",
        )
    else:
        r.add(WARN, "net:llm", f"HTTP {resp.status_code}")


def _model_ids(resp: httpx.Response) -> list[str]:
    try:
        data = resp.json().get("data") or []
    except Exception:
        return []
    return [str(m.get("id")) for m in data if isinstance(m, dict) and m.get("id")]


async def _report_candidate_chat_ids(
    r: Report, client: httpx.AsyncClient, token: str
) -> list[tuple[str, str]]:
    """Print the chat ids this bot can see, so TELEGRAM_CHAT_ID can be pasted in.

    Returns them as well as logging them, because the most common failure is not a missing id
    but a *mangled* one: group and channel ids are negative, and dropping the sign silently
    turns a group into a non-existent user chat. The caller then names that directly rather
    than leaving the operator to compare two similar numbers by eye.
    """
    try:
        resp = await client.get(f"https://api.telegram.org/bot{token}/getUpdates")
        body = resp.json()
    except Exception as exc:
        r.add(WARN, "telegram chat id", f"could not fetch updates: {type(exc).__name__}")
        return []

    if not body.get("ok"):
        r.add(WARN, "telegram chat id", f"getUpdates failed: {body.get('description')}")
        return []

    candidates = extract_chat_ids(list(body.get("result") or []))
    if not candidates:
        r.add(
            WARN,
            "telegram chat id",
            "no chats visible. Open your bot and send it /start (a bot cannot message you "
            "first), then re-run doctor.",
        )
        return []
    for chat_id, label in candidates:
        r.add(OK, "telegram chat id", f"TELEGRAM_CHAT_ID={chat_id}  [{label}]")
    return candidates


def _arxiv_probe_url(cfg: Settings) -> str:
    """A *valid* arXiv query, shaped like the one the pipeline sends.

    arXiv returns 400 for `/api/query?max_results=1` with no `search_query`, so a probe that
    omits it reports the probe's own defect as a network failure. Using a real category also
    means doctor verifies the query form the app depends on, not just that DNS resolves.
    """
    category = cfg.profile.categories[0] if cfg.profile.categories else "cs.AI"
    return f"https://export.arxiv.org/api/query?search_query=cat:{category}&max_results=1"


def _weight_maps_off_by(cfg: Settings) -> dict[str, float]:
    """Weight maps that do not sum to 1.0.

    Checked individually: the rubric and triage maps are separate scales, so summing them
    together yields 2.0 and says nothing. This check exists because the design shipped a
    triage map of .27/.27/.27/.20 that summed to 1.01.
    """
    from screener.domain.types import RUBRIC_WEIGHTS, TRIAGE_WEIGHTS

    maps: dict[str, dict[str, float]] = {
        "rubric": {k: float(v) for k, v in RUBRIC_WEIGHTS.items()},
        "triage": {k: float(v) for k, v in TRIAGE_WEIGHTS.items()},
    }
    return {
        name: round(sum(m.values()), 6)
        for name, m in maps.items()
        if abs(sum(m.values()) - 1.0) > 1e-9
    }


def _python_version() -> str:
    import sys

    v = sys.version_info
    tag = f"{v.major}.{v.minor}.{v.micro}"
    return f"{tag} (>=3.12 required for PEP 695 generics)" if v >= (3, 12) else f"{tag} TOO OLD"


def _httpx_client_error() -> str | None:
    """Report whether httpx can build a client at all in this environment.

    Probes the symptom rather than re-implementing httpx's NO_PROXY grammar. httpx parses
    every NO_PROXY entry while *constructing* a client, so a single unsupported entry — a
    bracketed IPv6 literal such as "[::1]" where "::1" was meant — makes every HTTP call in
    the process impossible, and the traceback points at httpx internals rather than at the
    environment that caused it. Diagnosing that is doctor's job.
    """
    try:
        client = httpx.Client()
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    client.close()
    return None


async def _check_http(r: Report, client: httpx.AsyncClient, name: str, url: str) -> None:
    """Probe one endpoint, and be precise about what each status means.

    A blanket "anything below 500 is fine" hides the difference between *the server refused
    us* (401 — reachable, credentials are checked separately) and *we sent a bad request*
    (400 — a defect on our side). The latter shipped once already, disguised as an OK line,
    because arXiv 400s on a query with no `search_query`.
    """
    try:
        resp = await client.get(url)
    except httpx.HTTPError as exc:
        r.add(WARN, f"net:{name}", f"unreachable: {type(exc).__name__}")
        return
    except Exception as exc:
        r.add(WARN, f"net:{name}", f"{type(exc).__name__}: {exc}")
        return

    code = resp.status_code
    if code < 300:
        r.add(OK, f"net:{name}", f"HTTP {code}")
    elif code in (401, 403):
        # Reachable; whether the credential is good is a separate, explicit check.
        r.add(OK, f"net:{name}", f"HTTP {code} (reachable; auth checked separately)")
    elif code == 429:
        r.add(WARN, f"net:{name}", "HTTP 429 (rate limited, but reachable)")
    elif 400 <= code < 500:
        r.add(FAIL, f"net:{name}", f"HTTP {code} — request rejected, so the request is wrong")
    else:
        r.add(WARN, f"net:{name}", f"HTTP {code} (server-side error)")
