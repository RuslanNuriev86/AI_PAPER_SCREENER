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
        "DEEPSEEK_API_KEY set"
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
    else:
        r.add(FAIL, "telegram config", "TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID required")
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
        await _check_http(r, client, "arxiv", "https://export.arxiv.org/api/query?max_results=1")
        await _check_http(r, client, "llm", f"{cfg.llm_base_url.rstrip('/')}/models")
        await _check_http(r, client, "github", "https://api.github.com/rate_limit")
        if cfg.telegram_bot_token:
            await _check_http(
                r,
                client,
                "telegram",
                f"https://api.telegram.org/bot{cfg.telegram_bot_token}/getMe",
            )
    return r


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
    try:
        resp = await client.get(url)
        # 401/403 still proves reachability; only transport failures are interesting here.
        if resp.status_code < 500:
            r.add(OK, f"net:{name}", f"HTTP {resp.status_code}")
        else:
            r.add(WARN, f"net:{name}", f"HTTP {resp.status_code}")
    except httpx.HTTPError as exc:
        r.add(WARN, f"net:{name}", f"unreachable: {type(exc).__name__}")
    except Exception as exc:
        r.add(WARN, f"net:{name}", f"{type(exc).__name__}: {exc}")
