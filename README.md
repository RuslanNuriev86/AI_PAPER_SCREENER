# Agent Papers Daily

Finds the most promising new arXiv papers on AI agents, writes short faithful summaries with
an explicit *why this matters*, and delivers a daily digest to Telegram.

**Status: v0 — walking skeleton, working.** `make check` green (ruff, `mypy --strict`,
89 tests), verified against live arXiv. Full design in [`docs/DESIGN.md`](docs/DESIGN.md);
§15 defines what v0 is and is not.

## What v0 does

```
fetch → gate → single LLM call per shortlisted paper → compose → deliver
```

Plus the two things v0 cannot defer without losing data forever:

- the **seen-set**, keyed `(arxiv_id, version)`, so a paper is gated once and never
  re-reviewed for the whole 5-day lookback window;
- **watchlist enrolment and the T+14 rung**, because outcome labels cannot be collected
  retroactively. A paper enrolled today yields its T+180 measurement in six months.

Deliberately *not* in v0: the triage/review cascade, the faithfulness verifier, enrichment,
the full ladder, calibration reports, bounded resurfacing. See §15.

## Measured behaviour

One run against live arXiv (5-day window, 8 categories):

| Stage | Result |
|---|---|
| scanned | **2,635** papers |
| gate-passing | **530** (20.1%) |
| reviewed | **16** (capped by `review_top_k`) |
| second run | **0 fresh** — the seen-set makes the window idempotent |

The design doc's original estimate was ~250 scanned / ~50 gated, so the funnel was ~10×
larger than assumed. §13.1 in the design records the correction and its cost consequences.

## LLM provider

**DeepSeek, model `deepseek-flash`** ([docs](https://api-docs.deepseek.com/quick_start/pricing)),
for both tiers at v0. The adapter speaks OpenAI-shaped chat-completions, which DeepSeek serves
at `https://api.deepseek.com`; the Anthropic-shaped endpoint would be
`https://api.deepseek.com/anthropic`.

```bash
DEEPSEEK_API_KEY=sk-...          # or LLM_API_KEY / OPENAI_API_KEY
SCREENER_LLM_DEEP=deepseek-flash
SCREENER_LLM_THINKING=false      # leave false: see below
```

**Leave thinking mode off.** DeepSeek enables it by default, and thinking mode *silently
ignores `temperature`* — no error, no effect. The rubric's justification is that a model is
repeatable and a human can audit it, so scoring runs with `{"thinking": {"type": "disabled"}}`
and a unit test asserts the field is present. `screener doctor` reports which mode is active.

Switching providers is a config change, not a code change: `DEEPSEEK_BASE_URL` plus the two
model ids. `deepseek-v4-pro` is priced in the ledger for a v1 split of triage vs review.

## Setup

Python 3.12 is required (PEP 695 generics in the `LLM` port). **Use `make`, not bare `uv`**:
this host mounts uv's default cache and Python directory read-only, so `uv.toml` redirects the
cache and the `Makefile` exports `UV_PYTHON_INSTALL_DIR` (which `uv.toml` cannot express —
uv 0.8.16 rejects it as an unknown field). See DESIGN.md §16.

```bash
make install          # create .venv on Python 3.12 and sync dependencies
cp .env.example .env  # then fill in tokens
make doctor           # validate config, DB, network, credentials
make dry              # full pipeline, no Telegram send
```

## Commands

| Command | What it does |
|---|---|
| `screener run --mode daily` | one run: fetch → gate → score → compose → send |
| `screener dry-run` | same pipeline, no send, no outbox write |
| `screener doctor` | validate config, credentials, DB, network, clock, proxy env |
| `screener revisit` | measure due outcome rungs (separate job, off the delivery path) |
| `screener replay --date` | rebuild a digest from stored data (no network, no send) |
| `screener backtest --since` | re-score stored rankings under candidate weights |
| `screener feedback` | poll Telegram for replies (text only — see below) |
| `screener eval` / `prune` / `stats` | maturity coverage, retention, row counts |

## Two host gotchas worth knowing

**`NO_PROXY` with a bracketed IPv6 literal breaks every HTTP call.** If `NO_PROXY` contains
`[::1]`, httpx raises `InvalidURL` while merely *constructing* a client, so nothing network
works and the traceback points at httpx internals. Use `::1`, not `[::1]`. `screener doctor`
now detects and reports this explicitly.

**Reactions do not work in a private chat.** Telegram's `message_reaction` update requires the
bot to be an *administrator*, and admin is a group/channel-only concept. The digest therefore
uses text replies (`Reply 👍 / 👎 / 🔥`), which the existing reply path already captures — no
admin rights, no second service, no Bot API surprises.

## Tests

```bash
make check    # ruff + mypy --strict + pytest
```

The pure domain logic (gate, seen-set semantics, compose chunking, scoring, maturity rungs)
is tested without network or a database. Two invariants are asserted structurally rather than
by convention: `score()` can never read an outcome (temporal leakage), and every weight map
sums to exactly 1.0.
