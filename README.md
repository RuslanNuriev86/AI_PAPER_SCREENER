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
| scanned | **3,679** papers (10 arXiv requests, 33 s cold / 0.07 s cached) |
| gate-passing | **196–530** depending on the day (≈5–20%) |
| reviewed | **16** (capped by `review_top_k`) |
| delivered | **4 picks in 2 messages**, $0.016 |
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

### Getting `TELEGRAM_CHAT_ID`

It is not something you look up — it is whatever Telegram reports once the bot has received
something, so the order matters:

1. Message [@BotFather](https://t.me/BotFather) → `/newbot` → copy the token into
   `TELEGRAM_BOT_TOKEN`.
2. **Open your bot and send it `/start`.** A bot cannot initiate a conversation, so until you
   do this `getUpdates` is empty and there is nothing to read. (For a group or channel, add the
   bot instead — its id appears on join via `my_chat_member`, before anyone types anything.)
3. Run `make doctor`. With the token set but no chat id, it prints every chat the bot can see:

```
[  ok  ] telegram chat id       TELEGRAM_CHAT_ID=88776655  [private (You)]
[  ok  ] telegram chat id       TELEGRAM_CHAT_ID=-1001234567890  [supergroup (Agents Digest)]
```

Paste the one you want into `.env`. **Positive ids are DMs** (what §16 decision 2 assumes);
group and channel ids are negative. `doctor` calls `getUpdates` without an `offset`, so this is
non-destructive and will not consume updates that `screener feedback` needs.

`doctor` also **verifies** the id with `getChat` rather than trusting that the variable is
non-empty, because a wrong id fails the same way a good one passes every local check and only
surfaces as a bare `400 Bad Request` on the first send. Two failures it names explicitly:

```
[ fail ] telegram chat       400 Bad Request: chat not found — TELEGRAM_CHAT_ID=5402616139 …
[  ok  ] telegram chat id    TELEGRAM_CHAT_ID=-5402616139  [group (ai_papers)]
[ fail ] telegram chat id    looks like a group id with the sign dropped: set TELEGRAM_CHAT_ID=-5402616139
```

**The sign matters.** Group and channel ids are negative; dropping the minus turns a group into
a non-existent user chat, and that is the single most common cause of this error.

## Commands

| Command | What it does |
|---|---|
| `screener run --mode daily` | one run: fetch → gate → score → compose → send |
| `screener dry-run` | same pipeline, no send, no outbox write |
| `screener doctor` | validate config, credentials, DB, network, clock, proxy env |
| `screener revisit` | measure due outcome rungs (separate job, off the delivery path) |
| `screener replay --date` | rebuild a digest from stored data (no network, no send) |
| `screener replay --date --write-outbox` | rebuild a digest and park it for the next run to send |
| `screener rearm --date` | make a day's reviewed papers fresh again (recovery, see below) |
| `screener web` | browse what was found, delivered and rated at http://127.0.0.1:8765 |
| `screener explain <id>` | print exactly why one paper scored what it scored |
| `screener backtest --since` | re-score stored rankings under candidate weights |
| `screener feedback` | poll Telegram for replies (text only — see below) |
| `screener eval` / `prune` / `stats` | maturity coverage, retention, row counts |

## Browsing what happened

```bash
screener web                 # http://127.0.0.1:8765
screener web --port 9000
```

Read-only and localhost-only: it opens the database with `mode=ro`, so it can never write, and it
refuses a non-loopback bind unless you pass `--allow-remote` (there is no authentication, so a
public bind publishes your whole paper history). To reach it from elsewhere, tunnel:

```bash
ssh -L 8765:127.0.0.1:8765 your-host
```

Pages: overall statistics and per-run funnels; digests by day with each item's review, rating
basis and reactions; searchable papers over any period; top-rated by the screener; top-rated *by
readers*; and a per-paper page showing the full rating arithmetic and measured signals.

**Reader reactions need the bot to be an administrator** of the group — Telegram does not deliver
`message_reaction` updates to a plain member. The subscription is already enabled, so promoting
`PaperScreener26_bot` in *ai_papers* (Group settings → Administrators) is the only step needed;
written replies are captured either way. `screener feedback` polls every 15 minutes under its
systemd timer.

## When a digest is reviewed but never delivered

A failed send parks the rendered digest in `outbox/`, and the **next run retries it
automatically** before doing anything else — no human in the loop. Recovery works from either
outbox artifact, so losing the `.json` sidecar does not lose the digest.

If the outbox entry itself is gone, the papers are still in the seen-set and no future run will
rebuild them. `rearm` is the way back:

```bash
screener rearm --date 2026-09-30   # clears only that day's *reviewed* papers
screener run                       # re-reviews them (~$0.02) and sends
```

It clears the seen-set rows for the shortlisted papers only — the other few thousand stay seen,
so recovery costs one re-review rather than a re-fetch of a day. It also clears *undelivered*
delivery rows but never a successful send, so at-most-once delivery still holds: `rearm` cannot
make the reader receive something twice.

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
