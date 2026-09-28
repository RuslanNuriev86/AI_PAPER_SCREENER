# Agent Papers Daily — System Design

A service that finds the most promising arXiv papers on AI agents, writes short,
faithful, high-signal summaries with an explicit "why this matters" rationale, and
delivers a daily digest to Telegram.

**Status:** design only — no implementation yet.
**Target:** Python 3.12, `uv`-managed, run as a scheduled CLI job.

---

## 1. Problem definition

### 1.1 What the hard part actually is

The naive version of this product — "fetch new arXiv papers, ask an LLM to summarize
them, post to Telegram" — fails for four specific reasons. The design exists to solve
those four, not to wrap an API.

| Failure mode | Why it happens | Design response |
|---|---|---|
| **Noise: "agent" is a wildly overloaded word** | cs.AI/cs.LG daily output is full of RL agents, agent-based simulation, economic/market agents, biological agents, and control-theory multi-agent systems. A keyword match on "agent" is ~70% wrong. | Two-layer relevance: cheap deterministic gate + LLM classification with explicit negative examples (§5). |
| **"Promising" is invisible on day 0** | Citation counts, the usual proxy for importance, are 0 for a paper announced yesterday. Ranking by citations is structurally useless here. | Rank on *predictable* features: novelty vs. cited prior art, evidence quality, lab/author track record, code release, early community signal — plus an explicit calibrated impact forecast that is later back-tested (§6.5, §13.4). |
| **LLM summaries are generic and unfalsifiable** | Without a contract, models emit "paves the way for future work" filler, and invent numbers that were never in the abstract. | Rubric-bound prompts, "why it matters" restricted to four named lenses, banned-phrase lint, and a separate faithfulness checker that verifies every claim against the source text (§8). |
| **Daily digest quietly degrades** | A scheduler that stops running, a Telegram token that expires, or an LLM outage produces silence nobody notices. | Idempotent runs, structured run records, heartbeat monitoring, and degraded-but-delivered fallback instead of nothing (§12). |

### 1.2 Success criteria

- **Precision over recall.** 3–6 papers/day, and *zero* is an acceptable and better
  outcome than padding the digest. Target ≥ 70% of delivered papers rated
  useful/relevant on reaction feedback.
- **Every item is self-contained.** Reading the digest alone tells you what was done,
  why it matters, and what is weak about it — without opening the PDF.
- **Reproducible and tunable.** Selection is deterministic given stored assessments;
  ranking weights live in YAML, not in prompts or code.
- **Cost predictable and capped.** Hard per-run USD ceiling; a run that would exceed
  it degrades rather than overspends.
- **Unattended.** Runs daily with no human action; failures are loud, not silent.

### 1.3 Non-goals (v1)

- No full literature review / citation-graph analysis.
- No PDF parsing for v1 (abstract + metadata only; full-text tier is v1.5).
- No web UI, no multi-user tenancy, no per-user personalization.
- No reading of arXiv full text into a public artifact — we summarize and link.

---

## 2. Design principles

1. **LLMs produce features; Python makes decisions.** The model returns *sub-scores and
   prose*; a pure function combines them into a rank and a decision. This keeps ranking
   auditable, testable, and tunable without prompt surgery, and it means a model
   upgrade or a bad generation cannot silently change the selection policy.
2. **Cheap cascade, expensive core.** Deterministic filters → cheap model over many
   candidates → strong model over few finalists. Cost scales with candidates × tiers,
   so the funnel order is the cost model.
3. **Grounding over eloquence.** Any numeric or factual claim in the output must be
   traceable to the source text; a verifier enforces it.
4. **Idempotent by construction.** Re-running any day is safe. Delivery is keyed by
   `(run_date, arxiv_id)` with a "sent at most once ever" guarantee (§10).
5. **Ports and adapters, no framework.** Each external system (arXiv, OpenAlex/HF,
   LLM, Telegram) sits behind a narrow `Protocol`. The pipeline is testable with fakes
   and has no import of any vendor SDK.
6. **Laconic but boring on purpose.** Standard library + a handful of small
   dependencies; explicit SQL over an ORM; one entrypoint; no async framework beyond
   `asyncio` + `httpx`.

---

## 3. Architecture

Modular monolith, one deployable, invoked as a scheduled CLI. Pipeline stages are
async functions over typed models; every I/O boundary is an adapter injected at the
composition root.

```
                    ┌────────────────────────── screener run (CLI, cron/launchd) ─────────────────────────┐
                    │                                                                                      │
  arXiv API ───────▶│ 1 fetch ─▶ 2 gate ─▶ 3 enrich ─▶ 4 triage ─▶ 5 review ─▶ 6 rank ─▶ 7 compose ─▶ 8 send │───▶ Telegram
  (Atom/XML)        │           (pure)   (S2/OpenAlex/  (cheap LLM,  (strong LLM,  (pure,     (pure,      │
                    │                     HF/GitHub)     batched)     bounded)     gates)     chunked)    │
                    │                                                                                      │
                    │        ┌─────────────── SQLite: papers · assessments · runs · deliveries · feedback ──┴──┐
                    └────────┤  seen-set (at-most-once delivery) · score history · cost ledger · backtest data  │
                             └────────────────────────────────────────────────────────────────────────────────┘
```

### Layers

```
src/screener/
  domain/        # pure: models, gate rules, scoring, digest composition. No I/O, no vendor imports.
    models.py         Paper, Enrichment, Assessment, Ranking, Run, Digest
    relevance.py      deterministic agentic gate + topic classification
    scoring.py        Weights, score(), select() with MMR-style diversification
    compose.py        Ranking -> Telegram-safe HTML chunks
    style.py          banned-phrase lint, summary-collision detection
  ports.py       # Protocols: PaperSource, Enricher, LLM, Notifier, Repository, Clock
  adapters/
    arxiv.py          Atom API client (rate-limited, paginated, retrying)
    enrich.py         Semantic Scholar / OpenAlex / HF Daily / GitHub adapters
    llm_openai.py     structured-output client (one of N interchangeable providers)
    llm_anthropic.py
    telegram.py       sendMessage / chunking / reactions polling
    sqlite_repo.py    stdlib sqlite3 repository, migrations from .sql files
  pipeline/
    fetch.py  gate.py  enrich.py  triage.py  review.py  rank.py  compose.py  deliver.py
    run.py            # orchestrator: builds deps, runs stages, records the Run
  prompts/
    triage.v1.md  review.v2.md  verify.v1.md     # versioned prompt files, not string literals
  config.py       # pydantic-settings: env + YAML profiles
  cli.py          # typer app: run | dry-run | replay | feedback | backtest | doctor
  migrations/*.sql
tests/
  fixtures/       # recorded Atom/JSON responses, golden digest output
  eval/           # ~30 hand-labeled papers for offline precision/recall
```

**Why a monolith:** one process, one schedule, one store. Every boundary that could
later become a service is already a `Protocol`, so splitting is mechanical if it is
ever needed — and it will not be.

**Dependency budget:** `httpx`, `pydantic`, `pydantic-settings`, `typer`, `tenacity`,
`feedparser`, `structlog`, `PyYAML` + one provider SDK. Dev: `pytest`, `ruff`,
`mypy --strict`, `respx`. Nothing else without a written reason.

---

## 4. Pipeline stages

| # | Stage | Input → Output | Cost | Typical time | Failure behaviour |
|---|---|---|---|---|---|
| 1 | **fetch** | window → `list[Paper]` | free | 5–20 s (3 s/request, ~8 queries) | retry ×3 w/ backoff; abort run if zero papers *and* source errored |
| 2 | **gate** | `Paper` → `Paper \| None` | free | <1 s | pure function, always succeeds |
| 3 | **enrich** | `Paper` → `Enrichment` | free (polite pools) | 10–30 s | best-effort: missing enrichment only lowers `pedigree`/`buzz` |
| 4 | **triage** | 40–80 papers → `Triage` scores | ~$0.02 | 20–40 s | batch failures retried, then fall back to gate-only ordering |
| 5 | **review** | top 12–20 → `Assessment` | $0.20–$1.50 | 60–180 s | per-paper failure drops that paper; ≥3 failures ⇒ degraded run |
| 6 | **rank** | `Assessment[]` → `Ranking[]` | free | <1 s | pure, deterministic |
| 7 | **compose** | `Ranking[]` → HTML chunks | free | <1 s | pure; asserts no chunk > 4096 chars |
| 8 | **deliver** | chunks → message ids | free | 2–5 s | retry; on final failure the digest is written to `outbox/` and a heartbeat alert fires |

Stage boundaries are also persistence boundaries: `papers`, `assessments` and
`deliveries` are written so a failed run can be resumed and every digest is
reconstructable (`screener replay --date`).

---

## 5. Interest model: deciding what "AI agents" means

The single most important quality lever. Relevance is **two layers**, cheap first.

**Layer 1 — deterministic gate (`domain/relevance.py`).** Rules over title, abstract
and categories, evaluated in order:

1. **Category prefilter** (necessary, not sufficient): `cs.AI, cs.CL, cs.MA, cs.LG,
   cs.SE, cs.CR, cs.HC, cs.RO` (RO only for embodied/computer-use agents).
2. **Positive signal:** ≥1 strong term (`LLM agent`, `agentic`, `tool use`,
   `function calling`, `computer use`, `agent memory`, `multi-agent LLM`,
   `agent framework`, `agent benchmark`, `trajectory`, `agentic RL`, `MCP`,
   `skill library`, `web agent`, `SWE agent`) — matched on word boundaries, with
   weighted scoring, not a bare `in` check.
3. **Negative signal / exclusions:** `agent-based model`, `multi-agent system` *when
   paired with* power grid / traffic / epidemic / market simulation, `RL agent` for
   locomotion/manipulation without a language model, `biological agent`,
   `pharmacological agent`, `economic agent`, `intelligent agent` in the classical
   MAS/BDI sense, `nanoparticle agent`. Exclusions are *subtractive*, not absolute —
   a paper with two strong agentic terms survives one exclusion term (this asymmetry
   is deliberate: better to triage a few extra papers than to drop a good one).
4. **Hard rejects:** withdrawn, no abstract, <120 words abstract, `replaced` with
   only cosmetic changes and already delivered.

Output is a `RelevanceHint {score: float, matched: list[str], excluded: list[str]}`
that travels with the paper and is fed to the LLM as evidence, not as a decision.

**Layer 2 — LLM classification** in triage (§6.3): the model sees the same text plus
the hint and must output `agentic: bool` with a one-line justification and explicit
`not_agentic` red flag. Ambiguous cases get resolved by the model, not by ever-growing
regexes.

**Profile config** (`config/profile.yaml`) holds all of the above as data:

```yaml
name: "LLM agents & agentic systems"
categories: [cs.AI, cs.CL, cs.MA, cs.LG, cs.SE, cs.CR, cs.HC]
lookback_days: 5          # > moderation lag (1-4 days); dedupe handles the overlap
strong_terms: ["agentic", "LLM agent", "tool use", "function calling", "computer use",
               "web agent", "SWE agent", "agent memory", "multi-agent LLM", "agent benchmark"]
weak_terms:   ["agent", "trajectory", "planning", "reflection", "MCP", "orchestration"]
exclude_patterns:
  - "agent-based model"
  - "(power grid|traffic|epidemic|market|supply chain).{0,40}multi-agent"
  - "\\b(nanoparticle|pharmacological|biological|chemical) agent"
boost_topics: ["evaluation & benchmarks", "multi-agent coordination", "memory & context",
               "computer use", "safety & oversight", "agentic RL", "infrastructure/protocols"]
per_topic_cap: 2
```

Adding a subfield is a YAML edit. Subfield coverage is enforced later by `per_topic_cap`
in selection (§6.4) so the digest does not become five memory papers.

---

## 6. Assessment and selection

### 6.1 Rubric — the definition of "most proficient and promising"

Seven scored dimensions, each 0–10, defined tightly enough that a model is repeatable
and a human can audit it. Scores are LLM-produced *features*; weights are ours.

| Dimension | Default weight | Definition (what earns a 9) |
|---|---|---|
| `relevance` | 0.20 | Directly advances LLM-agent capability/reliability/evaluation *and* matches the interest profile. 9 = squarely in a boosted topic. |
| `novelty` | 0.20 | 9 = a new mechanism, formulation, or measurement the field did not have; 3 = a recombination of known techniques. Penalize "we apply X to Y" with no new insight. |
| `rigor` | 0.15 | 9 = multiple strong baselines, ablations, error bars/seeds, honest limitations, released artifacts. 3 = single baseline or self-reported only. |
| `evidence_strength` | 0.10 | Magnitude *and* credibility of the demonstrated result (quoted numbers, eval suite size, held-out conditions, human eval). |
| `impact_forecast` | 0.20 | Calibrated expectation this becomes a standard reference/framework component within 12 months. Anchored by written anchors (see below), not vibes. |
| `reproducibility` | 0.05 | Code, data, artifacts released and plausibly runnable; protocol/benchmark released. |
| `pedigree` | 0.05 | Venue acceptance (from the arXiv `comment` field), strong lab/author track record (h-index, prior influential work) as *capped* signals. |
| `early_signal` | 0.05 | HF Daily Papers upvotes, GitHub stars, notable adjacent-lab uptake. Capped at 10 and never decisive — it is the most gameable signal. |

**Impact forecast anchors** (stored in the prompt, so the scale is stable across runs):

- 9–10 = likely to be a named baseline/component others build on within a year;
- 7–8 = likely to be widely cited and replicated;
- 5–6 = solid contribution, respectable citations;
- 3–4 = incremental, niche citations;
- 0–2 = superseded quickly or not reproducible.

### 6.2 Hard gates (applied before scoring)

`not_agentic` · `pure_survey` (surveys are allowed only in a dedicated weekly recap) ·
`no_technical_contribution` (position papers, editorials) · `marketing/whitepaper` ·
`withdrawn`. A gated paper is recorded with a reason and never delivered. Gates are
*documented and visible* in the digest's scan footer — a good paper we filtered by
mistake is discoverable, not invisible.

### 6.3 Stage 4 — triage (cheap model, all candidates)

One batched call per ~10 papers. For each: `agentic`, `relevance`, `novelty`, `impact`
(coarse 0–10), `is_survey`, `red_flags`, 15-word reason. Purpose is ordering, not
judgment. Output is validated against a Pydantic schema; failures retry once with the
validation error appended, then the paper falls back to gate-only ordering.

### 6.4 Stage 5–6 — review, then deterministic selection

The strong model reviews the top 12–20 (§7). Selection is then a pure function:

```python
def select(ranked: list[Ranking], cfg: Selection) -> list[Ranking]:
    kept = [r for r in ranked if r.score >= cfg.min_score and not r.gated]
    kept.sort(key=lambda r: r.score, reverse=True)
    picks: list[Ranking] = []
    for r in kept:                                  # greedy, diversified
        if len(picks) == cfg.max_papers: break
        if _over(cfg.per_topic_cap, r.topics, picks): continue
        if _over(cfg.per_lab_cap, r.lab, picks): continue
        if _too_similar(r, picks, cfg.max_tag_jaccard): continue
        picks.append(r)
    return picks
```

Diversification (a light MMR over topic tags) is what keeps the digest *useful*: five
variants of the same RAG-improvement idea is a worse product than one of them plus
three unrelated advances. A "headline" pick (≥8.5) is exempt from the topic cap.

### 6.5 The ranker

```python
composite = Σ weight_i * dim_i  −  0.5 * len(soft_red_flags)  −  1.0 * len(hard_red_flags)
```

`score()` also emits the full component breakdown, stored with the ranking. That
breakdown is the audit trail: any digest item can be explained months later, and
weight changes can be replayed offline against history (`screener backtest`).

---

## 7. Summarization: making the output actually good

Every reviewed paper yields a strict, length-bounded record (validated by Pydantic):

```python
class Review(BaseModel):
    tldr: str            = Field(max_length=220)   # one sentence, no preamble
    what_they_did: str   = Field(max_length=420)   # mechanism, not motivation
    why_it_matters: str  = Field(max_length=420)   # restricted to named lenses
    caveats: str         = Field(max_length=240)   # the honest weakness
    lenses: list[Literal["capability", "method", "safety", "adoption"]]
    tags: list[str]
    evidence_quotes: list[str]                     # verbatim spans from the source
    scores: Scores                                 # the rubric dimensions
    red_flags: list[Literal[...]]
    model: str; prompt_version: str; cost_usd: float
```

**The four lenses** are the mechanism that makes "why is this paper important and
perspective" concrete instead of flattering. The model must pick the 1–2 that apply
and write to them:

- **capability** — what agents can now do that they could not before;
- **method** — what the field can now measure, build, or compare that it could not;
- **safety** — what risk/oversight implication follows, if any;
- **adoption** — what will plausibly show up in frameworks/products within a year.

If none applies, the honest output is an empty `lenses` list and a low `impact_forecast`
— which is exactly the outcome that should lose to a stronger paper.

**Anti-slop, deterministically enforced (`domain/style.py`):**

- **Banned phrases** ⇒ regenerate once, then downgrade with a log warning:
  "paves the way", "significant contribution", "this paper is important",
  "opens new avenues", "state-of-the-art results" (unless a quoted number follows),
  "revolutionary", "groundbreaking", "delve", "in today's rapidly evolving".
- **Summary collision:** if two items' `tldr`+`what_they_did` have high token-Jaccard
  similarity, both are regenerated with a "be specific; other items cover X" hint.
  This is the main defence against a digest that reads as five interchangeable paragraphs.
- **Numeric claims** must appear in `evidence_quotes` (exact substring match against the
  source). Unexplained numbers are stripped or flagged.

**Faithfulness verifier** (cheap model, one call per delivered paper): given abstract +
generated text, return `{supported: bool, unsupported_claims: [...]}`. Any unsupported
claim triggers a rewrite of that field; a second failure drops the paper and logs it.
This turns "the summary is clear" from a hope into a checked property.

**Full-text tier (v1.5, opt-in):** for the top ~8 papers, fetch `arxiv.org/html/{id}v1`
(the new arXiv HTML for post-2023 papers; `ar5iv` as fallback), extract
methods/experiments sections, truncate to ~15k tokens, and re-run the review with the
system prompt cached. Rationale: abstracts are self-serving; rigour and
reproducibility judgments get much sharper against the actual experimental section.
Cost roughly 3–5× on the reviewed subset, which is why it is a tier and not the default.

---

## 8. Digest format and Telegram delivery

### 8.1 Rendered example

```
🧠 <b>Agent Papers — Daily</b>
Tue 2025-09-30 · 5 picks · 214 new papers scanned · 38 agent-relevant · $0.41

<b>1. AgentRM: Process Reward Models for Long-Horizon Agent Trajectories</b>  🔥 9.1
<code>2509.18422</code> · cs.LG cs.AI · accepted NeurIPS'25
<a href="https://arxiv.org/abs/2509.18422">abs</a> · <a href="...">pdf</a> · <a href="...">code</a>

<b>TL;DR</b> Trains a reward model on step-level agent trajectories and uses it to
rerank tool-use plans, gaining 18 points over outcome-only RL on WebArena.

<b>What they did</b> They collect 40k step-labelled trajectories from four agent
scaffolds and train a process reward model that scores partial plans; the model is
then used both for reranking and as a dense RL signal.

<b>Why it matters</b> <i>[method]</i> Step-level supervision was the missing piece
for reliably comparing agent scaffolds; a reusable PRM turns trajectory quality into
a measurable, trainable quantity instead of a benchmark score.

<b>Weak spot</b> Gains are shown on three web benchmarks only; no evidence it
transfers to code or embodied agents, and the labelling procedure needs a strong
teacher model.

<b>Signals</b> code ✅ · 3rd paper from this group on agent rewards · HF 61 👍
```

Then a footer:

```
<i>Skipped: 6 surveys · 3 position papers · 4 below threshold (best 6.1)</i>
<i>Reply 👍 / 👎 or react to tune tomorrow's ranking.</i>
```

Design rules: each item ≤ ~900 characters; ≤ 5 items per message; **split on item
boundaries**, never mid-item; Telegram's 4096-character limit is asserted in code, not
hoped for (§9). Links are always explicit `abs`/`pdf`/`code` — no auto-previews
(`disable_web_page_preview=True`) so the digest stays scannable.

### 8.2 Message strategy

- `parse_mode=HTML` (more forgiving than MarkdownV2 escaping rules).
- Multi-message digests: item 1 shipped as a standalone "headline" message, the rest
  as one or two follow-ups — so a notification preview already shows the top pick.
- Optional weekly recap (Sunday) as an editable single message.

### 8.3 Feedback loop (what makes it improve)

Reactions and replies are captured via `getUpdates` with
`allowed_updates=["message_reaction", "message"]` on a `screener feedback` command
(runs 5 minutes after delivery, plus a daily sweep). Signals:

- 👍 / 👎 / 🔥 reaction → `feedback` row against `(arxiv_id, run)`.
- Free-text reply → stored verbatim; a weekly job extracts preference statements
  ("more theory", "less benchmark-only work") into `profile.notes`, which are injected
  into the review prompt as soft guidance. Human language becomes policy without
  editing weights by hand.
- **Weekly backtest job:** for picks from 1/3/6/12 months ago, refresh citation counts,
  HF upvotes and GitHub stars, then compute Spearman correlation between predicted
  `impact_forecast` and observed signal. Report drift in the run log. This is the only
  honest way to know whether "promising" means anything, and it is a design requirement,
  not a nice-to-have.

---

## 9. Telegram adapter details

- Send: `POST /bot{token}/sendMessage` with `chat_id`, `text`, `parse_mode=HTML`,
  `link_preview_options.is_disabled=true`, retry on 429 honouring `retry_after`.
- Reactions polling: `POST /bot{token}/getUpdates` with a stored `offset` so reactions
  are consumed exactly once.
- Idempotency: message ids are persisted with `(run_date, arxiv_id)`, so a re-run after
  a crash cannot double-post.
- Failure: after retries, the rendered digest is written to `outbox/{date}.html` and a
  heartbeat alert fires; the next run notices the unsent outbox and offers it first.

---

## 10. Data model (SQLite)

Chosen for zero-ops persistence and exact reproducibility. Written with stdlib
`sqlite3` (WAL mode) and plain `.sql` migrations — no ORM, because the schema is five
tables and every query is clearer as SQL.

```sql
CREATE TABLE papers (
  arxiv_id TEXT PRIMARY KEY,          -- version-less base id: '2509.18422'
  version INTEGER NOT NULL,
  title TEXT NOT NULL, abstract TEXT NOT NULL,
  authors TEXT NOT NULL,              -- JSON array
  categories TEXT NOT NULL,           -- JSON array
  primary_category TEXT NOT NULL,
  submitted_at TEXT NOT NULL,         -- ISO-8601 UTC
  abs_url TEXT NOT NULL, pdf_url TEXT NOT NULL,
  comment TEXT, code_url TEXT,
  agentic_hint REAL,                  -- deterministic gate score
  first_seen_at TEXT NOT NULL
);

CREATE TABLE enrichment (             -- one row per (paper, signal snapshot)
  arxiv_id TEXT NOT NULL REFERENCES papers(arxiv_id),
  source TEXT NOT NULL,               -- 's2' | 'openalex' | 'hf_daily' | 'github'
  payload TEXT NOT NULL, fetched_at TEXT NOT NULL,
  PRIMARY KEY (arxiv_id, source, fetched_at)
);

CREATE TABLE assessments (
  arxiv_id TEXT NOT NULL REFERENCES papers(arxiv_id),
  stage TEXT NOT NULL,                -- 'triage' | 'review' | 'verify'
  prompt_version TEXT NOT NULL, model TEXT NOT NULL,
  created_at TEXT NOT NULL,
  scores TEXT NOT NULL,               -- validated JSON
  text TEXT NOT NULL,                 -- tldr / what_they_did / why_it_matters / caveats
  red_flags TEXT NOT NULL, cost_usd REAL NOT NULL,
  PRIMARY KEY (arxiv_id, stage, prompt_version)
);

CREATE TABLE runs (
  run_id TEXT PRIMARY KEY, started_at TEXT NOT NULL, finished_at TEXT,
  status TEXT NOT NULL,               -- 'ok' | 'degraded' | 'failed'
  stats TEXT NOT NULL, config_hash TEXT NOT NULL, cost_usd REAL NOT NULL DEFAULT 0
);

CREATE TABLE deliveries (
  run_id TEXT NOT NULL REFERENCES runs(run_id),
  arxiv_id TEXT NOT NULL REFERENCES papers(arxiv_id),
  rank INTEGER NOT NULL, score REAL NOT NULL,
  message_id TEXT, sent_at TEXT,
  PRIMARY KEY (run_id, arxiv_id)
);
-- at-most-once ever: a real send blocks any future re-send of the same paper
CREATE UNIQUE INDEX delivered_once ON deliveries(arxiv_id) WHERE message_id IS NOT NULL;

CREATE TABLE feedback (
  message_id TEXT NOT NULL, arxiv_id TEXT NOT NULL,
  kind TEXT NOT NULL,                 -- 'reaction' | 'reply'
  value TEXT NOT NULL, created_at TEXT NOT NULL,
  PRIMARY KEY (message_id, arxiv_id, kind, value)
);
```

Notes: dry runs insert `deliveries` rows with `message_id IS NULL`, so they are
auditable yet do not consume the at-most-once guarantee. Retention: everything is
kept (it is kilobytes/day) except `enrichment.payload` for papers older than 180 days.

---

## 11. Module contracts

The whole system is these five protocols. Every adapter is swappable, every test uses
fakes, and no vendor type ever crosses a boundary.

```python
# ports.py
class PaperSource(Protocol):
    async def fetch(self, since: datetime, until: datetime, profile: Profile) -> list[Paper]: ...

class Enricher(Protocol):
    async def enrich(self, papers: Sequence[Paper]) -> Mapping[str, Enrichment]: ...

class LLM(Protocol):
    async def parse[T: BaseModel](self, *, model: str, prompt: Prompt, payload: str,
                                  schema: type[T], temperature: float = 0.0) -> T: ...

class Notifier(Protocol):
    async def send(self, chunks: Sequence[str]) -> list[str]: ...   # message ids

class Repository(Protocol):
    def unseen(self, ids: Sequence[str]) -> set[str]: ...
    def save_papers(self, papers: Sequence[Paper]) -> None: ...
    def save_assessment(self, a: Assessment) -> None: ...
    def record_run(self, run: Run) -> None: ...
    def record_delivery(self, run_id: str, picks: Sequence[Ranking], ids: Sequence[str]) -> None: ...
```

The orchestrator reads as the pipeline it is — this is the file a new engineer opens
first, and it should be readable in one screen:

```python
# pipeline/run.py
async def execute(cfg: Settings, now: datetime) -> Run:
    async with build_deps(cfg) as deps:                       # httpx clients, repo, ledger
        async with deps.ledger.cap(cfg.budget_usd):
            window = (now - timedelta(days=cfg.profile.lookback_days), now)
            papers = await deps.source.fetch(*window, cfg.profile)
            deps.repo.save_papers(papers)

            fresh = [p for p in papers if p.arxiv_id in deps.repo.unseen([p.arxiv_id for p in papers])]
            gated = [p for p in fresh if (hint := gate(p, cfg.profile))]
            enriched = await deps.enricher.enrich(gated)
            triaged = await triage.run(gated, enriched, deps.llm, cfg)        # cheap, all
            reviewed = await review.run(top_k(triaged, cfg.review_top_k), deps.llm, cfg)  # strong
            verified = await verify.run(reviewed, deps.llm, cfg)              # faithfulness
            ranked = rank.run(verified, cfg)                                  # pure
            picks = select(ranked, cfg.selection)                             # pure
            digest = compose.run(picks, triaged, now, cfg)                    # pure

            ids = [] if cfg.dry_run else await deps.notifier.send(digest.chunks)
            deps.repo.record_delivery(now.date(), picks, ids)
            return Run.of(now, ranked, digest, cost=deps.ledger.spent)
```

Illegal states are unrepresentable: `Assessment` requires scores and text; a `Ranking`
cannot exist without an `Assessment`; `compose.run` cannot emit a chunk longer than the
Telegram limit because a validator asserts it.

---

## 12. Reliability, observability, operations

### 12.1 Failure modes and responses

| Failure | Detection | Response |
|---|---|---|
| arXiv API down / slow | timeout or 5xx on all queries | retry w/ backoff; if still failing, **skip the day with a one-line notice** ("no digest: source unavailable") — silence must not be ambiguous |
| arXiv returns empty window | 0 results after gate | send nothing; log `empty` status. Never fabricate content |
| LLM provider outage | circuit breaker on the parse client | fall back to the secondary provider; else **degraded digest**: metadata + abstract quotes, clearly labelled "auto-summary unavailable" |
| LLM returns invalid JSON | Pydantic validation error | one repair retry with the error appended; second failure drops the paper |
| Cost cap approached | ledger mid-run | stop reviewing further papers and ship what is verified (partial digest beats overspend) |
| Telegram send fails | non-2xx after retries | write `outbox/{date}.html`, ping heartbeat, surface the previous unsent digest next run |
| Scheduler stops entirely | no heartbeat for >26 h | external monitor (healthchecks.io / Uptime Kuma) alerts by email |
| Duplicate delivery | unique partial index rejects | swallowed and logged as `already_delivered` |
| Bad prompt version | offline eval suite drops below threshold | CI gate blocks the prompt change (§13.5) |

### 12.2 Observability

- Structured JSON logs (`structlog`) with `run_id` bound on every line.
- One `runs` row per run: stage counts, per-stage latency, tokens, USD, model ids,
  prompt versions, `config_hash`. This is the primary debugging surface.
- `screener doctor`: validates config, Telegram token + chat id (sends a test message),
  LLM credentials, DB writability, arXiv reachability, disk, and clock/timezone.
- `screener replay --date YYYY-MM-DD`: rebuilds a digest from stored data with current
  config, no network, no sends. Mandatory tool for tuning weights.

### 12.3 Secrets

Env vars / `.env` (git-ignored): `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`,
`OPENAI_API_KEY` / `ANTHROPIC_API_KEY`, optional `SEMANTIC_SCHOLAR_API_KEY`,
`CONTACT_EMAIL` (arXiv/OpenAlex polite-pool User-Agent). Logs never contain tokens or
full message text; `arxiv_id`s and message ids only.

### 12.4 Scheduling

arXiv announces **Sunday–Thursday at 20:00 ET** (no Friday/Saturday announcements) and
moderation delays publication by 1–4 days — so the design uses a **5-day sliding
lookback window plus the seen-set**, rather than "since last run". That makes timezone
and announcement timing non-issues: late-announced papers are still caught.

Recommended schedule:

- **Tue–Fri 07:30 ET** — daily digest covering the previous evening's announcement.
- **Mon 07:30 ET** — daily digest covering Sunday night's batch (largest of the week).
- **Sun 09:00 ET** — optional "Week in Agents" recap (surveys allowed, ranked by the
  week's scores and by reaction feedback).

Deployment options, in order of preference:

1. **Always-on host (recommended).** Launchd on this Mac, or a $5 VPS / Fly.io machine
   with a small volume: SQLite needs a persistent disk, and cron is one line.
   `launchd` plist with `RunAtLoad=false`, `StandardErrorPath` to a log, and the
   `doctor` check as a weekly job.
2. **Fly.io / small container with a volume** — better uptime than a laptop; adds an
   image build and a volume, nothing else.
3. **GitHub Actions cron** — free and external, but stateless: SQLite must become Turso
   / libSQL or the state file must be committed to a private repo, and the cron is
   best-effort (can be delayed 5–30 min). Acceptable fallback, not first choice.

---

## 13. Cost, testing, evaluation

### 13.1 Cost model (per run, ~250 new papers → ~50 gated → ~16 reviewed)

| Tier | Triage model | Review model | Full text | Est. $/run | Est. $/month |
|---|---|---|---|---|---|
| Lean | small (mini/haiku class) | small | no | ~$0.05 | **~$1–2** |
| Balanced (recommended) | small | mid (sonnet/4o class) | no | ~$0.30–0.60 | **~$8–15** |
| Premium | small | frontier | top 8 only | ~$1.20–1.80 | **~$30–45** |

Levers: prompt caching for the static rubric system prompt (−60–80% input cost on the
review tier), provider Batch APIs (−50%, acceptable for a once-a-day job), abstract
truncation, and the cascade itself. A hard `budget_usd` cap per run (default $2)
guarantees the ceiling regardless of input volume.

### 13.2 Performance

End-to-end ≤ 4 minutes, budgeted: fetch 20 s, enrich 30 s, triage 40 s, review 150 s
(bounded concurrency 4, `asyncio.TaskGroup`), compose/send 10 s. Well inside any
scheduler window.

### 13.3 Test strategy

- **Pure unit tests** for `gate`, `score`, `select`, `compose`, `style` — the highest
  value tests, no network, deterministic.
- **Adapter tests** with `respx`/recorded fixtures: Atom parsing, retry/429 behaviour,
  Telegram chunking, S2/HF schema drift. Never live network in CI.
- **Golden-file test** for the rendered digest (HTML mode) so formatting regressions
  are visible in diffs.
- **Invariants:** no chunk > 4096 chars; re-running a date never yields new deliveries;
  every numeric claim in output appears in a source quote.
- **LLM contract tests** against a fake `LLM` port (including malformed JSON, refusals,
  truncated output) to prove degradation paths work.

### 13.4 Offline evaluation (the part that makes this a system, not a script)

A hand-labelled set of ~30 historical papers (include / borderline / exclude, plus a
1–10 desirability rating) is the regression suite for the *product*. Run weekly and on
every prompt or weight change, reporting:

- gate precision/recall vs. labels;
- Spearman correlation between composite score and human ratings;
- **impact calibration**: predicted `impact_forecast` vs. observed citations/upvotes for
  3/6/12-month-old picks (reliability curve). If the forecast is uninformative, the
  `impact` weight is reduced and the honest conclusion is recorded.

### 13.5 Prompt and weight change control

Prompts are versioned files (`review.v2.md`); every `Assessment` row records the
version, so the effect of a prompt change is measurable before it ships. CI runs the
eval suite on a prompt change and fails if precision drops more than 5 points.
Weights are YAML+`config_hash`; `screener replay` shows exactly which papers a weight
change would have added or dropped.

---

## 14. Security, privacy, legal

- **arXiv API terms:** descriptive `User-Agent` including contact email, ≥3 s between
  calls, one fetch per query per day (the API explicitly asks for caching, and
  `updated` only changes at midnight), `max_results` ≤ 2000 per slice, prefer
  `export.arxiv.org` (not the main site), fall back to OAI-PMH only if bulk need arises.
- **Only abstracts/metadata are sent to LLM providers** — publicly posted content, no
  personal data beyond author names. Full-text tier respects arXiv licenses and stores
  nothing beyond derived notes.
- **No redistribution:** the service publishes original summaries and links; it does not
  mirror PDFs or reproduce long excerpts (quotes are short spans for verification only,
  stored, not delivered).
- **One recipient.** Personal Telegram chat, token scoped to that bot, no public channel
  by default. Rate limits respected (Telegram: ≤ 1 message/s per chat, 4096 chars).

---

## 15. Roadmap

**v0 — walking skeleton (1–2 days).** `fetch → gate → single LLM call → compose →
Telegram`, SQLite seen-set, launchd job, `doctor`. Proves the whole path end to end
with deliberately dumb internals.

**v1 — the product (3–5 days).** Triage cascade, full rubric + deterministic ranker,
faithfulness verifier, style lint, HTML digest format, `dry-run`/`replay`, eval fixture
set, budget cap, heartbeat.

**v1.5 — depth.** Enrichment (S2/OpenAlex/HF/GitHub), full-text tier for finalists,
weekly recap mode, reaction feedback capture, backtest job.

**v2 — personalization.** Learned weights from feedback, profile notes extracted from
replies, per-topic sections, semantic search over history (`/search` in the bot).

**v3 — breadth.** Additional sources (OpenReview, HF Daily, major lab blogs), "second
look" resurfacing when a past pick gains traction, a static Markdown/HTML archive.

---

## 16. Decisions

Confirmed by the product owner (these are the defaults the design is built on):

| # | Decision | Confirmed choice | Consequence for the build |
|---|---|---|---|
| 1 | LLM tiering | **Balanced** — small model triages, mid-tier model reviews | `llm_fast` / `llm_deep` are two configured models behind one `LLM` port; budgets in §13.1 use the Balanced row |
| 2 | Deployment host | **This Mac, via launchd** | SQLite stays a local file; need wake-time handling (`pmset repeat wake`), `doctor` as a weekly launchd job, and a heartbeat monitor because a sleeping laptop is the top silent-failure risk |
| 3 | Topic scope | **LLM/agentic systems only** | Negative examples for classical MAS/MARL and agent-based simulation are mandatory in both the gate and the triage prompt |
| 4 | Digest shape | **3–6 quality-gated picks, ~900 chars each; 0 is acceptable** | `selection.max_papers = 6`, `min_score = 6.5`, `per_topic_cap = 2` |
| 5 | Full-text review | Deferred to v1.5 (top 8 only) | v1 reviews abstracts; `review.py` accepts an optional `full_text` field from day one so the tier drops in without refactoring |
| 6 | Summary language | English (papers are English) | One generation pass; a `language` config key exists for later |

### launchd specifics (decision 2)

Two plists in `~/Library/LaunchAgents/`, both pointing at `uv run screener run`:

- `com.agents.digest.daily.plist` — `StartCalendarInterval` 07:30 local, Tue–Fri
  (Mon included via a second entry, since Sunday's announcement lands Monday).
- `com.agents.digest.weekly.plist` — Sunday 09:00, `--mode weekly`.
- `com.agents.digest.doctor.plist` — Sunday 08:00, `screener doctor` → heartbeat.

Because the Mac may be asleep at 07:30, `launchd` fires the job on wake (it coalesces
missed calendar events), which is why the run is idempotent and window-based rather
than clock-dependent. `pmset repeat wakeorpoweron MTWRF 07:25:00` makes the schedule
dependable. A missed day is self-healing: the 5-day lookback plus the seen-set means
the next run still covers the gap.
