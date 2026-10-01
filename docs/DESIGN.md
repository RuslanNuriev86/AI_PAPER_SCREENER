# Agent Papers Daily — System Design

A service that finds the most promising arXiv papers on AI agents, writes short,
faithful, high-signal summaries with an explicit "why this matters" rationale, and
delivers a daily digest to Telegram.

**Status:** design only — no implementation yet. Contracts in §6.4, §10 and §11 are now
concrete enough to build v1 against; §15 records what v0 needs versus what can wait.
**Target:** Python 3.12 (`uv`-managed), run as a scheduled CLI job on a Linux host via
systemd timers (§12.5) — sleeping or rebooting is expected and handled, not assumed away.
Prerequisites and two verified environment gotchas are in §16.

---

## 1. Problem definition

### 1.1 What the hard part actually is

The naive version of this product — "fetch new arXiv papers, ask an LLM to summarize
them, post to Telegram" — fails for four specific reasons. The design exists to solve
those four, not to wrap an API.

| Failure mode | Why it happens | Design response |
|---|---|---|
| **Noise: "agent" is a wildly overloaded word** | cs.AI/cs.LG daily output is full of RL agents, agent-based simulation, economic/market agents, biological agents, and control-theory multi-agent systems. A keyword match on "agent" is ~70% wrong. | Two-layer relevance: cheap deterministic gate + LLM classification with explicit negative examples (§5). |
| **"Promising" is invisible on day 0** | Citation counts, the usual proxy for importance, are 0 for a paper announced yesterday. | **Stop rating on day 0.** The digest ranks the cohort that has just reached **T+14**, where citations, repo stars and venue are measurable facts rather than guesses (§5.2). Quality is still judged from the text — novelty, rigor, evidence — but *impact* is observed, not forecast. Nothing is predicted that could instead be measured. |
| **LLM summaries are generic and unfalsifiable** | Without a contract, models emit "paves the way for future work" filler, and invent numbers that were never in the abstract. | Rubric-bound prompts, "why it matters" restricted to four named lenses, banned-phrase lint, and a separate faithfulness checker that verifies every claim against the source text (§8). |
| **Daily digest quietly degrades** | A scheduler that stops running, a Telegram token that expires, or an LLM outage produces silence nobody notices. | Idempotent runs, structured run records, heartbeat monitoring, and degraded-but-delivered fallback instead of nothing (§12). |

### 1.2 Success criteria

- **Precision over recall.** 3–6 papers/day, and *zero* is an acceptable and better
  outcome than padding the digest. Target ≥ 70% of delivered papers rated
  useful/relevant on feedback. Precision governs what is *delivered*; recall is
  not chased in the digest but *measured* by the maturity loop (§6.6.4), which is the only
  place a missed paper can be discovered rather than assumed away.
- **Every item is self-contained.** Reading the digest alone tells you what was done,
  why it matters, and what is weak about it — without opening the PDF.
- **Reproducible and tunable.** Selection is deterministic given stored assessments;
  ranking weights live in YAML, not in prompts or code.
- **Grounded in measured evidence.** Every rating names its basis: which signals were read,
  at what age, and how each contributed. For a T+14 cohort that means citations, repo stars
  and venue — facts, not forecasts — and the digest prints them (§7.3). The remaining
  LLM-judged dimensions are explicitly the *quality* half, never dressed up as impact.
- **Measurably calibrated, or honestly not.** The quality half of the rating is checked
  against the T+90/T+180 rungs, which the T+14 rating cannot yet see. Target Spearman ≥ 0.35
  at T+90 and a non-flat reliability curve. If a matured rung says the quality judgment is
  uninformative, its weight is cut and the negative result is recorded (§6.6.4) — not
  re-prompted until the number looks better.
- **Cost predictable and capped.** A per-run USD ceiling that is *soft*: a run that
  reaches it stops spending and still delivers what it already verified, recorded
  `degraded`, rather than overspending or going silent (§11, §12.1).
- **Unattended.** Runs daily with no human action; failures are loud, not silent.

### 1.3 Non-goals (v1)

- No full literature review / citation-graph analysis. Citation *counts* are read as an
  outcome signal (§6.6.2); the citation graph is never traversed, and there is no
  related-work expansion, co-citation clustering, or author-graph mining.
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
   `(arxiv_id, version, kind)` with a "sent at most once per kind" guarantee (§10).
5. **Ports and adapters, no framework.** Each external system (arXiv, OpenAlex/HF,
   LLM, Telegram) sits behind a narrow `Protocol`. The pipeline is testable with fakes
   and has no import of any vendor SDK.
6. **Laconic but boring on purpose.** Standard library + a handful of small
   dependencies; explicit SQL over an ORM; one entrypoint; no async framework beyond
   `asyncio` + `httpx`.
7. **Labels are the non-renewable asset.** A paper enrolled today yields its T+180
   outcome in six months; a paper never enrolled yields nothing, forever. So enrolment
   starts in the walking skeleton and measurement ages are fixed rungs, even while the
   calibration that consumes them is still being built (§6.6). Collection and
   exploitation are deliberately decoupled: you can always analyse old labels later, you
   can never retroactively collect them.
8. **The delivery path is sacred.** Nothing that can fail, slow down, or cost money may
   block the one job whose output a human is waiting for. The maturity loop is a separate
   command with its own budget and its own failure domain for exactly this reason
   (§4, §6.6).

---

## 3. Architecture

Modular monolith, one deployable, invoked as a scheduled CLI. Pipeline stages are
async functions over typed models; every I/O boundary is an adapter injected at the
composition root.

```
┌─ screener run (CLI, systemd timer) ────────────────────────────────────────────────────────────────────────────┐
│                                                                                                               │
│  arXiv ─▶ 1 fetch ─▶ 2 gate ─▶ 3 enrich ─▶ 4 triage ─▶ 5 review ─▶ 6 rank ─▶ 7 compose ─▶ 8 send ──▶ Telegram │
│  (Atom)            (pure)     (S2/OpenAlex   (cheap LLM)  (strong LLM)  (pure)     (pure)                     │
│                                HF/GitHub)                  bounded)     gates)     chunked)                   │
│                                                                                                               │
└───────────┬───────────────────────────────────────────────────────────────────────────────────────────────────┘
            │  every gated paper enrolled with its day-0 verdict (§6.6.1)
            ▼
┌─ screener revisit (separate job, off the delivery path) ──────────────────────────────────────────────────────┐
│                                                                                                               │
│  S2/OpenAlex/HF/GitHub/arXiv-vN ─▶ R1 plan ─▶ R2 probe ─▶ R3 grade ─▶ R4 calibrate ─▶ R5 resurface            │
│                                    due rungs   best-effort   0-10 label   reports      ≤1 dated second look   │
│                                                                                                               │
└───────────────────────────────────────────────────────────────────────┬───────────────────────────────────────┘
                                                                        └──▶ next digest, ≤ 1 dated "second look" (§6.6.5)
┌─ SQLite ──────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  papers · assessments · runs · deliveries · feedback · watchlist · outcomes · revisits · calibration          │
│  seen-set (at-most-once delivery) · score history · cost ledger · calibration drift series                    │
└───────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Layers

```
src/screener/
  domain/        # pure: models, gate rules, scoring, digest composition. No I/O, no vendor imports.
    models.py         Paper, GateResult, RelevanceHint, TriageScores, Scores, SoftFlag,
                      HardFlag, Review, Assessment, Ranking, Selection, Digest, Run,
                      OutcomeSignal, Rung, WatchlistEntry, CalibrationReport
    relevance.py      deterministic agentic gate + topic classification
    scoring.py        Weights, score(), select() with MMR-style diversification
    compose.py        Ranking -> Telegram-safe HTML chunks + item -> chunk map
    style.py          banned-phrase lint, summary-collision detection
    maturity.py       rung scheduling, outcome_scale mapping, cohort stats, calibration, miss audit
  ports.py       # Protocols: Clock, PaperSource, Enricher, ImpactProbe, LLM, Notifier, Repository
  adapters/
    arxiv.py          Atom API client (rate-limited, paginated, retrying,
                      single connection — see the ToU constraint, §14)
    enrich.py         Semantic Scholar / OpenAlex / HF Daily / GitHub adapters
    llm_openai.py     structured-output client (one of N interchangeable providers)
    llm_anthropic.py
    telegram.py       sendMessage / chunking / getUpdates (message updates only, §8.3)
    heartbeat.py      dead-man's-switch ping (§12.2)
    probe.py          outcome probes: S2/OpenAlex citations, GitHub stars, HF upvotes,
                      arXiv vN/comment re-read, venue lookup (reuses enrich.py clients)
    sqlite_repo.py    stdlib sqlite3 repository, migrations from .sql files
  pipeline/
    fetch.py  gate.py  enrich.py  triage.py  review.py  rank.py  compose.py  deliver.py
    run.py            # orchestrator: builds deps, runs stages, records the Run
    revisit.py        # plan | probe | grade | calibrate | resurface (§6.6) — off the delivery path
  prompts/
    triage.v1.md  review.v2.md  verify.v1.md  miss_audit.v1.md
  config.py       # pydantic-settings: env + YAML (profile, selection, triage, revisit,
                  #                  outcome_scale) + config_hash
  cli.py          # typer app: run [--mode daily|weekly] | dry-run | replay | feedback
                  #            | revisit | backtest | doctor | eval | prune
  migrations/*.sql
  config/         # profile.yaml · selection.yaml · triage_weights.yaml
                  #   · outcome_scale.yaml · revisit.yaml  (one file per Settings field)
  deploy/systemd/ # *.timer + *.service (§12.5)
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
| 1 | **fetch cohort** | the T+14 window (§5.2) → `list[Paper]` | free | 30–70 s cold, ~0 s cached (**sequential on one connection**, large pages, day-cached) | retry ×3 w/ adaptive backoff; **one dead category must not lose the rest**; a total outage ends the run with a notice, not a traceback |
| 2 | **gate** | `Paper` → `GateResult` | free | <1 s | pure function, always succeeds; reason + `RelevanceHint` persisted |
| 2b | **dedupe_revisions** | `list[Paper]` → `list[Paper]` | free | <1 s | stateful (§6.2); drops a cosmetically-revised paper already delivered |
| 3 | **enrich** | `Paper` → `Enrichment` | free (polite pools) | 10–30 s | **core, not optional** (§5.2): citations + venue from Semantic Scholar, stars from GitHub, upvotes from HF. A dead source renormalises its dimension away rather than scoring 0 |
| 4 | **triage** | 40–80 papers → `Triage` scores | ~$0.02 | 20–40 s | batch failures retried, then fall back to gate-only ordering |
| 5 | **review** | top `review_top_k` (=16) → `Assessment` | $0.20–$1.50 | 60–180 s | per-paper failure drops that paper; ≥3 failures ⇒ degraded run |
| 6 | **rank** | `Assessment[]` → `Ranking[]` | free | <1 s | pure, deterministic |
| 7 | **compose** | `Ranking[]` → HTML chunks + item→chunk map | free | <1 s | pure; asserts no chunk > 4096 chars |
| 8 | **deliver** | chunks → message ids | free | 2–5 s | retry; on final failure the digest is written to `outbox/` and a heartbeat alert fires |

Stage boundaries are also persistence boundaries: `papers`, `gate_results`, `assessments`
and `deliveries` are written so a failed run can be resumed and every digest is
reconstructable (`screener replay --date`).

### Flow B — the revisit pipeline (`screener revisit`, §6.6)

Not on the delivery path. Its own command, its own job, its own budget, its own failure
domain — a total probe outage cannot delay or degrade a digest (§2, principle 8).

| # | Stage | Input → Output | Cost | Failure behaviour |
|---|---|---|---|---|
| R1 | **plan** | `watchlist` × `now` → due `(paper, rung)[]` | free | pure; rungs are due-dated from `cohort_date`, so a missed day measures late rather than losing data |
| R2 | **probe** | due list → `OutcomeSignal[]` | free (polite pools) | per-source best-effort; a dead source drops one component and the grade renormalises — never zeroes the paper |
| R3 | **grade** | signals → `matured_impact` ∈ 0–10 | free | pure; config-driven mapping (§6.6.3) |
| R4 | **calibrate** | graded cohort → `CalibrationReport` | ~$0.01–0.05 | analysis only; a failure here is logged and ignored |
| R5 | **resurface** | report → ≤ 1 dated "second look" candidate | free | pure; feeds the *next* digest — a sent digest is never edited |

Idempotent per `(arxiv_id, rung)`: re-running the same day re-measures nothing already
recorded.

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

### 5.2 The rating point is T+14, and the cohort is defined by it

The unit of the digest is not "papers announced recently" but **the cohort that has just
reached T+14**. A run collects the papers whose announcement date falls in

```
[now − (cohort_age_days + catch_up_days),  now − cohort_age_days]
```

with `cohort_age_days: 14` and `catch_up_days: 2`. The catch-up range is what makes a missed
day self-healing: the host may sleep, and a paper must not silently miss its only rating
opportunity. The `(arxiv_id, version)` seen-set guarantees each paper is digested exactly once
however many times it falls inside the window.

Why 14 and not 0, 7 or 30:

| age | citations | repo stars | venue | verdict |
|---|---|---|---|---|
| T+0 | 0 by construction | 0 unless pre-released | none | unratable — every impact signal is absent |
| **T+14** | small but non-zero, and separating | **the most informative signal at this age** | occasionally present (authors post after acceptance) | **chosen** |
| T+90 | usable for ranking | mature | common | better signal, but a three-month-old digest has lost its news value |
| T+180 | strong | mature | near-complete | the calibration rung, not the delivery rung |

Two honest constraints, stated so nobody designs around a signal that does not exist:
**citations at T+14 are small** (most papers read 0–2), so the scale is compressed and a single
citation moves the score a lot; and **venue acceptance is rare at T+14**, because most decisions
land later. Both are *measured* rather than assumed — and when a signal is absent it is
renormalised away, never scored zero (§6.5).

The consequence for the product: the digest is a fortnight behind the arXiv firehose. That is
the trade the design makes deliberately — a rating that is grounded in observed evidence beats a
rating that is timely and unfalsifiable.

**Profile config** (`config/profile.yaml`) holds all of the above as data:

```yaml
name: "LLM agents & agentic systems"
categories: [cs.AI, cs.CL, cs.MA, cs.LG, cs.SE, cs.CR, cs.HC, cs.RO]  # RO: embodied/computer-use only
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
min_gate_score: 2.0   # one strong term = 2.0, one weak term = 1.0, each exclusion = -1.0
                      # 2.0 means "a weak term alone is not enough" — see §13.1 for the
                      # measurement that set this default
```

Adding a subfield is a YAML edit. `boost_topics` is also the closed `Topic` vocabulary
(§6.4): it is the only place subfield names are defined, and selection's caps refer to it.

**`per_topic_cap` is not here.** It is a selection parameter, and it lives in exactly one
place — `selection.per_topic_cap` (§6.4) — because it was previously duplicated in this
profile and again in §16, which is how the same key ends up with three homes and two
values. Rule: **profile config decides what is relevant; selection config decides what is
printed.**

---

## 6. Assessment and selection

### 6.1 Rubric — the definition of "most proficient and promising"

Eight scored dimensions, each 0–10, in two explicitly separated halves. The split is the point:
**quality is judged, impact is measured**, and the digest must never present the first as if it
were the second.

| Dimension | Weight | Source | Definition (what earns a 9) |
|---|---|---|---|
| `relevance` | 0.12 | LLM, from text | Directly advances LLM-agent capability/reliability/evaluation *and* matches the profile. 9 = squarely in a boosted topic. |
| `novelty` | 0.16 | LLM, from text | 9 = a new mechanism, formulation, or measurement the field did not have; 3 = a recombination. Penalise "we apply X to Y" with no insight. |
| `rigor` | 0.13 | LLM, from text | 9 = multiple strong baselines, ablations, error bars/seeds, honest limitations, released artifacts. 3 = single baseline or self-reported only. |
| `evidence_strength` | 0.08 | LLM, from text | Magnitude *and* credibility of the demonstrated result: quoted numbers, eval-suite size, held-out conditions, human eval. |
| `reproducibility` | 0.06 | LLM, corroborated | Code, data and artifacts released and plausibly runnable. Corroborated by `repo_signal` — a claim of released code with no findable repo scores lower. |
| `citation_signal` | 0.08 | **measured at T+14** | Citations already accrued. **Normally absent** (§6.1.0): measured zero for every cohort sampled, because citations accrue to the published DOI while we hold the arXiv DOI. Kept as a dimension so the T+90 rung can carry it once a merging source exists. |
| `repo_signal` | 0.25 | **measured at T+14** | GitHub stars on the linked repo, **counted only if the repo was created near the paper** (§6.1.0). The only measured signal with real spread at this age. |
| `venue_signal` | 0.12 | **measured at T+14** | A stated venue or acceptance, from the arXiv `comment` field. Absent at this age in every cohort measured; decisive when present. |

The quality half totals 0.55 and the measured half 0.45. `impact_forecast` is **gone**: it was a
guess about citations, and at T+14 we can simply read them. `pedigree` and `early_signal` are
likewise gone — folded into `venue_signal` and `repo_signal`, which measure the same things
instead of inferring them.

#### 6.1.0 What the signals actually measure at each age (measured, not assumed)

Before choosing the rating point, the signals were sampled from live cohorts (40–60 gated papers
per age), querying OpenAlex by arXiv DOI for citations and the GitHub API for stars:

| age | cited papers | max citations | papers with a findable repo | star distribution |
|---|---|---|---|---|
| T+14 | **0 / 50** | **0** | 4 / 40 (10%) | 7, 4, 2 |
| T+30 | **0 / 39** | **0** | 8 / 40 (20%) | 34432, 1147, 52, 3, 1, 1, 1, 1 |
| T+60 | 1 / 40 | 1 | 6 / 40 (15%) | 2117, 15, 4, 0, 0, 0 |
| T+90 | **0 / 39** | **0** | 6 / 40 (15%) | 22, 1, 0, 0 |

Three conclusions, and the first two are uncomfortable:

1. **Citations are unusable as a rating input at T+14 — and, as measured here, at every age
   sampled.** Zero of 39 papers at T+90 showed a citation. The constraint is not paper age but
   *attribution*: citations accrue to a paper's **published version DOI**, while we look up the
   **arXiv DOI**, so the preprint record stays at zero forever. `citation_signal` therefore stays
   in the rubric as a first-class dimension that is normally *absent* (and renormalised away,
   §6.5) rather than a 0.20 weight that silently contributes nothing. Making it real needs a
   source that merges preprint and published records — Semantic Scholar's API key, since its
   unauthenticated pool 429s — and that is a v1.5 task with its own measurement.
2. **Repo stars discriminate, but raw counts are contaminated.** The T+30 sample contains a
   34,432-star repository. That is not a paper's own traction; it is a paper linking to a
   *pre-existing* popular project. Stars are therefore only counted when the repository was
   created within `repo_young_days` (default 45) of the paper's announcement, which is a signal
   about *this* paper. Without that guard one link can dominate a cohort.
3. **Venue is absent at T+14** and OpenAlex reports only "arXiv (Cornell University)" — the
   preprint record. A venue is decisive when present, which is why it keeps a place, but it is
   rare and must render as "no venue yet" rather than as silence.

The rating point stays at **T+14**, but on the strength of the signals that survive this
measurement — repo traction, code release, and two weeks of citation-independent hindsight —
not on the citation counts the table shows are not there.

#### 6.1.1 Anchors are derived from the observed T+14 distribution

The measured dimensions use written anchors in `config/outcome_scale.yaml`, re-derived from
cohort quantiles rather than invented. That matters more here than anywhere else in the design,
because a naive mature-paper scale (say "50 citations is a 9") would score **every** T+14 paper
between 0 and 1 and flatten the whole measured half to noise.

Anchors must also **never move silently**: they live in config, they enter `config_hash`, and the
calibration report prints the cohort's p50/p90/p99 per rung so drift is visible (§6.6.3).

**Impact forecast anchors** are retained in the prompt only as the calibration check for the
quality half — the T+90 rung is what tells us whether a `novelty` 9 at T+14 predicted a paper
that mattered.

### 6.2 Hard gates (applied before scoring, and re-checkable after review)

```python
class GateResult(BaseModel):
    """The single gate contract (§4 stage 2). Replaces the earlier three-way conflict
    between `Paper | None`, a bare `RelevanceHint`, and a truthiness check."""
    paper: Paper                                  # carried forward; keeps the pipeline linear
    keep: bool
    reason: HardFlag | None = None                # set iff keep is False (§6.2 promise)
    hint: RelevanceHint                           # §5 evidence, fed to triage — never a decision
```

`not_agentic` · `pure_survey` (surveys are allowed only in a dedicated weekly recap) ·
`no_technical_contribution` (position papers, editorials) · `marketing/whitepaper` ·
`withdrawn`. These are exactly the `HardFlag` members of §6.4 — one vocabulary, two
places it can be discovered (gate, review). A gated paper is recorded with its reason in
`gate_results` and never delivered. Gates are *documented and visible* in the digest's
scan footer — a good paper we filtered by mistake is discoverable, not invisible.

Rejection is decided by `keep`, never by the truthiness of the result object — a Pydantic
model is always truthy, so the earlier `if (hint := gate(...))` filter would have admitted
every paper and made the entire gate a no-op.

Gate-stage rejections come from the paper text alone. One §6.2 rule — "`replaced` with only
cosmetic changes **and already delivered**" — depends on delivery state, so it is
deliberately **not** in `gate()`. It lives in a separate stateful
`dedupe_revisions(fresh, repo)` step (§11), which keeps `gate` pure and testable
(§2, principle 1).

### 6.3 Stage 4 — triage (cheap model, all candidates)

One batched call per ~10 papers. Output is validated against a Pydantic schema; failures
retry once with the validation error appended, then the paper falls back to gate-only
ordering.

```python
class TriageScores(BaseModel):
    agentic: bool
    relevance: float = Field(ge=0, le=10)
    novelty: float = Field(ge=0, le=10)
    impact: float = Field(ge=0, le=10)
    rigor: float = Field(ge=0, le=10)
    is_survey: bool
    soft_flags: list[SoftFlag] = []
    hard_flag: HardFlag | None = None        # e.g. NOT_AGENTIC / NO_TECHNICAL_CONTRIBUTION:
                                             # §5 requires triage to be able to raise these
    reason: str = Field(max_length=140)      # ~15 words
```

Purpose is ordering, not judgment — so the ordering key must be *stated*, not implied.
Triage emits four of the rubric's eight dimensions, so the default is the **§6.1 rubric
weights renormalised over exactly those four**. §6.1 gives relevance .20, novelty .20,
impact .20, rigor .15 — total .75 — so dividing through yields **.2667 / .2667 / .2667 /
.20**, and the cheap tier inherits the expensive tier's preferences instead of inventing a
second value system:

```yaml
# config/triage_weights.yaml  (literally §6.1 renormalised over its 4 observable dims)
weights: {relevance: 0.27, impact: 0.27, novelty: 0.26, rigor: 0.20}  # .2667x3 -> .27,.27,.26
disqualify: {agentic: false, is_survey: true}   # forced to the bottom, never dropped here
```

(The exact values are .2667 each; the third is written .26 so the four round to exactly
1.00 rather than 1.01. A test asserts `sum(weights) == 1.0` for every weight map, which is
how the earlier `.27/.27/.27/.20` slip would have been caught.)

The keys are in config because triage ordering is the cheapest thing to get wrong and the
cheapest to retune — but the *default* is derived, not hand-picked, so the two tiers cannot
drift apart by accident. (An earlier draft of this section listed 0.45/0.25/0.20/0.10 and
described it as renormalised; those numbers were neither derived nor consistent with the
claim, which is exactly the kind of plausible-looking constant that silently reorders the
funnel.)

`triage_order = Σ w_i · dim_i`, then any paper matching `disqualify` is sorted below all
others (it still reaches the footer stats, and `is_survey` papers remain eligible for the
weekly recap). `top_k(triaged, cfg.review_top_k)` sorts by that key.

`review_top_k` is a single config value, **default 16** — resolving the previous spread
between the stage table's "12–20" and the cost model's "~16"). It is bounded by budget,
not by taste: at ~$0.03
per reviewed paper in the Balanced tier it is the dominant cost lever (§13.1).

### 6.4 Stage 5–6 — review, then deterministic selection

The strong model reviews the top `review_top_k` (§7). Selection is then a pure function
over *typed* inputs. The models are the contract, so they come first.

```python
# domain/models.py — the closed vocabularies. Everything else refers to these.
Topic = Literal["evaluation & benchmarks", "multi-agent coordination", "memory & context",
                "computer use", "safety & oversight", "agentic RL", "infrastructure/protocols"]
Dimension = Literal["relevance", "novelty", "rigor", "evidence_strength",
                    "impact_forecast", "reproducibility", "pedigree", "early_signal"]

class SoftFlag(StrEnum):            # penalised, never disqualifying (§6.5)
    NO_CODE, SINGLE_BASELINE, NO_ABLATION, NO_ERROR_BARS, SELF_REPORTED_ONLY, \
    NARROW_BENCHMARK, OVERCLAIMED, NO_LIMITATIONS = ...

class HardFlag(StrEnum):            # disqualifying; §6.2's gates, by another route
    NOT_AGENTIC, PURE_SURVEY, NO_TECHNICAL_CONTRIBUTION, MARKETING_WHITEPAPER, WITHDRAWN = ...
```

```python
class Scores(BaseModel):
    """§6.1 rubric. None means *unobservable*, not zero — §6.5 renormalises."""
    relevance: float = Field(ge=0, le=10)
    novelty: float = Field(ge=0, le=10)
    rigor: float = Field(ge=0, le=10)
    evidence_strength: float = Field(ge=0, le=10)
    impact_forecast: float = Field(ge=0, le=10)
    reproducibility: float = Field(ge=0, le=10)
    pedigree: float | None = None            # needs enrichment (venue/author record)
    early_signal: float | None = None        # needs enrichment (HF/stars)

class Ranking(BaseModel):
    """Emitted only by rank.run(). A Ranking cannot exist without an Assessment (§11)."""
    arxiv_id: str
    assessment: Assessment
    score: float
    components: dict[Dimension, float]      # weight_i * dim_i, post-renormalisation
    effective_weights: dict[Dimension, float]   # what was actually used, for replay
    soft_flags: list[SoftFlag] = []
    disposition: Literal["eligible", "gated"] = "eligible"
    hard_flag: HardFlag | None = None       # set here only by *review*; gate-stage
                                            # rejections never become Rankings at all
    topics: list[Topic] = []                # normalised from review.tags; drives per_topic_cap
    lab: str | None = None                  # senior-author institution, see below
    tags: list[str] = []                    # raw model tags, kept for MMR similarity

class Selection(BaseModel):
    max_papers: int = 6
    min_score: float = 6.5
    per_topic_cap: int = 2
    per_lab_cap: int = 3
    max_tag_jaccard: float = 0.60
    headline_score: float = 8.5             # exempt from per_topic_cap

class Enrichment(BaseModel):
    """Day-0 signals from external sources (§4 stage 3). All optional: absence is normal
    at v1 and must not be read as a negative signal (§6.5)."""
    arxiv_id: str
    citations: int | None = None            # S2 / OpenAlex, if the paper is old enough
    influential_citations: int | None = None
    hf_upvotes: int | None = None
    hf_daily_rank: int | None = None
    stars: int | None = None                # GitHub, resolved from paper.code_url
    institutions: list[str] | None = None   # OpenAlex affiliations, ordered as arXiv lists
                                            # authors; last entry = senior author, which is
                                            # what Ranking.lab is derived from
    venue: str | None = None                # only from a later version's comment field
    sources_ok: list[str] = []              # which adapters answered, for §6.5 reporting
```

```python
def select(ranked: list[Ranking], cfg: Selection) -> list[Ranking]:
    kept = [r for r in ranked if r.score >= cfg.min_score and r.disposition == "eligible"]
    kept.sort(key=lambda r: r.score, reverse=True)
    picks: list[Ranking] = []
    for r in kept:                                  # greedy, diversified
        if len(picks) == cfg.max_papers: break
        headline = r.score >= cfg.headline_score
        if not headline and _over(cfg.per_topic_cap, r.topics, picks): continue
        if not headline and _over(cfg.per_lab_cap, r.lab, picks): continue
        if _too_similar(r, picks, cfg.max_tag_jaccard): continue
        picks.append(r)
    return picks
```

Diversification (a light MMR over topic tags) is what keeps the digest *useful*: five
variants of the same RAG-improvement idea is a worse product than one of them plus
three unrelated advances.

**Where `topics`, `lab` and `disposition` come from — no undecided names:**

- `topics` — the review model's `tags` are routed onto the closed `Topic` vocabulary of
  §5 `boost_topics` by the explicit anchor table in `types.TOPIC_ANCHORS`. Tags that match
  nothing are dropped from `topics` but kept in `tags` for similarity, so no free-form string
  reaches `per_topic_cap`.

  **Anchors are explicit data because deriving them fails in both directions.** Deriving from
  the topic names lets generic domain words match everything — on live data "multi" matched
  "multi-agent coordination" for a paper tagged `multi-hop-qa`, and six of six papers collapsed
  into one topic, which turned `per_topic_cap` from a diversification rule into a *global* cap
  on the digest. Deriving also yields **zero** anchors for "agentic RL", because its only
  distinctive word is one of the generic ones, so that topic could never be matched at all.
  The three matching rules are: a phrase anchor matches the raw tag; an anchor of ≥6 characters
  matches a tag word sharing its first 6 (so "benchmark"/"benchmarks" line up); a shorter anchor
  must match a whole word (otherwise "eval" fires on "ret**rieval**" and "rl" on "wo**rl**dly").
- `lab` — the **normalised institution of the last author** (senior-author convention),
  taken from `Enrichment.institutions` (OpenAlex author affiliations). `None` when
  enrichment is absent or has no affiliation, in which case `per_lab_cap` simply does not
  apply to that paper. We deliberately do **not** guess a lab from an author-name string.
- `disposition` — `gated` is set by *review* discovering a `HardFlag` (§6.2 gates applied
  after review), which is the only way a `Ranking` can be disqualified. Rejections at the
  gate stage never reach `rank.run` at all, so `select` never sees them. This is what
  makes `r.gated` representable (§11).

### 6.5 The ranker

```python
composite = Σ_i (w_i · dim_i)  −  0.5 · len(soft_flags)      # over observable dims only
```

**Hard flags do not appear in the penalty term.** They set `disposition="gated"`, which
excludes the paper outright, so penalising them as well would be double-counting — and
the previous `− 1.0 · len(hard_red_flags)` form was unimplementable because no model
carried a severity field. Severity is now the type: `SoftFlag` vs `HardFlag`.

**Missing dimensions are renormalised, not zeroed.** `pedigree` and `early_signal` are
`None` until enrichment exists (§15 defers enrichment to v1.5). `score()` then drops their
weights and redistributes proportionally across the six observable dimensions, recording
`effective_weights` on the `Ranking`. This is the same rule as §6.6.3: an unmeasured
signal must not silently become a low score.

The v1 weights are therefore fully determined, not left to the implementer. §6.1's eight
weights total 1.00; the six observable ones total 0.90, so dividing through gives:

| | relevance | novelty | rigor | evidence | impact | reproducibility | pedigree | early_signal |
|---|---|---|---|---|---|---|---|---|
| v1.5+ (§6.1) | .200 | .200 | .150 | .100 | .200 | .050 | .050 | .050 |
| **v1 (renormalised)** | **.222** | **.222** | **.167** | **.111** | **.222** | **.056** | — | — |

Both rows sum to 1.0. `Ranking.effective_weights` records which row was applied, so a
weight change or a v1→v1.5 upgrade is replayable against history rather than silently
altering every past comparison.

`score()` also emits the full component breakdown, stored with the ranking. That
breakdown plus `effective_weights` is the audit trail: any digest item can be explained
months later, and weight changes can be replayed offline against history
(`screener backtest`).

The day-0 composite is computed from day-0 features only. Matured outcome signals
(§6.6) are **never** blended into it: they do not exist at T+0, and using them would make
the backtest circular and the success criteria meaningless. They change the product only
through offline calibration — weights, gates, thresholds — and the explicit dated
resurfacing rule (§6.6.5).

### 6.6 The maturity loop: measuring promise instead of only guessing it

Day-0 ranking is a *forecast*, and a forecast that is never checked is a guess with
better branding. The maturity loop is the check: every paper the gate lets through is
enrolled in a watchlist, then re-measured at fixed ages after announcement using signals
that only exist with time — citations, repo stars, venue acceptance, community pickup.
Those measurements are the ground truth that says whether "promising" in this system
means anything at all.

#### 6.6.1 Enrolment and cohort

Every paper that survives the deterministic gate is enrolled in `watchlist` with its
day-0 verdict recorded:

| `score_band` | Definition | Why it is in the cohort |
|---|---|---|
| `delivered` | selected and sent | calibrates what we actually shipped |
| `above_min` | scored ≥ `min_score`, not selected (crowded out by topic/lab cap or diversification) | separates *ranker* error from *policy-cap* error |
| `mid` | scored 4.0 – `min_score` | the reliability curve needs its middle |
| `low` | scored < 4.0 | anchors the bottom of the curve |
| `gate_only` | passed the gate, never reviewed (budget / `top_k` cutoff) | recall check on the triage funnel |

Enrolment is one cheap row, and it is the part that must start early. A paper enrolled
today yields its T+180 rung in six months; no later engineering recreates a label that
was never collected (§2, principle 7).

The cohort is deliberately **not** just the delivered papers. Measuring only what we
shipped answers "were our picks good?" and cannot answer "what did we miss?" — which is
the more expensive error, and the one a ranker is most likely to be making.

Per-band daily caps bound the cost and are applied at enrolment (values in §6.6.2), so
the cohort size is a config decision rather than a consequence of how busy arXiv was.

#### 6.6.2 Measurement ages

One daily job measures whatever is due. Ages are fixed rungs, not "whenever we get to it":

| Rung | What it reads | Why this age |
|---|---|---|
| **T+14** | GitHub stars on the repo named by `papers.code_url` — populated at **fetch** from arXiv metadata, so it does not depend on the v1.5 `Enricher` — plus HF Daily upvotes, code-release existence, arXiv `vN` revision count, whether the `comment` field changed, cheap social pickup | the earliest horizon at which anything has separated |
| **T+90** | citation count and influential-citation count (S2/OpenAlex), star velocity, venue field if present | citations become a usable ranking signal at about a quarter |
| **T+180** | venue acceptance (arXiv `comment` on a later version, S2/OpenReview `venue`), matured citations | acceptance decisions actually land here; this is the rung that validates the `pedigree` weight |

Two constraints stated up front, so nobody designs around signals that do not exist:
**citation counts are near-dead at T+14** (most papers still read 0), and **venue
acceptance is unavailable before T+180** for most papers. A horizon longer than one day
is the right instinct; the ladder is what makes it true *per signal* rather than on
average.

```yaml
# config/revisit.yaml
rungs: [14, 90, 180]
max_lateness_days: {14: 21, 90: 45, 180: 90}   # beyond this the rung is 'missed', not back-filled
cohort:                                        # per-day caps, stratified by score band (§6.6.1)
  delivered: all
  above_min: 20
  mid: 20
  low: 10
  gate_only: 15
resurface:
  min_matured_impact: {above_min: 8.0, mid: 8.0, low: 9.0}   # bar rises as day-0 disagreed more
  rung: latest_matured
  max_per_digest: 1
  max_per_week: 3
audit_prompt: miss_audit.v1.md
probe_sources: [s2, openalex, github, hf_daily, arxiv_vn]
```

**Lateness is expected, not exceptional.** The host may be asleep or a probe may be
down, so a rung records `actual_age_days` alongside its nominal age: a T+14 rung measured
on day 17 is still the T+14 rung, and still usable. Past `max_lateness_days` the rung is
written as `missed` rather than back-filled, because a measurement at T+45 is not a T+14
measurement. T+90/T+180 tolerate proportionally more jitter, per the config above.

#### 6.6.3 From measurements to a label

Raw counts are stored verbatim and never discarded. `matured_impact` (0–10) is derived
from them by an explicit, config-driven mapping in `config/outcome_scale.yaml`, defined
with written anchors exactly the way `impact_forecast` is (§6.1). Without this, "predicted
8" and "observed 8" are not the same statement and the correlation between them is
theatre.

```yaml
# config/outcome_scale.yaml  (defaults; re-anchor from observed quantiles)
t14:                       # stars dominate; citations ignored — they are still ~0
  stars:      {9: 1000, 8: 300, 7: 150, 6: 80, 5: 30, 4: 15, 3: 5, 2: 1, 1: 0}
  hf_upvotes: {9: 200, 7: 80, 5: 30, 3: 10, 1: 0}
  weights: {stars: 0.5, hf_upvotes: 0.3, code_release: 0.1, revisions: 0.1}
t90:
  citations:  {9: 30, 8: 22, 7: 15, 6: 10, 5: 6, 4: 3, 3: 2, 2: 1, 1: 0}
  weights: {citations: 0.6, influential_citations: 0.2, stars: 0.2}
t180:
  weights: {venue: 0.5, citations: 0.4, stars: 0.1}
```

Absolute anchors drift as a field grows, so the calibration report also prints the
cohort's count quantiles (p50/p90/p99) per rung. When p90 citations at T+90 has moved
materially from what the anchors assumed, the anchors are re-derived — a config change
recorded in `config_hash`, not a code change.

**Missing signal is not zero signal.** A paper with no linked repo is graded on the
components it has, with weights renormalised, and `components_present` is stored, so no
paper is ever penalised for a repo that does not exist.

#### 6.6.4 What the loop is for

Four outputs, in descending order of value:

1. **False-negative audit — the main event.** Papers in the `gate_only` / `low` / `mid`
   bands whose `matured_impact` reaches ≥ 7.5 at T+90. One cheap LLM call per paper
   classifies *why* it was missed: gate term missing / triage under-scored / review
   under-scored / crowded out by a cap. A recurring "gate term missing" verdict is the
   only credible evidence that the deterministic gate is too strict, and it produces a
   concrete YAML diff instead of a feeling.
2. **Reliability curve and rank correlation.** Predicted `impact_forecast` (binned) vs
   observed `matured_impact`, plus Spearman and MAE, reported per rung, with drift tracked
   in the `calibration` table so "is the forecast getting better?" has an answer.
   - **Confound that must be controlled:** stars and citations follow *subfield*
     popularity and author marketing, not just quality. The report therefore prints
     correlation overall **and within each topic tag**. A ranker that merely predicts
     "this is a popular subfield" scores well overall and poorly within topic — the
     within-topic number is the honest one.
3. **Digest-lift estimate — a natural experiment.** Papers just above `min_score` were
   delivered; papers just below were not, and they are otherwise similar. Comparing their
   T+90 outcomes estimates how much of a paper's attention *the digest itself* caused.
   This is worth knowing on its own, and it is a warning label on outputs 1 and 2: if
   delivery has a large causal effect, observed outcomes partly measure our own
   amplification, and calibration must then be read on the un-delivered cohort.
4. **Weight and threshold tuning.** The loop *proposes* changes (cut `impact_forecast`
   weight when the curve is flat; relax a gate term; raise `min_score`) and
   `screener replay` / `screener backtest` show exactly which historical papers each
   change would have added or dropped (§13.5). Proposals are applied offline by a human,
   never live by the job.

If the forecast turns out uninformative once a rung has matured, the honest output is a
recorded negative result and a reduced `impact` weight — not a re-worded prompt iterated
until the number looks better.

#### 6.6.5 Bounded resurfacing ("second look")

The digest's identity is *new papers*, so the maturity loop never silently re-ranks old
ones. It may resurface them only under an explicit rule:

- **Qualifies** if: not already delivered; it was actually scored (so `gate_only` papers
  are excluded); no §6.2 hard gate applies; and `matured_impact` at the latest matured
  rung clears a bar that **scales with how badly day-0 disagreed** — ≥ 8.0 for the
  `above_min` and `mid` bands, ≥ 9.0 for `low`. The confidence required rises as the
  original judgment gets more negative, so popularity alone cannot overturn a considered
  low score.
- **`above_min` is the strongest case.** Those papers already scored above threshold and
  were crowded out by `per_topic_cap`/diversification — a miss the digest *structurally*
  creates, and the one it can most cheaply correct.
- **`gate_only` papers are excluded by construction.** They have no day-0 score, and a
  gate miss is fixed by amending the gate (§6.6.4) — which then also helps every future
  paper that term matches. Shipping the one paper would not.
- **Capped** at 1 per digest and 3 per week, so it can never crowd out new work.
- **Labelled**: its own section, always carrying the original announcement date, so it
  cannot be mistaken for a new paper.
- **Suppressed** if the day's new-paper picks already fill the digest. New papers win
  ties by construction.

#### 6.6.6 Temporal leakage is forbidden

`matured_impact` never enters the day-0 composite of a *new* paper. It cannot exist at
T+0, and blending a future signal into a live ranking makes the backtest circular and the
success criteria (§1.2) meaningless. Matured signal reaches the product through exactly
two channels: **(a)** offline calibration that changes weights, gates and thresholds, and
**(b)** the explicit, dated, capped resurfacing rule above. Any code path that puts an
outcome signal into `score()` is a bug, and §13.3 asserts it.

#### 6.6.7 First measurement of the day-0 score distribution

Sixteen live papers reviewed with `deepseek-flash` (top of the gate ranking, 3-day window),
after the gate and prompt fixes recorded in §5 and §7:

| | value |
|---|---|
| min / median / max composite | 0.28 / **4.67** / 6.56 |
| soft flags per paper | min 2, median 3, **max 7** |
| soft-flag penalty | mean **1.72 pts**, max 3.50 pts |
| spread, best to median paper | 1.89 pts |

Two things follow, and the second was not anticipated:

1. **`min_score = 6.5` is slightly too high for this scorer**: it admits 2 of 16, below the
   §1.2 target band of 3–6 picks. A threshold of 6.0 admits 4, which is inside the band. This
   is one sample from one window, so it is a prompt to re-measure rather than a licence to
   retune a product threshold on n=16 — but the current value is not obviously right, and
   before this measurement nobody had looked at the distribution it sits on.
2. **The soft-flag penalty is doing more work than intended.** At 0.5 per flag and a median of
   3 flags, the penalty averages 1.72 points — **91% of the entire spread between the best and
   the median paper**. The ranker is therefore ranking largely by *how many flags the model
   chose to list*, and flag count is prompt-sensitive: a prompt that enumerates eight flags and
   asks for honesty will collect more of them. A clean 7.5 raw paper lands at 4.0 with seven
   flags. The coefficient needs calibrating against the maturity labels, and the flag list in
   the prompt should probably be pruned to items that are disqualifying in aggregate.

Neither is a bug; both are the kind of thing only measurement could reveal. They are recorded
because §13.4's calibration loop is what will settle them, and because a threshold nobody has
looked at the distribution of is an assumption, not a decision.

**Outcome of that measurement: `min_score` was lowered from 6.5 to 5.5.** A second independent
16-paper sample peaked at 6.44, so 6.5 admitted *zero* papers and the digest shipped empty. The
two samples together give: 6.5 -> 0–2 picks, 6.0 -> 4, 5.5 -> 5, 5.0 -> 7–8. At 5.5 both samples
land mid-band for the §1.2 target of 3–6. The soft-flag coefficient is deliberately *not*
changed yet — that alters ranking order rather than a cutoff, so it waits for the maturity
labels rather than being tuned by eye.

---

## 7. Summarization: making the output actually good

Every reviewed paper yields a strict, length-bounded record (validated by Pydantic):

```python
class Review(BaseModel):
    tldr: str            = Field(max_length=220)   # one sentence, no preamble
    what_they_did: str   = Field(max_length=420)   # mechanism, not motivation
    why_it_matters: str  = Field(max_length=420)   # restricted to named lenses
    caveats: str         = Field(max_length=240)   # the honest weakness
    lenses: list[Lens]                             # Lens = Literal["capability","method","safety","adoption"]
    tags: list[str]                                # raw; mapped to Topic in §6.4
    evidence_quotes: list[str]                     # verbatim spans from the source
    scores: Scores                                 # §6.4; pedigree/early_signal may be None
    soft_flags: list[SoftFlag] = []                # penalise (§6.5)
    hard_flag: HardFlag | None = None              # disqualify → disposition="gated"
    model: str; prompt_version: str; cost_usd: float
```

`red_flags` was previously a single `list[Literal[...]]` with the literal set left as an
ellipsis and no severity field — so §6.5's soft/hard penalty split had nothing to read.
Severity is now carried by the *type*: `soft_flags` penalise, `hard_flag` disqualifies.
Both vocabularies are closed and defined once in §6.4. The four `lenses` and the `Scores`
dimensions are likewise closed `Literal`/`StrEnum` types rather than bare `str`, so a
typo is a validation error instead of a silent new category.

**The four lenses** are the mechanism that makes "why is this paper important and
perspective" concrete instead of flattering. The model must pick the 1–2 that apply
and write to them:

- **capability** — what agents can now do that they could not before;
- **method** — what the field can now measure, build, or compare that it could not;
- **safety** — what risk/oversight implication follows, if any;
- **adoption** — what will plausibly show up in frameworks/products within a year.

If none applies, the honest output is an empty `lenses` list and a low `impact_forecast`
— which is exactly the outcome that should lose to a stronger paper.

#### 7.3 Every rating prints its basis

A score with no stated basis is an assertion. Each digest item therefore carries one compact
line naming the signals that produced it, and `screener explain <arxiv_id>` prints the full
breakdown. The line distinguishes the two halves by construction, because they come from
different places:

```
<b>Basis</b> 7.0 = quality 7.8×0.50 + impact 6.2×0.50
  impact: 41★ repo · 2 citations · no venue yet  (T+14)
  quality: novelty 8 · rigor 7 · relevance 8 · evidence 6 · repro 6
```

Three rules make that honest:

1. **Measured signals show their raw value and age.** `41★`, `2 citations`, `T+14` — never a
   bare number whose provenance the reader has to guess. A signal that was not measured says so
   ("no venue yet") rather than being omitted, so absence is visible as absence.
2. **The quality half is labelled as judgment.** It is prose-derived and the reader is told so;
   nothing about it implies measurement.
3. **No forecast appears anywhere.** With the rating point at T+14 there is no predicted
   impact to disclose, which removes the entire class of "the model thinks this will be big"
   claims from the digest.

The same structure is what `replay` and the maturity reports read, so a rating printed in the
digest can be fully reconstructed later from stored data.

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

**Thinking mode is disabled for scoring, and that is a correctness requirement.** DeepSeek
enables thinking mode by default and it **silently ignores `temperature`** — no error, no
effect. A rubric whose whole justification is "a model is repeatable and a human can audit it"
(§6.1) cannot be built on a sampler that discards its own temperature, so the adapter sends
`{"thinking": {"type": "disabled"}}` on every scoring call, and a unit test asserts the field
is present. Turning it on is a deliberate, visible act (`SCREENER_LLM_THINKING=true`) that
`doctor` reports with the repeatability warning attached.

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
<i>Reply 👍 / 👎 / 🔥 to rate an item and tune tomorrow's ranking.</i>
```

Bounded resurfacing (§6.6.5), when it fires, is its own clearly-labelled block — at most
one item, always carrying the original announcement date:

```
🔁 <b>Second look</b> — published 2025-09-16, missed then, aged well

<b>Trajectory Entropy Predicts Agent Failure</b> <code>2509.11204</code>
Skipped at 5.8 on day 0; now +312 ★ and 24 citations in 90 days.
<a href="https://arxiv.org/abs/2509.11204">abs</a> · <a href="...">code</a>
```

The announcement date is mandatory in this block. A resurfaced paper must never be
mistakable for a new one.

Design rules: each item ≤ ~900 characters; **exactly one paper per message** (§8.4), which
also means splitting always happens on item boundaries; Telegram's 4096-character limit is
asserted in code, not hoped for (§9). Links are always explicit `abs`/`pdf`/`code` — no
auto-previews (`link_preview_options.is_disabled=true`) so the digest stays scannable.

### 8.2 Message strategy

- `parse_mode=HTML` (more forgiving than MarkdownV2 escaping rules).
- **One paper per message** (§8.4). The header rides on the first paper's message and the
  footer on the last, so a digest of *n* picks is *n* messages.
- **Only the first message notifies**; the rest are sent with
  `disable_notification=true`. Without that, one paper per message would mean one phone
  buzz per paper, which is a worse experience than the ambiguity it fixes.
- Optional weekly recap (Sunday) as an editable single message.

### 8.3 Feedback loop (what makes it improve)

**Two mechanisms, both per-user: emoji reactions and written replies.** An earlier draft
concluded reactions were impossible, on the reasoning that Decision 2 pinned delivery to a
personal chat and `message_reaction` updates require the bot to be an *administrator*, which
is a group/channel-only concept. **That premise was wrong about this deployment: the digest
is delivered to a group** (`ai_papers`), where administrator status exists and reactions are
therefore available. Verified against the live API rather than the documentation: the bot was
a plain `member`, and Telegram delivered `message_reaction` updates to it *zero* times until it
is promoted.

So `screener feedback` submits `allowed_updates=["message", "message_reaction"]`. Subscribing
costs nothing while the bot is a member — it simply receives no reaction updates — which means
promoting it needs no code change. Until then the reactions view states the cause instead of
looking broken, and replies are captured either way.

Signals:

- A **reaction** on a message → `feedback` row of kind `'reaction'`, value = the emoji, against
  the single paper that message carried. Because a reaction can be taken back, `new_reaction`
  is treated as the authoritative current set for that user and replaces their previous rows;
  a taken-back 👎 that stayed counted would silently corrupt the reader ranking.
- A **reply** beginning with 👍 / 👎 / 🔥 (optionally followed by text) → kind `'rating'`
  against the resolved paper. Any other free-text reply → stored verbatim as kind `'reply'`; a
  weekly job extracts preference statements ("more theory", "less benchmark-only work") into
  `profile.notes`, injected into the review prompt as soft guidance. Human language becomes
  policy without editing weights by hand.

**An unplaceable reaction is refused, not guessed.** A reaction is attached to a *message*, not
to a region of one, so a message carrying two papers makes the reaction unattributable. Earlier
code returned "first by rank", silently crediting every reaction on a two-paper message to
whichever paper ranked higher — an invisible corruption of exactly the ranking
`most rated papers by users` is built on. Attribution now returns nothing when a message holds
more than one paper, logs `feedback.ambiguous_message` with the candidates, and drops the
update. Dropping is recoverable; misfiling is not. §8.4 removes the cause.

**Impact calibration is no longer a separate weekly job.** An earlier draft refreshed
citations/stars/upvotes for picks from 1/3/6/12 months ago in one weekly batch. That is
now the maturity loop (§6.6): daily, rung-based, and covering gate-passing *rejects* as
well as picks — because a weekly pass over delivered papers can answer "were our picks
good?" and never "what did we miss?". `screener revisit --calibrate` renders the reports
the weekly job used to produce, plus the false-negative audit and the digest-lift
estimate that it could not.

### 8.4 One paper per message

The digest used to pack up to five items per message. In the live database **four of five
messages carried two papers**, which made the majority of reader reactions unplaceable — and
the same ambiguity applied to a reply, since quoting a message quotes all of it.

**One paper per message, therefore, is an attribution requirement rather than a formatting
choice.** It makes a rating identify a paper by construction, for reactions and replies alike,
and it is asserted in code (`test_no_message_ever_carries_two_papers`) rather than assumed. The
cost is a burst of messages in the chat, which `disable_notification` on all but the first
keeps from becoming a burst of notifications.

Historical rows are unaffected and remain readable: the four two-paper messages predate the
change, and their reactions were never captured. Nothing needs migrating — the constraint binds
new digests.

---

## 9. Telegram adapter details

- Send: `POST /bot{token}/sendMessage` with `chat_id`, `text`, `parse_mode=HTML`,
  `link_preview_options.is_disabled=true`, retry on 429 honouring `retry_after`.
- Sending also sets `disable_notification=true` on every message after the first (§8.2), so a
  multi-message digest still announces itself once.
- Feedback polling: `POST /bot{token}/getUpdates` with
  `allowed_updates=["message", "message_reaction"]` and an offset persisted in `kv_state` so
  each update is consumed exactly once. Reaction updates arrive only once the bot is an
  administrator of the group (§8.3).
- Idempotency: message ids are persisted with `(arxiv_id, version, kind)` behind the
  `delivered_once` partial index (§10), so a re-run after a crash cannot double-post.
- Failure: after retries, the rendered digest is written to `outbox/{date}.html` and a
  heartbeat alert fires. The next run **retries the unsent digest automatically** before
  composing a new one, labelling it "previously unsent" — nothing is "offered" to a human,
  because there is no human in the loop (§1.2, *Unattended*). If the outbox entry is older
  than the current window it is sent as its own labelled message and then retired.

---

## 10. Data model (SQLite)

Chosen for zero-ops persistence and exact reproducibility. Written with stdlib
`sqlite3` (WAL mode) and plain `.sql` migrations — no ORM, because the schema is
**fourteen** tables and every query is clearer as SQL.

Three schema-wide rules, because violating each of them is what made the earlier draft
unimplementable:

1. **Runs are created first.** `runs` rows are inserted at the *start* of a run
   (`status='running'`) and updated at the end. Every other table may FK to
   `runs(run_id)` without ordering hazards.
2. **Append-only where history matters.** Assessments and ledger entries are never
   overwritten; "current" is a query, not an UPDATE.
3. **Every JSON column names its model.** No unlabelled `text` blobs.

```sql
CREATE TABLE papers (
  arxiv_id TEXT NOT NULL,             -- version-less base id: '2509.18422'
  version INTEGER NOT NULL,
  title TEXT NOT NULL, abstract TEXT NOT NULL,
  authors TEXT NOT NULL,              -- JSON array[str]
  categories TEXT NOT NULL,           -- JSON array[str]
  primary_category TEXT NOT NULL,
  submitted_at TEXT NOT NULL,         -- ISO-8601 UTC
  abs_url TEXT NOT NULL, pdf_url TEXT NOT NULL,
  comment TEXT, code_url TEXT,
  first_seen_at TEXT NOT NULL,
  PRIMARY KEY (arxiv_id, version)     -- a revised paper is a *new row*, not an update
);
```

**The seen-set is `gate_results`, keyed by `(arxiv_id, version)`** — there is no separate
`seen` table, because a second table could disagree with the first. Semantics, stated
once: *a paper-version is "seen" iff we have ever run the gate on it.* So
`repo.seen()` is `SELECT ... FROM gate_results WHERE (arxiv_id, version) IN (...)`, and
the 5-day sliding window is idempotent: a gate-rejected paper is not re-fetched for 5
days, and a gate-passed-but-unpicked paper is not re-triaged. Choosing "delivered" here
instead would re-review every non-pick five times (~5× cost); choosing "persisted here"
would make `fresh` always empty. This is the one decision v0 cannot defer, because
dedupe is v0's only state.

```sql
-- one row per (paper-version, gate evaluation) — pure input, stored for audit and dedupe
CREATE TABLE gate_results (
  arxiv_id TEXT NOT NULL, version INTEGER NOT NULL,
  run_id TEXT NOT NULL REFERENCES runs(run_id),
  keep INTEGER NOT NULL,
  reason TEXT,                        -- HardFlag when keep=0, else NULL
  hint TEXT NOT NULL,                 -- JSON RelevanceHint {score, matched[], excluded[]} (§5)
  created_at TEXT NOT NULL,
  PRIMARY KEY (arxiv_id, version),
  FOREIGN KEY (arxiv_id, version) REFERENCES papers(arxiv_id, version)
);

CREATE TABLE enrichment (             -- one row per (paper-version, signal snapshot)
  arxiv_id TEXT NOT NULL, version INTEGER NOT NULL,
  source TEXT NOT NULL,               -- 's2' | 'openalex' | 'hf_daily' | 'github'
  payload TEXT NOT NULL, fetched_at TEXT NOT NULL,
  PRIMARY KEY (arxiv_id, version, source, fetched_at),
  FOREIGN KEY (arxiv_id, version) REFERENCES papers(arxiv_id, version)
);

-- Append-only: a re-review adds a row, it never overwrites one. "Current" = latest
-- created_at for (arxiv_id, stage). The PK is idempotent *within* a run, and run_id gives
-- per-run cost attribution. The older PK (arxiv_id, stage, prompt_version) overwrote
-- created_at/model/cost_usd, destroying the score history §3 and §13.5 depend on.
-- Columns mirror the §11 Assessment model one-to-one:
--   payload  <- Assessment.triage | .scores | .verdict   (whichever `stage` populates)
--   review   <- Assessment.review                        (NULL unless stage='review')
--   flags    <- Assessment.soft_flags + .hard_flag
CREATE TABLE assessments (
  arxiv_id TEXT NOT NULL, version INTEGER NOT NULL,
  run_id TEXT NOT NULL REFERENCES runs(run_id),
  stage TEXT NOT NULL,                -- 'triage' | 'review' | 'verify'
  prompt_version TEXT NOT NULL, model TEXT NOT NULL,
  created_at TEXT NOT NULL,
  payload TEXT NOT NULL,              -- JSON: TriageScores | Scores | VerifyVerdict
  review TEXT,                        -- JSON: Review (§7); NULL for triage/verify
  flags TEXT NOT NULL,                -- JSON: {soft: [SoftFlag], hard: HardFlag|null}
  cost_usd REAL NOT NULL,
  PRIMARY KEY (arxiv_id, version, stage, prompt_version, model, run_id),
  FOREIGN KEY (arxiv_id, version) REFERENCES papers(arxiv_id, version)
);
CREATE INDEX assessments_latest ON assessments(arxiv_id, stage, created_at DESC);

CREATE TABLE runs (
  run_id TEXT PRIMARY KEY,
  started_at TEXT NOT NULL, finished_at TEXT,
  status TEXT NOT NULL,               -- 'running'|'ok'|'empty'|'degraded'|'failed'
  mode TEXT NOT NULL DEFAULT 'daily', -- 'daily' | 'weekly'
  stats TEXT NOT NULL, config_hash TEXT NOT NULL, cost_usd REAL NOT NULL DEFAULT 0
);

-- the score history and the audit trail for every digest item (§6.5, §13.5). Written by
-- rank.run(); `components` + `effective_weights` are what make a past ranking explainable
-- months later and a weight change replayable.
CREATE TABLE rankings (
  run_id TEXT NOT NULL REFERENCES runs(run_id),
  arxiv_id TEXT NOT NULL, version INTEGER NOT NULL,
  score REAL NOT NULL,
  components TEXT NOT NULL,           -- JSON: {Dimension: weight_i * dim_i}
  effective_weights TEXT NOT NULL,    -- JSON: the 6- or 8-dimension row actually used
  soft_flags TEXT NOT NULL,           -- JSON array[SoftFlag]
  disposition TEXT NOT NULL,          -- 'eligible' | 'gated'
  hard_flag TEXT,
  topics TEXT NOT NULL,               -- JSON array[Topic]
  lab TEXT, tags TEXT NOT NULL,       -- lab is NULL when enrichment has no affiliation
  created_at TEXT NOT NULL,
  PRIMARY KEY (run_id, arxiv_id, version),
  FOREIGN KEY (arxiv_id, version) REFERENCES papers(arxiv_id, version)
);

-- per-call spend, append-only: the ledger is the audit trail the cap is computed from
CREATE TABLE cost_ledger (
  run_id TEXT NOT NULL REFERENCES runs(run_id),
  seq INTEGER NOT NULL,
  created_at TEXT NOT NULL,
  stage TEXT NOT NULL, model TEXT NOT NULL,
  input_tokens INTEGER NOT NULL, output_tokens INTEGER NOT NULL,
  usd REAL NOT NULL,
  PRIMARY KEY (run_id, seq)
);

CREATE TABLE deliveries (
  run_id TEXT NOT NULL REFERENCES runs(run_id),
  arxiv_id TEXT NOT NULL, version INTEGER NOT NULL,
  kind TEXT NOT NULL,                 -- 'digest' | 'weekly_recap' | 'second_look'
  rank INTEGER NOT NULL, score REAL NOT NULL,
  message_id TEXT, sent_at TEXT,
  PRIMARY KEY (run_id, arxiv_id, version, kind),
  FOREIGN KEY (arxiv_id, version) REFERENCES papers(arxiv_id, version)
);
-- at-most-once per (paper, version) for the *daily digest*. A recap or a dated second
-- look is a different kind and is allowed; a materially revised version is a new row.
-- at-most-once per (paper-version, kind), across *all* kinds: a paper is never delivered
-- twice in a daily digest and never resurfaced twice. A weekly recap and a second look are
-- different kinds, so both remain deliverable for a paper the digest never carried; and a
-- materially revised version is a new (arxiv_id, version) row, so it is deliverable again.
-- The earlier `WHERE kind='digest'` + `ON (arxiv_id)` form blocked the documented roadmap.
CREATE UNIQUE INDEX delivered_once
  ON deliveries(arxiv_id, version, kind)
  WHERE message_id IS NOT NULL;

CREATE TABLE feedback (
  message_id TEXT NOT NULL,           -- the Telegram message the user replied to
  run_id TEXT NOT NULL REFERENCES runs(run_id),
  arxiv_id TEXT NOT NULL,
  kind TEXT NOT NULL,                 -- 'rating' | 'reply'
  value TEXT NOT NULL,                -- '👍' | '👎' | '🔥' | raw reply text
  created_at TEXT NOT NULL,
  PRIMARY KEY (message_id, arxiv_id, kind, value)
);

-- Small namespaced key/value store for single-row state. Two consumers today:
--   'telegram.get_updates_offset' -> exactly-once polling (§9)
--   'revisit.resurface_candidate' -> the <=1 pending second look (§6.6.5), set by the
--      revisit job and consumed by the next digest run
CREATE TABLE kv_state (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL, updated_at TEXT NOT NULL
);
```

`feedback` carries `run_id` because §8.3 promises attribution "against `(arxiv_id, run)`",
and `message_id` alone cannot survive the `deliveries` retention window. The
`message_id ↔ arxiv_id` mapping comes from `compose` (§11): the digest knows which
item landed in which chunk, so the notifier's per-chunk message ids are assigned back to
the papers in that chunk. Without that map there is no way to attribute a reply — which
was the earlier draft's unresolved hole.

```sql
-- ---- maturity loop (§6.6) ----------------------------------------------------
-- enrolment: every gate-passing paper, with the day-0 verdict that will be graded.
-- Keyed on arxiv_id (a paper's trajectory, not a version's), so no composite FK here.
CREATE TABLE watchlist (
  arxiv_id TEXT PRIMARY KEY,
  enrolled_version INTEGER NOT NULL,
  cohort_date TEXT NOT NULL,          -- announcement day it was assessed as new
  score_band TEXT NOT NULL,           -- 'delivered'|'above_min'|'mid'|'low'|'gate_only'
  day0_score REAL,                    -- composite; NULL if never reviewed
  day0_impact_forecast REAL,          -- the rubric dimension; NULL if never reviewed
  delivered INTEGER NOT NULL DEFAULT 0,
  enrolled_at TEXT NOT NULL,
  FOREIGN KEY (arxiv_id, enrolled_version) REFERENCES papers(arxiv_id, version)
);
CREATE INDEX watchlist_cohort ON watchlist(cohort_date, score_band);

-- one row per (paper, rung): a measurement at a fixed age, raw counts verbatim
CREATE TABLE outcomes (
  arxiv_id TEXT NOT NULL REFERENCES watchlist(arxiv_id),
  rung_days INTEGER NOT NULL,         -- 14 | 90 | 180 — the rung, not the actual age
  actual_age_days INTEGER NOT NULL,   -- lateness is expected and recorded (§6.6.2)
  status TEXT NOT NULL,               -- 'measured' | 'missed' (missed = tombstone, counts NULL)
  measured_at TEXT NOT NULL,
  citations INTEGER, influential_citations INTEGER,
  stars INTEGER, hf_upvotes INTEGER,
  venue TEXT, social_mentions INTEGER,
  revisions INTEGER, code_url TEXT,
  matured_impact REAL,                -- 0-10, mapped by config/outcome_scale.yaml
  components_present TEXT NOT NULL,   -- JSON; missing signal != zero signal
  raw TEXT NOT NULL,                  -- JSON per-source payload, for audit
  PRIMARY KEY (arxiv_id, rung_days)
);

-- one row per revisit job execution
CREATE TABLE revisits (
  revisit_id TEXT PRIMARY KEY, run_date TEXT NOT NULL,
  due INTEGER NOT NULL, measured INTEGER NOT NULL, missed INTEGER NOT NULL,
  per_source_errors TEXT NOT NULL,    -- JSON; source drift is loud, not silent
  cost_usd REAL NOT NULL DEFAULT 0, stats TEXT NOT NULL
);

-- calibration drift series: "is the forecast getting better?" across months.
-- The false-negative *ids* live in payload (JSON); the column is the count, because a
-- JSON array cannot be an INTEGER column (an earlier draft conflated the two).
CREATE TABLE calibration (
  report_date TEXT NOT NULL, rung_days INTEGER NOT NULL,
  n INTEGER NOT NULL, spearman REAL, spearman_within_topic REAL, mae REAL,
  false_negative_count INTEGER NOT NULL, payload TEXT NOT NULL,
  PRIMARY KEY (report_date, rung_days)
);
```

**Cross-version references.** `papers` is keyed `(arxiv_id, version)`, so anything that
points at a paper-version uses the composite key (`gate_results`, `enrichment`,
`assessments`, `rankings`, `deliveries`) and anything that tracks a paper's *trajectory*
over time keys on `arxiv_id` alone (`watchlist`, `outcomes`), carrying an
`enrolled_version` for provenance. `feedback.arxiv_id` is deliberately unconstrained:
attribution is a log, and a reply must never be rejected because of an FK to a paper row.

Notes: dry runs insert `deliveries` rows with `message_id IS NULL`, so they are
auditable yet do not consume the at-most-once guarantee. `runs` rows are inserted with
`status='running'` at the start of a run and updated at the end, which is what lets
`deliveries`, `gate_results`, `assessments`, `rankings` and `cost_ledger` carry a non-null
`run_id`. Retention: everything is kept (it is kilobytes/day) except `enrichment.payload`
for papers older than 180 days and `outcomes.raw` once a rung is older than 400 days — the
derived integers and `matured_impact` are kept forever, because they are the labels
(§2, principle 7).

---

## 11. Module contracts

The whole system is these **seven** protocols. Every adapter is swappable, every test uses
fakes, and no vendor type ever crosses a boundary. `Clock` is the seventh: it was listed
in §3 but missing here, and §13.3's fake-clock tests for the rung ladder need it, so
`now` is injected rather than read from the process.

First, the value types those signatures traffic in. The earlier draft used all of these by
name in the code contracts without defining any of them, which is why the ports could not
be typed, let alone implemented:

```python
# domain/models.py — the core records of §4 stages 1–5
class Paper(BaseModel):
    arxiv_id: str                        # version-less base id: '2509.18422'
    version: int                         # arXiv vN; (arxiv_id, version) is the dedupe unit
    title: str; abstract: str
    authors: list[str]; categories: list[str]; primary_category: str
    submitted_at: datetime; abs_url: str; pdf_url: str
    comment: str | None = None; code_url: str | None = None
    first_seen_at: datetime

class RelevanceHint(BaseModel):
    """§5 layer-1 output. Evidence handed to triage, never a decision."""
    score: float
    matched: list[str] = []
    excluded: list[str] = []

class Assessment(BaseModel):
    """One `assessments` row (§10). `stage` decides which payload field is populated —
    which is what the earlier unlabelled `text TEXT` column could not express."""
    arxiv_id: str; version: int; run_id: RunId
    stage: Stage; prompt_version: str; model: str; created_at: datetime
    cost_usd: float
    triage: TriageScores | None = None       # stage='triage'  (§6.3)
    scores: Scores | None = None             # stage='review'  (§6.4)
    review: Review | None = None             # stage='review'  (§7)
    verdict: VerifyVerdict | None = None     # stage='verify'  (§7)
    soft_flags: list[SoftFlag] = []
    hard_flag: HardFlag | None = None

class VerifyVerdict(BaseModel):
    """§7 faithfulness check: every claim traceable to the source text."""
    supported: bool
    unsupported_claims: list[str] = []

class Run(BaseModel):
    """A `runs` row (§10); `of()` is the only constructor the pipeline uses."""
    run_id: RunId; started_at: datetime; finished_at: datetime | None = None
    status: Status; mode: Literal["daily", "weekly"]
    stats: RunStats; config_hash: str; cost_usd: float
    @classmethod
    def of(cls, run_id: RunId, now: datetime, ranked: list[Ranking],
           digest: Digest, cost: float) -> "Run": ...

class DigestItem(BaseModel):
    arxiv_id: str; version: int
    chunk_index: int                     # which Telegram message carries this item

class Digest(BaseModel):
    """compose.run() output (§8). `items` is the item -> chunk map that makes feedback
    attributable (§10): send() returns one id per *chunk*, and only compose knows which
    paper went into which chunk."""
    chunks: list[str]                    # each <= 4096 chars, asserted (§13.3)
    items: list[DigestItem]
    kind: DeliveryKind = "digest"
    second_look: WatchlistEntry | None = None       # <= 1, dated (§6.6.5)
    def message_map(self, ids: Sequence[MessageId]) -> Mapping[str, MessageId]:
        """arxiv_id -> message id of the chunk it landed in. Without this the notifier's
        per-chunk ids could not be joined back to papers, so replies were unattributable."""
        ...

# Each config value type maps to exactly one YAML file, so config has one home per key.
class Profile(BaseModel):                # profile.yaml        (§5)
    name: str
    categories: list[str]; strong_terms: list[str]; weak_terms: list[str]
    exclude_patterns: list[re.Pattern]; boost_topics: list[Topic]
    lookback_days: int = 5
    notes: list[str] = []                # preference statements mined from replies (§8.3)
class Selection(BaseModel): ...          # selection.yaml      (§6.4) — the caps' single home
class TriageWeights(BaseModel): ...      # triage_weights.yaml (§6.3)
class OutcomeScale(BaseModel): ...       # outcome_scale.yaml  (§6.6.3)
class ResurfaceConfig(BaseModel): ...    # revisit.yaml `resurface:` (§6.6.5)
ScoreBand = Literal["delivered", "above_min", "mid", "low", "gate_only"]   # §6.6.1
```

```python
# domain/types.py — scalar aliases and small value types
RunId = NewType("RunId", str)            # uuid4 per run; the `runs` PK
PaperKey = tuple[str, int]               # (arxiv_id, version) — the dedupe unit (§10)
MessageId = NewType("MessageId", str)    # Telegram message id
Stage = Literal["triage", "review", "verify"]
DeliveryKind = Literal["digest", "weekly_recap", "second_look"]
Status = Literal["running", "ok", "empty", "degraded", "failed"]
Rung = Literal[14, 90, 180]              # days after announcement (§6.6.2)
Dimension = Literal[...]                 # §6.4
Topic = Literal[...]                     # §6.4, = profile.boost_topics
Lens = Literal["capability", "method", "safety", "adoption"]   # §7

class Prompt(BaseModel):
    """A versioned prompt *file*, never a string literal (§3). `version` is what lands in
    assessments.prompt_version, so a prompt change is measurable after the fact (§13.5)."""
    name: str                            # 'review' | 'triage' | 'verify' | 'miss_audit'
    version: str                         # 'v2' — parsed from the filename
    body: str
    def render(self, **kw) -> str: ...

class RunStats(BaseModel):
    """Per-stage counts and latencies; the `runs.stats` payload (§12.2)."""
    stage_counts: dict[Stage, int] = {}
    stage_seconds: dict[str, float] = {}
    gated_by_reason: dict[HardFlag, int] = {}
    @classmethod
    def empty(cls) -> "RunStats": ...
    @classmethod
    def of(cls, results, triaged, reviewed, picks, ledger) -> "RunStats": ...

class LedgerEntry(BaseModel):
    """One billed call; the `cost_ledger` row (§10)."""
    stage: Stage; model: str
    input_tokens: int; output_tokens: int; usd: float

class Ledger:
    """Budget tracker. **Not a port**: it has no external system behind it and no adapter,
    so it lives in `domain/ledger.py` as a pure class over a price table plus whatever
    `repository.ledger_add()` has already recorded. It is injected through the composition
    root like a port, which is why `deps.ledger` reads as one — but adding it to the
    protocol list would imply a swap-in adapter that does not and should not exist."""
    spent: float
    status: Status                      # 'ok' until the cap is reached, then 'degraded'
    exhausted: bool
    def soft_cap(self, usd: float) -> ContextManager["Ledger"]: ...
    def affordable(self, next_call: Estimate) -> bool: ...
    def charge(self, entry: LedgerEntry) -> None: ...   # appends to cost_ledger

class Settings(BaseModel):
    """config.py: env + YAML. One field per YAML file, plus env-provided secrets.
    `config_hash` is derived from this and stored on every run."""
    profile: Profile; selection: Selection            # profile.yaml · selection.yaml
    triage_weights: TriageWeights                     # triage_weights.yaml
    outcome_scale: OutcomeScale; revisit: RevisitConfig
    llm_fast: str; llm_deep: str                      # decision 1: triage / review tiers
    llm_fallback: str | None = None                   # §12.1 secondary-provider fallback
    budget_usd: float = 2.0; revisit_budget_usd: float = 0.10
    review_top_k: int = 16; mode: Literal["daily", "weekly"] = "daily"
    dry_run: bool = False; language: str = "en"
    heartbeat_url: str | None = None                  # HEARTBEAT_URL (§12.3)
    @property
    def config_hash(self) -> str: ...
```

```python
# domain/maturity.py — the maturity-loop value types (§6.6), one per §10 table
class WatchlistEntry(BaseModel):
    """A `watchlist` row (§10): a gate-passing paper with its day-0 verdict (§6.6.1)."""
    arxiv_id: str; enrolled_version: int
    cohort_date: date
    score_band: ScoreBand                    # 'delivered'|'above_min'|'mid'|'low'|'gate_only'
    day0_score: float | None = None          # None for gate_only
    day0_impact_forecast: float | None = None
    delivered: bool = False

class OutcomeSignal(BaseModel):
    """Raw probe result for one paper at one rung, before grading (§6.6.3)."""
    arxiv_id: str
    citations: int | None = None; influential_citations: int | None = None
    stars: int | None = None; hf_upvotes: int | None = None
    venue: str | None = None; social_mentions: int | None = None
    revisions: int | None = None; code_url: str | None = None
    sources_ok: list[str] = []               # which probes answered
    raw: dict[str, Any] = {}                 # per-source payload, for audit

class Outcome(BaseModel):
    """A graded rung: an `outcomes` row (§10). status='missed' leaves the signal empty."""
    arxiv_id: str; rung_days: Rung; actual_age_days: int
    status: Literal["measured", "missed"]; measured_at: datetime
    signal: OutcomeSignal
    matured_impact: float | None = None      # 0-10 via outcome_scale (§6.6.3)
    components_present: list[str] = []       # missing signal != zero signal

class CalibrationReport(BaseModel):
    """A `calibration` row plus the audit it produced (§6.6.4)."""
    report_date: date; rung_days: Rung
    n: int; spearman: float | None; spearman_within_topic: float | None; mae: float | None
    false_negative_count: int = 0
    false_negatives: list[str] = []          # arxiv_ids; persisted in `payload` (§10)
    def best_miss(self, cfg: RevisitConfig) -> WatchlistEntry | None: ...   # <= 1 candidate

class RevisitConfig(BaseModel):
    """config/revisit.yaml (§6.6.2). Config only — an execution is a RevisitRun."""
    rungs: list[Rung] = [14, 90, 180]
    max_lateness_days: dict[Rung, int] = {14: 21, 90: 45, 180: 90}
    cohort: dict[ScoreBand, int | Literal["all"]] = {}
    resurface: ResurfaceConfig               # min_matured_impact per band, caps
    audit_prompt: str = "miss_audit.v1.md"
    probe_sources: list[str] = []
    window: tuple[date, date] = ...          # calibration lookback

class RevisitRun(BaseModel):
    """A `revisits` row (§10): one execution of the revisit job."""
    revisit_id: str; run_date: date
    due: int; measured: int; missed: int
    per_source_errors: dict[str, str] = {}   # source -> error; drift is loud, not silent
    cost_usd: float = 0.0
    stats: RunStats
    @classmethod
    def of(cls, revisit_id: str, now: datetime, report: CalibrationReport) -> "RevisitRun": ...
```

Every name used by the ports below is now defined. That was the substance of the
architect's "decisions referenced by name but never defined" finding: the contracts were
readable as prose and unimplementable as code. Note that `Ledger` is deliberately *not* one
of the seven protocols — it is a pure domain class (§ above), and the protocol count refers
to the swappable I/O boundaries only.

```python
# ports.py
class Clock(Protocol):
    def now(self) -> datetime: ...

class PaperSource(Protocol):
    async def fetch(self, since: datetime, until: datetime, profile: Profile) -> list[Paper]: ...

class Enricher(Protocol):
    async def enrich(self, papers: Sequence[Paper]) -> Mapping[str, Enrichment]: ...

class ImpactProbe(Protocol):
    """Measures what happened to a paper at a fixed age. Never on the delivery path."""
    async def measure(self, arxiv_ids: Sequence[str], rung: Rung,
                      now: datetime) -> Mapping[str, OutcomeSignal]: ...

class LLM(Protocol):
    async def parse[T: BaseModel](self, *, model: str, prompt: Prompt, payload: str,
                                  schema: type[T], temperature: float = 0.0) -> T: ...

class Notifier(Protocol):
    async def send(self, chunks: Sequence[str]) -> list[MessageId]: ...   # one id per chunk

class Repository(Protocol):
    # lifecycle: the runs row exists before anything FKs to it
    def begin_run(self, now: datetime, config_hash: str, mode: str) -> RunId: ...
    def finish_run(self, run_id: RunId, status: Status, stats: RunStats, cost_usd: float) -> None: ...
    def begin_revisit(self, now: datetime) -> str: ...
    def finish_revisit(self, run: RevisitRun) -> None: ...

    def seen(self, keys: Sequence[PaperKey]) -> set[PaperKey]: ...   # = gate_results (§10)
    def save_papers(self, papers: Sequence[Paper]) -> None: ...
    def save_gate_results(self, run_id: RunId, results: Sequence[GateResult]) -> None: ...
    def delivered_versions(self, arxiv_ids: Sequence[str]) -> Mapping[str, set[int]]: ...
    def save_assessment(self, run_id: RunId, a: Assessment) -> None: ...
    def latest_assessment(self, key: PaperKey, stage: Stage) -> Assessment | None: ...
    def save_rankings(self, run_id: RunId, ranked: Sequence[Ranking]) -> None: ...   # score history
    def record_delivery(self, run_id: RunId, picks: Sequence[Ranking],
                        kind: DeliveryKind, message_of: Mapping[str, MessageId]) -> None: ...
    def ledger_add(self, run_id: RunId, entry: LedgerEntry) -> None: ...

    # maturity loop (§6.6) — separate command, same store
    def enrol(self, entries: Sequence[WatchlistEntry]) -> None: ...
    def due_rungs(self, now: datetime, cfg: RevisitConfig) -> list[tuple[Rung, list[WatchlistEntry]]]: ...
    def save_outcomes(self, outcomes: Sequence[Outcome]) -> None: ...
    def cohort(self, window: tuple[date, date]) -> list[WatchlistEntry]: ...
    def save_calibration(self, report: CalibrationReport) -> None: ...
    def set_resurface_candidate(self, entry: WatchlistEntry | None) -> None: ...  # -> kv_state
    def take_resurface_candidate(self) -> WatchlistEntry | None: ...              # <- kv_state
```

Two signature changes matter. `unseen(ids) -> set[str]` became
`seen(keys) -> set[PaperKey]` because dedupe is per `(arxiv_id, version)`, and because
returning what *is* seen is the safe direction: a caller that inverts it wrongly looks
empty rather than looking fresh. `record_delivery` now takes `run_id` (a real run id from
`begin_run`, not a `date`), a `kind`, and `message_of` — the `arxiv_id → message_id` map
that `compose` produces and that feedback attribution depends on (§10).

The orchestrator reads as the pipeline it is — this is the file a new engineer opens
first, and it should be readable in one screen:

```python
# pipeline/run.py
async def execute(cfg: Settings, now: datetime) -> Run:
    async with build_deps(cfg) as deps:
        run_id = deps.repo.begin_run(now, cfg.config_hash, cfg.mode)  # runs row first (§10)
        stats = RunStats.empty()                                      # bound for the failure path
        try:
            with deps.ledger.soft_cap(cfg.budget_usd) as ledger:      # soft: never raises on breach
                window = (now - timedelta(days=cfg.profile.lookback_days), now)
                papers = await deps.source.fetch(*window, cfg.profile)
                deps.repo.save_papers(papers)

                window_keys = [(p.arxiv_id, p.version) for p in papers]
                fresh = [p for p in papers
                         if (p.arxiv_id, p.version) not in deps.repo.seen(window_keys)]
                fresh = dedupe_revisions(fresh, deps.repo)   # stateful; keeps gate() pure (§6.2)

                results = [gate(p, cfg.profile) for p in fresh]     # -> GateResult, pure
                deps.repo.save_gate_results(run_id, results)        # this IS the seen-set (§10)
                kept = [r.paper for r in results if r.keep]         # explicit, not truthiness

                enriched = await deps.enricher.enrich(kept)
                triaged = await triage.run(kept, enriched, deps, ledger)
                reviewed = await review.run(top_k(triaged, cfg.review_top_k), deps, ledger)
                verified = await verify.run(reviewed, deps, ledger)   # faithfulness
                ranked = rank.run(verified, cfg)                      # pure
                picks = select(ranked, cfg.selection)                 # pure
                second_look = deps.repo.take_resurface_candidate()    # <=1, dated (§6.6.5)
                digest = compose.run(picks, triaged, now, cfg, second_look)   # pure

                ids = [] if cfg.dry_run else await deps.notifier.send(digest.chunks)
                deps.repo.record_delivery(run_id, picks, kind="digest",
                                          message_of=digest.message_map(ids))   # item -> message
                deps.repo.enrol(enrolment(ranked, triaged, kept, now, cfg))     # §2, principle 7
                stats = RunStats.of(results, triaged, reviewed, picks, ledger)
            deps.repo.finish_run(run_id, ledger.status, stats, ledger.spent)
            return Run.of(run_id, now, ranked, digest, cost=ledger.spent)
        except Exception:
            deps.repo.finish_run(run_id, "failed", stats, deps.ledger.spent)
            raise
```

`digest.message_map(ids)` is the piece the earlier draft was missing: `compose` knows
which item went into which chunk, `send` returns one id per chunk, so the digest can hand
back `arxiv_id -> message_id` and `feedback` becomes attributable (§10).

**The budget cap is soft, and that is the whole point.** The earlier draft used
`async with deps.ledger.cap(...)`, which can only signal breach by raising — yet §12.1
requires *shipping* the verified part of the digest on breach. A context manager that
raises cannot ship anything. `soft_cap` instead sets `ledger.exhausted` and every spending
stage checks `ledger.affordable(next_call)` before its next call: on exhaustion the review
tier stops taking new papers, whatever is verified ships, and the run is recorded
`degraded`. It still raises for genuine provider faults, which is a different failure with
a different response (§12.1).

`gate()` returning a typed `GateResult` fixes three defects at once: the walrus filter
that was a no-op (a Pydantic model is always truthy), the three-way contract conflict
between §4/§5/§11, and §6.2's "recorded with a reason" promise — `reason` is now persisted
in `gate_results`.

```python
# pipeline/revisit.py — separate command, separate failure domain (§6.6)
async def execute_revisit(cfg: Settings, now: datetime) -> RevisitRun:
    async with build_deps(cfg) as deps:                        # no notifier, no digest deps
        revisit_id = deps.repo.begin_revisit(now)              # own row, own bookkeeping
        with deps.ledger.soft_cap(cfg.revisit_budget_usd):     # own small cap (§13.1)
            for rung, cohort in deps.repo.due_rungs(now, cfg.revisit):
                signals = await deps.probe.measure([e.arxiv_id for e in cohort], rung, now)
                deps.repo.save_outcomes(grade(signals, rung, cfg.outcome_scale, now))  # pure
            report = calibrate(deps.repo.cohort(cfg.revisit.window), cfg)              # pure
            deps.repo.save_calibration(report)
            deps.repo.set_resurface_candidate(report.best_miss(cfg.revisit))  # <= 1, dated
    return RevisitRun.of(revisit_id, now, report)
```

Illegal states are unrepresentable: `Assessment` requires scores and text; a `Ranking`
cannot exist without an `Assessment`; `compose.run` cannot emit a chunk longer than the
Telegram limit because a validator asserts it; `Outcome` cannot exist without an enrolled
`WatchlistEntry`, so an outcome signal can never attach to a paper that was never scored;
and a `Ranking` with `disposition="gated"` is filtered by `select()` before compose can
see it.

---

## 12. Reliability, observability, operations

### 12.1 Failure modes and responses

| Failure | Detection | Response |
|---|---|---|
| arXiv API down / slow | timeout or 5xx on all queries | retry w/ backoff, then per-category isolation: **one dead category must not lose the seven that worked**. A total outage raises `SourceUnavailable`, and the run ends with status `failed`, a heartbeat failure, and a one-line notice to the reader ("no digest today — source unavailable"). The first real outage produced a ten-minute run and a 200-line traceback instead; the fetch now also carries a wall-clock budget after which it returns what it has, because a digest from six categories beats one that never finishes |
| arXiv returns empty window | 0 results after gate | send nothing; log `empty` status. Never fabricate content |
| LLM provider outage | circuit breaker on the parse client | fall back to the secondary provider; else **degraded digest**: metadata + abstract quotes, clearly labelled "auto-summary unavailable" |
| LLM returns invalid JSON | Pydantic validation error | one repair retry with the error appended; second failure drops the paper |
| Cost cap approached | `ledger.affordable()` returns false mid-run | **soft** cap (§11): stop taking new papers into review, ship what is verified, record the run `degraded`. The cap is a spending ceiling, not an exception — a raising cap could not ship (§12.1) |
| Telegram send fails | non-2xx after retries | write `outbox/{date}.html`, ping heartbeat, **retry it automatically at the head of the next run** labelled "previously unsent" — never wait for a human (§1.2) |
| Scheduler stops entirely | no heartbeat for >26 h | external monitor (healthchecks.io / Uptime Kuma) alerts by email |
| Duplicate delivery | `delivered_once` partial index rejects | swallowed and logged as `already_delivered`. Unique per `(arxiv_id, version, kind)`, so the daily digest never repeats a paper, a second look is never resurfaced twice, and a weekly recap or a materially revised version is still deliverable |
| Delivery/assessment FK violation | `FOREIGN KEY constraint failed` on insert | cannot happen by construction: `begin_run` creates the `runs` row before any stage writes (§10) |
| Bad prompt version | offline eval suite drops below threshold | CI gate blocks the prompt change (§13.5) |
| Revisit job fails / host asleep | `revisits` row missing for a date, or `missed` rungs | **digest is unaffected by construction** — not on the delivery path. Rungs are due-dated, so a late job measures them late with `actual_age_days` recorded; past `max_lateness_days` the rung is written `missed` rather than back-filled (§6.6.2) |
| Outcome source drift (S2 / OpenAlex / GitHub / HF schema or quota) | `per_source_errors` in the revisit stats | best-effort per source: a dead source drops that component and the grade renormalises, never zeroes it (§6.6.3). GitHub needs a token — unauthenticated is 60 req/h, which a daily cohort exceeds |
| Calibration says the forecast is uninformative | reliability curve flat / Spearman below §1.2 target at a matured rung | record the negative result, cut the `impact` weight, replay against history before shipping (§6.6.4) |
| Outcome data leaking into day-0 scoring | unit test asserting `score()` never reads an `Outcome` (§13.3) | fail the build. This is a correctness invariant, not a style preference (§6.6.6) |

### 12.2 Observability

- Structured JSON logs (`structlog`) with `run_id` bound on every line.
- One `runs` row per run: `status`, `mode`, `config_hash`, total USD, and a `stats` JSON
  blob of per-stage counts and latencies. This is the primary debugging surface.
- The per-call detail the `runs` row deliberately does *not* duplicate lives one table
  over: `cost_ledger` has tokens and model per billed call, and `assessments` has the model
  and `prompt_version` per stage. Normalising this was the fix for the earlier claim that a
  single `runs` row carried "tokens, model ids, prompt versions" — it had no such columns.
- One `revisits` row per revisit job: due/measured/missed counts, `per_source_errors`,
  cost. Rung coverage (`measured / due` per rung) is the health metric that matters — a
  silently stalled probe source looks exactly like "papers stopped getting cited" unless
  this is watched, so it is logged on every run and alerted on two consecutive empties.
- Calibration drift is a first-class series (`calibration` table), so "the forecast is
  getting worse" is a trend line rather than an anecdote.
- `screener doctor`: validates config, Telegram token + chat id (sends a test message),
  LLM credentials, DB writability, arXiv reachability, **probe reachability (S2/OpenAlex/
  GitHub/HF) and `GITHUB_TOKEN` validity**, disk, and clock/timezone.
- `screener replay --date YYYY-MM-DD`: rebuilds a digest from stored data with current
  config, no network, no sends. Mandatory tool for tuning weights.
- `screener backtest --since YYYY-MM-DD`: re-scores stored `rankings`/`assessments` under a
  candidate weight map and reports which papers would have been added or dropped (§6.6.4).
  Offline and manual — it is *not* scheduled, and it is distinct from
  `revisit --calibrate`, which renders maturity statistics rather than replaying policy.
- `screener revisit --calibrate --date YYYY-MM-DD`: re-renders a calibration report from
  stored outcomes with no network — the offline half of the maturity loop.
- `screener eval [--update-notes]`: runs the §13.4 suites against the labelled set and the
  matured cohorts, and mines reply text into `profile.notes`. Weekly via
  `screener-maintenance.timer` (§12.5) — previously an orphan job with no owner.
- `screener prune --older-than 180d`: applies the §10 retention rules. Also weekly, also
  previously an orphan.
- **Heartbeat.** The run ends by pinging `HEARTBEAT_URL` (§12.3) via the `Heartbeat`
  adapter — on success, on `empty`, and on failure alike, since what the external monitor
  (healthchecks.io / Uptime Kuma) detects is *absence*, not badness. `screener revisit`
  and `screener feedback` ping the same URL on their own schedules, so a stalled probe
  source or a dead poll loop is visible rather than looking like "papers stopped getting
  cited". This is the mechanism behind §12.1's "no heartbeat for >26 h" detector, which the
  earlier draft named three times without ever giving it a URL, a config key, or a caller.

### 12.3 Secrets

Env vars / `.env` (git-ignored): `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`,
`OPENAI_API_KEY` / `ANTHROPIC_API_KEY`, optional `SEMANTIC_SCHOLAR_API_KEY`,
`GITHUB_TOKEN` (outcome probes; unauthenticated 60 req/h cannot cover a daily cohort),
`CONTACT_EMAIL` (arXiv/OpenAlex polite-pool User-Agent), and `HEARTBEAT_URL` (the
dead-man's-switch ping target, §12.2 — optional in dev, required in production, and
validated by `screener doctor`). Logs never contain tokens or full message text;
`arxiv_id`s and message ids only.

Note on the reaction constraint (§8.3): the bot needs no admin rights anywhere, because
the design polls `message` updates only. That also means no `TELEGRAM_API_ID`/`API_HASH`
and no MTProto user session — a class of secret and a long-lived second service the
reaction-based design would have required.

### 12.4 Scheduling

arXiv announces **Sunday–Thursday at 20:00 ET** (no Friday/Saturday announcements) and
moderation delays publication by 1–4 days — so the design uses a **5-day sliding
lookback window plus the seen-set**, rather than "since last run". That makes timezone
and announcement timing non-issues: late-announced papers are still caught.

Recommended schedule:

- **Tue–Fri 07:30 ET** — daily digest covering the previous evening's announcement.
- **Mon 07:30 ET** — daily digest covering Sunday night's batch (largest of the week).
- **Daily 08:15 ET — `screener revisit`** — measures whichever rungs are due (§6.6.2).
  Deliberately after the digest and in its own process: it competes with neither the
  digest's latency budget nor the arXiv/Telegram rate limits, and skipping it is safe.
- **Every 15 min — `screener feedback`** — polls `getUpdates` for ratings and replies,
  consuming updates exactly once via the persisted offset (§8.3). Frequent because it is
  cheap, and because a reply that lands 20 hours later is not feedback any more.
- **Sun 09:00 ET** — optional "Week in Agents" recap (surveys allowed, ranked by the
  week's scores and by feedback).
- **Sun 08:00 ET — `screener doctor`**, then `screener eval` and `screener prune`.

**Every scheduled job has an owner.** The earlier draft named five recurring jobs in prose
(feedback sweep, backtest, eval suite, reply mining, 180-day prune) and gave plists to
three, so two of them would simply never have run. The full set is above. `backtest` is
deliberately **not** scheduled — it is a manual offline tool (§12.2), because re-scoring
history is something a human runs while considering a weight change, not something that
should fire on a timer. Reply mining runs inside the weekly `eval` command, because it
edits `profile.notes` and therefore wants the same review step as a weight change.

### 12.5 Deployment and scheduling units

**Target: this Linux host, systemd.** The earlier draft targeted macOS launchd with
`pmset` wake scheduling, but the project lives on Linux (WSL2, openEuler 24.03) where
`launchctl` and `pmset` do not exist and plists cannot be tested. Writing Mac config
blind while developing on Linux is a way to ship scheduling that has never once been
executed, so the units below are `systemd` timers — runnable and verifiable here.
macOS remains possible later; it is a second set of units, not the primary target.

**Every `OnCalendar` carries an explicit timezone.** The schedule is defined in ET
(arXiv's clock), but `systemd` timers default to the *host's* local time, and this host is
not on ET. So each timer uses the timezone suffix — `OnCalendar=Mon..Fri 07:30 America/New_York` —
rather than relying on the machine's `TZ`. Without this, the digest silently drifts by the
host offset and the "previous evening's announcement" claim stops being true. Daylight
saving is then handled by the timezone database, not by us.

Units in `deploy/systemd/` (five timers, one service each):

| Unit | `OnCalendar` (all suffixed `America/New_York`) | Command |
|---|---|---|
| `screener-daily.timer` | `Mon..Fri 07:30` | `screener run --mode daily` |
| `screener-revisit.timer` | `*-*-* 08:15` | `screener revisit` |
| `screener-feedback.timer` | `*:0/15` | `screener feedback` |
| `screener-weekly.timer` | `Sun 09:00` | `screener run --mode weekly` |
| `screener-maintenance.timer` | `Sun 08:00` | `screener doctor && screener eval && screener prune` |

Shared settings for every service: `Type=oneshot`, `User=` the project owner,
`WorkingDirectory=`, `EnvironmentFile=.env`, `StandardOutput=journal`,
`StandardErrorPath` superseded by `journalctl -u`, and crucially:

- **`Persistent=true`** on every timer. This is systemd's equivalent of launchd's
  coalescing-on-wake: a job missed while the machine was suspended runs once on the next
  boot rather than being skipped. Combined with the 5-day lookback and the seen-set, that
  keeps a missed day self-healing (§12.4).
- **`After=network-online.target`**, so a post-boot run does not immediately fail fetch.
- No `Restart=` for the digest — a retry loop around a job that spends money needs the
  budget cap in front of it, and the next scheduled run is the retry. `screener feedback`
  is the exception: `Restart=on-failure` with `RestartSec=30` is safe and desirable.

`--mode` resolves a dangling reference: the earlier draft's launchd plists invoked
`--mode weekly` while the CLI list had no such flag. The CLI is now
`screener run [--mode daily|weekly] | dry-run | replay | feedback | revisit | backtest | doctor | eval | prune`,
matching §3.

Deployment options, in order of preference:

1. **A Linux host with systemd (recommended).** This box, or a $5 VPS with a small
   volume: SQLite needs a persistent disk, and a timer is one file.
2. **Container with a volume** — better uptime than a laptop; adds an image build and a
   volume, nothing else. systemd timers become a supervisor or a container scheduler.
3. **GitHub Actions cron** — free and external, but stateless: SQLite must become Turso
   / libSQL or the state file must be committed to a private repo, and the cron is
   best-effort (can be delayed 5–30 min). Acceptable fallback, not first choice.

---

## 13. Cost, testing, evaluation

### 13.1 Cost model

**The v0 funnel, measured against live arXiv** (2,635 papers over a 5-day window across the
8 prefilter categories, one run):

| Stage | DESIGN.md estimate | **Measured** | Ratio |
|---|---|---|---|
| scanned | ~250 | **2,635** | 10.5× |
| gate-passing | ~50 | **530** (20.1%) | 10.6× |
| reviewed | ~16 | **16** (`review_top_k`) | 1.0× |

Two corrections follow, and both matter for money:

1. **"~250 new papers" conflated per-day volume with per-run volume.** The run reads a
   5-day lookback window, so it scans roughly five days of output from eight categories,
   not one day. The per-run figure is ~2,600, not ~250.
2. **The gate's pass rate is the real cost driver, and it was undefined.** At a threshold of
   1.0 the gate passed 1,035 papers (39%), of which **498 passed on the single word "agent"**
   — precisely the failure mode §1.1 says the design exists to prevent. The
   `min_gate_score: 2.0` default in §5 ("a weak term alone is not enough") brings the gate to
   530 (20.1%) with visibly on-topic survivors.

The `review_top_k = 16` cap is therefore doing all the work of bounding spend: without it v0
would have made 530 LLM calls per run instead of 16. This is why the cap is a first-class
config value rather than a constant, and why v1's triage tier needs a pre-triage bound of its
own — at 530 triage candidates, ~53 batched calls, the cheap tier is no longer cheap.


| Tier | Triage model | Review model | Full text | Est. $/run | Est. $/month |
|---|---|---|---|---|---|
| Lean | small (mini/haiku class) | small | no | ~$0.05 | **~$1–2** |
| Balanced (recommended) | small | mid (sonnet/4o class) | no | ~$0.30–0.60 | **~$8–15** |
| Premium | small | frontier | top 8 only | ~$1.20–1.80 | **~$30–45** |

**Provider pricing (DeepSeek, per 1M tokens).** `deepseek-flash`: input \$0.30 peak /
\$0.15 off-peak on a cache miss, output \$1.20 peak. `deepseek-v4-pro`: input \$1.32,
output \$3.96. Off-peak is half of peak, and peak is only 01:00–04:00 and 06:00–10:00 UTC
Mon–Fri — a 07:30 ET digest run lands in a peak window, so the peak rate is the honest one to
plan against.

**Cache hits are ~50× cheaper than misses** (\$0.006 vs \$0.30 per 1M input tokens on
`deepseek-flash`). The static rubric system prompt (§7) is identical on every call in a run, so
it should hit the KV cache and cost almost nothing — far better than the "−60–80%" this
section originally assumed for generic prompt caching. The ledger deliberately prices every
call at the *cache-miss peak* rate, so it over-estimates spend; that is the safe direction for
a cap, and it means the real bill should come in under the recorded figure.

Levers: prompt caching for the static rubric system prompt (see above — larger than the
original estimate), provider Batch APIs (−50%, acceptable for a once-a-day job), abstract
truncation, and the cascade itself. A `budget_usd` cap per run (default $2) bounds the
ceiling regardless of input volume. Per §11 it is a **soft** cap: on breach the run stops
spending and ships the verified part of the digest as `degraded`. A cap that could only
raise would have no way to deliver the partial digest §12.1 requires.

**The maturity loop is a separate, near-zero-cost budget line (§6.6).** Outcome probes hit
free public APIs; the only LLM spend is the false-negative audit, one cheap call per miss,
typically 0–5 per day. It runs under its own `revisit_budget_usd` cap (default $0.10/run)
so it can never eat the digest's. With the per-band cohort caps of §6.6.2 this is well
under $1/month across all three rungs — the real cost of the loop is storage and
attention, not tokens.

### 13.2 Performance

End-to-end ≤ 4 minutes, budgeted: fetch 20 s, enrich 30 s, triage 40 s, review 150 s
(bounded concurrency 4, `asyncio.TaskGroup`), compose/send 10 s. Well inside any
scheduler window.

The revisit job has no latency budget because nothing waits on it. Ballpark per run:
≤ 6 batched probe calls for a ~50-paper due cohort, well under a minute.

### 13.3 Test strategy

- **Pure unit tests** for `gate`, `score`, `select`, `compose`, `style`, and all of
  `maturity` (rung scheduling, outcome mapping, cohort stats, calibration math) — the
  highest value tests, no network, deterministic.
- **Adapter tests** with `respx`/recorded fixtures: Atom parsing, retry/429 behaviour,
  Telegram chunking, S2/HF/GitHub schema drift and 404s on deleted repos. Never live
  network in CI.
- **Golden-file test** for the rendered digest (HTML mode) so formatting regressions
  are visible in diffs — including the "second look" block.
- **Invariants:** no chunk > 4096 chars; re-running a date never yields new deliveries;
  every numeric claim in output appears in a source quote; **`score()` never reads an
  `Outcome`** (§6.6.6 — a temporal-leakage guard, asserted by test); a resurfaced item
  always carries its original announcement date; re-running a revisit date re-measures
  nothing already recorded; **`select()` never returns a `disposition="gated"` paper**;
  `gate()` is pure (same input, same `GateResult`, no repository access); weights in
  `Ranking.effective_weights` always sum to 1.0 after renormalisation, and so does every
  weight map in config — `sum(§6.1) == 1.0`, `sum(triage_weights) == 1.0`, and
  `sum(v1 effective) == 1.0`. A sum test is cheap and would have caught the `.27/.27/.27/
  .20` slip (§6.3) without a reviewer.)
- **LLM contract tests** against a fake `LLM` port (including malformed JSON, refusals,
  truncated output) to prove degradation paths work.
- **Gate-threshold regression:** a paper scoring 1.0 (one weak term, nothing else) must be
  rejected. This is the single highest-leverage test in the suite: measured on live data,
  removing this rule lets 48% of gate-passers through on the word "agent" (§13.1).
- **Ledger tests** for the soft cap: a run whose next call would exceed `budget_usd` stops
  spending, still ships the verified digest, and records `status='degraded'` — the
  behaviour a raising cap made impossible.
- **Fake-clock tests** for the rung ladder: no rung fires early, a late rung records
  `actual_age_days`, and a rung past `max_lateness_days` is written `missed` rather than
  back-filled (§6.6.2).
- **`v1`-shaped run test**: with enrichment absent, `pedigree`/`early_signal` are `None`
  and the composite renormalises over six dimensions rather than scoring them as zero
  (§6.5). This is the test that keeps the v1/v1.5 boundary honest.

### 13.4 Offline evaluation (the part that makes this a system, not a script)

A hand-labelled set of ~30 historical papers (include / borderline / exclude, plus a
1–10 desirability rating) is the regression suite for the *product*. Run weekly and on
every prompt or weight change, reporting:

- gate precision/recall vs. labels;
- Spearman correlation between composite score and human ratings;
- **impact calibration**: predicted `impact_forecast` vs. `matured_impact` across the
  watchlist cohort at T+14/T+90/T+180 — reliability curve, Spearman, MAE, reported
  overall **and within each topic tag** so a ranker that only predicts "popular subfield"
  cannot look good (§6.6.4). If the forecast is uninformative at a matured rung, the
  `impact` weight is reduced, the change is replayed against history, and the honest
  conclusion is recorded.
- **false-negative audit**: the missed-but-aged-well list (§6.6.4) is a standing eval
  input. Every confirmed miss becomes a regression fixture in the labelled set, which is
  how that set grows past its initial 30 papers without hand-curation effort.

Note the two eval suites answer different questions and neither replaces the other. The
hand-labelled 30 measure *human judgment of relevance and quality* on a fixed set; the
maturity loop measures *outcomes of papers we actually processed*, at a scale and latency
no human labelling can match. The first is available on day one; the second is the only
one that can falsify `impact_forecast`.

### 13.5 Prompt and weight change control

Prompts are versioned files (`review.v2.md`); every `Assessment` row records the
version, so the effect of a prompt change is measurable before it ships. CI runs the
eval suite on a prompt change and fails if precision drops more than 5 points.
Weights are YAML+`config_hash`; `screener replay` shows exactly which papers a weight
change would have added or dropped.

Matured outcomes participate in change control without ever being applied automatically:
`screener revisit --calibrate` emits proposals (weight deltas, gate-term additions,
`min_score` moves) as a reviewable diff, each annotated with the papers it would have
changed. A proposal is merged by a human, which keeps §2, principle 1 intact — LLMs and data produce
features and evidence; the policy stays in versioned config that a person approved.

---

## 14. Security, privacy, legal

- **arXiv API terms.** Verified against the [Terms of Use](https://info.arxiv.org/help/api/tou.html):
  descriptive `User-Agent` including contact email; **"make no more than one request every
  three seconds, and limit requests to a single connection at a time"** — the second clause
  is the binding constraint the earlier draft omitted, so the ~8 daily queries are issued
  **sequentially on one pooled connection**, not parallelised. Parallelising them would be
  a ToU violation, not just a slowdown. Also: one fetch per query per day (the API asks for
  caching, and `updated` only changes at midnight), `max_results` ≤ 2000 per slice, prefer
  `export.arxiv.org` (not the main site), OAI-PMH only if bulk need arises.
- **Request count is the real rate-limit lever, and caching is mandatory.** Measured: at
  `max_results=100` per page the 5-day, 8-category window costs ~30 requests; at 1000 it costs
  **10**. arXiv rate-limits by request, and a day of debugging produced a sustained 429/503
  storm from our own traffic. The ToU's "no need to call more than once a day — please cache" is
  therefore implemented literally: raw per-category results are cached for the calendar day
  (`updated` only changes at midnight), which took a repeat fetch from 33 s to 0.07 s. Adaptive
  backoff doubles the inter-request interval on a 429 and honours `Retry-After`.
- **arXiv metadata is CC0.** Per the same ToU, descriptive metadata (title, abstract,
  authors, identifiers, categories) is released under CC0, so storing, transforming and
  sharing it is explicitly permitted. E-print *content* is not — which is exactly the line
  the next bullet draws.
- **Only abstracts/metadata are sent to LLM providers** — publicly posted content, no
  personal data beyond author names. The full-text tier (v1.5) respects arXiv licenses and
  stores nothing beyond derived notes.
- **No redistribution:** the service publishes original summaries and links; it does not
  mirror PDFs or reproduce long excerpts (quotes are short spans for verification only,
  stored, not delivered).
- **Third-party metrics.** Citation counts, star counts and upvotes are read from public
  APIs (S2/OpenAlex/HF/GitHub) under their terms, cached, and stored as small integers
  with source attribution; they are never redistributed or published. GitHub quota is met
  with an authenticated token, and per-source error rates are logged so a source that
  turns off access degrades quietly instead of silently zeroing a signal.
- **One recipient.** Personal Telegram chat, token scoped to that bot, no public channel
  by default. Rate limits respected (Telegram: ≤ 1 message/s per chat, 4096 chars).

---

## 15. Roadmap

**v0 — walking skeleton (1–3 days).** `fetch → gate → single LLM call → compose →
Telegram`, the **`gate_results` seen-set keyed by `(arxiv_id, version)`** (§10 — v0 cannot
defer this, it is v0's only state), a systemd timer, `doctor`, **plus the `watchlist` and
`outcomes` tables, the probe adapter, watchlist enrolment, and a minimal T+14 rung (stars
and upvotes only, no calibration, no calibration reports)**. Proves the whole path
end to end with deliberately dumb internals — and starts the label clock on day one,
because labels cannot be collected retroactively (§2, principle 7). The rung is
deliberately the cheapest part of §6.6 to build; everything else in that section can wait,
the enrolment cannot.

**v1 — the T+14 rating.** *Moved forward from v1.5 by decision 9:* a rating grounded in
measured evidence is only possible with the signals, so enrichment (Semantic Scholar citations
and venue, GitHub stars, HF upvotes) is part of the core path rather than an optional tier. The
rubric gains its measured half, and every digest item prints its basis (§7.3). Without this the
product is the T+0 forecast this design set out to replace.

**v1.5 — depth + the maturity loop.** Full-text tier for finalists, weekly recap mode, **the
T+90 and T+180 rungs, the calibration reports, the false-negative audit, and `screener revisit`
as its own systemd timer.** The `watchlist`/`outcomes` tables and the probe adapter are not new
here — they exist already. What v1.5 adds is the rest of the ladder, which is what turns the
quality half of the rating into something measurable.

**v2 — calibration-driven tuning + personalization.** Weight/threshold/gate proposals
derived from matured rungs, shipped through `replay` and human review (§6.6.4, §13.5);
bounded resurfacing ("second look") once the miss rate is understood well enough to trust
it; inline-keyboard `callback_query` feedback if the text-emoji loop proves too coarse
(§8.3); learned weights from feedback, profile notes extracted from replies, per-topic
sections, semantic search over history (`/search` in the bot).

**v3 — breadth.** Additional sources (OpenReview, HF Daily, major lab blogs), a static
Markdown/HTML archive, multi-field profiles.

---

## 17. Web UI — browsing what was found, sent and rated

A localhost-only, read-only view over `screener.db` (`screener web`, §17.1). It exists for the
question the digest cannot answer: *what has this thing been doing?* — which papers went out on
which day, what the funnel looked like, and what readers did with it.

### 17.1 Two hard constraints

| Constraint | Why |
|---|---|
| **Read-only** | Every connection opens with SQLite's `mode=ro`. The UI has no write path at all, so browsing cannot corrupt the store the pipeline depends on. It also never migrates: a database below the required schema version is reported as needing `screener stats` (or any pipeline command) rather than being silently upgraded by a viewer. |
| **Localhost only** | There is no authentication, so exposure *is* the security boundary. The default bind is `127.0.0.1`, and a non-loopback address is refused unless `--allow-remote` is passed explicitly — the failure mode of an accidentally-public digest history is worse than the inconvenience of an SSH tunnel. |

Server-rendered Jinja2 with no JavaScript and no build step: the pages work with scripting off,
and there is nothing to compile before reading them.

### 17.2 The views

| Route | Question it answers |
|---|---|
| `/` | Overall statistics: papers fetched, gate pass rate, delivered count, spend, run-status breakdown, readers leaderboard, per-day digest table, and a per-run funnel |
| `/days`, `/day/{date}` | What was delivered on one day, item by item, with each paper's review prose, basis table and reactions; plus that day's runs, funnel and notes |
| `/papers` | Delivered papers over a selectable period, searchable by title, abstract or id |
| `/top?by=score` | Papers **we** rated highest in the period |
| `/top?by=users` | Papers **readers** rated highest in the period |
| `/reactions` | Every reaction and reply, per user and per emoji, with the most-rated papers |
| `/paper/{id}` | One paper's full history: rating arithmetic, measured signals, watchlist, outcome rungs, review, delivery, readers |

### 17.3 Where the views get their numbers

Three decisions that would otherwise produce quietly wrong pages:

1. **Days come from `substr(ts, 1, 10)`, not `date(ts)`.** Timestamps are stored with the local
   UTC offset, and SQLite's `date()` normalises to UTC — so a run at 02:00 local would be filed
   under the previous day. The string prefix is the local date as recorded, which is what a
   reader means by "that day".
2. **Assessments are joined per `run_id`.** Re-arming and re-running reviews the same paper
   again under a new run, and the schema permits that. Without the run filter the join multiplied
   every item: in the live database one paper appeared three times on its day page.
3. **A reader score is `CASE value WHEN …`, weighted by emoji**, not a count. One 🔥 is a stronger
   signal than one 👍, and a 👎 subtracts. `users` counts *distinct people*, because three
   reactions from one enthusiast is not three readers agreeing.

### 17.4 Reader reactions (§8.3)

Both mechanisms are live and both are stored per user: a **written reply** quoting a digest
message, and an **emoji reaction** on it. Both resolve to a paper through `deliveries`, which is
the only record of which chunk carried which item.

One measured constraint: Telegram delivers `message_reaction` updates **only when the bot is an
administrator** of the chat. The subscription is enabled regardless, so reactions begin flowing
the moment the bot is promoted and no code changes then. Until it is, the page says so
explicitly rather than looking broken, and replies are captured either way.

Because a reaction can be taken back, `new_reaction` is treated as the authoritative current set
for that user: each update replaces that user's prior reaction on the message. A taken-back 👎
that stayed counted would silently corrupt the reader ranking.

**Storage.** `feedback` is keyed `(message_id, arxiv_id, kind, value, tg_user_id)` (migration
002). The v1 key omitted the user, so it could not distinguish three readers giving the same 👍
and could not undo one of them. `tg_user_id` is `NOT NULL DEFAULT 0` because SQLite treats NULLs
as distinct in a primary key, which would defeat the uniqueness; 0 means "unknown".

## 16. Decisions

Confirmed by the product owner (these are the defaults the design is built on):

| # | Decision | Confirmed choice | Consequence for the build |
|---|---|---|---|
| 1 | LLM tiering | **Balanced**, on **DeepSeek**: `deepseek-flash` for both tiers at v0 (added in review: was provider-agnostic) | `llm_fast` / `llm_deep` are two configured models behind one `LLM` port. The adapter speaks OpenAI-shaped chat-completions, which DeepSeek serves at `https://api.deepseek.com`. One model serves both tiers at v0; v1 can split them to `deepseek-flash` (triage) + `deepseek-v4-pro` (review) with no code change |
| 2 | Deployment host | **Linux host, via systemd timers** (revised in review: was "this Mac, via launchd") | SQLite stays a local file. `Persistent=true` on every timer gives the missed-job self-healing that launchd's coalescing did; `doctor` becomes a weekly timer. The revision is because the project lives on Linux/WSL2 where `launchctl` and `pmset` do not exist — plists could not be tested (§12.5). macOS units become a later port, not the target |
| 3 | Topic scope | **LLM/agentic systems only** | Negative examples for classical MAS/MARL and agent-based simulation are mandatory in both the gate and the triage prompt |
| 4 | Digest shape | **3–6 quality-gated picks, ~900 chars each; 0 is acceptable** | `selection.max_papers = 6`, `selection.per_topic_cap = 2`, and `min_score = 5.5` — calibrated down from 6.5 after measurement showed 6.5 admitted 0–2 of 16 reviewed papers (§6.6.7) — all in the `selection` block, which is their single home; §5 states explicitly that `per_topic_cap` is *not* a profile key |
| 5 | Full-text review | Deferred to v1.5 (top 8 only) | v1 reviews abstracts; `review.py` accepts an optional `full_text` field from day one so the tier drops in without refactoring |
| 6 | Summary language | English (papers are English) | One generation pass; a `language` config key exists for later |
| 7 | Long-horizon promise check | **Enrol every gate-passing paper; measure at T+14 / T+90 / T+180** (added in review) | New `watchlist`/`outcomes`/`revisits`/`calibration` tables, an `ImpactProbe` port, `screener revisit` as a separate daily job, `config/outcome_scale.yaml`, and `matured_impact` labels that feed calibration — never the day-0 score (§6.6) |
| 8 | What the loop is allowed to change | **Calibration always; resurfacing only under a bounded, dated rule** (added in review) | Digest identity stays "new papers": ≤ 1 resurfaced item per digest, 3 per week, always carrying its original announcement date, outranked by new papers (§6.6.5) |
| 9 | **Rating point** | **T+14, not T+0** (added in review) | The digest ranks the cohort that just reached T+14 (§5.2), so citations, repo stars and venue are measured rather than forecast. Consequences: `enrich` moves from v1.5 into the core path, `impact_forecast` is replaced by `citation_signal`/`repo_signal`/`venue_signal` (§6.1), and every item prints its basis (§7.3). The digest is deliberately a fortnight behind the firehose |
| 10 | **Delivery target and reactions** | Delivery is to a **group** (`ai_papers`), not a personal chat | §8.3 called reactions impossible because it assumed the digest went to a 1:1 DM, where a bot cannot be an administrator. The deployment is a group, so reactions are available once the bot is promoted — verified against the live API, which delivered zero reaction updates while it remained a plain member. Consequences: `allowed_updates` includes `message_reaction`; **one paper per message** (§8.4) so a reaction identifies a paper; and an unplaceable reaction is refused rather than guessed |

### Toolchain prerequisites (verified in this environment)

Two things must be settled before v0 runs anywhere, both verified rather than assumed:

- **Python 3.12+ is required.** The `LLM` port uses PEP 695 generic syntax
  (`async def parse[T: BaseModel](...)`), which needs ≥3.12. This host has only 3.11.6
  system-wide plus a uv-managed 3.13.7; 3.12.11 is available for download. So
  `uv python install 3.12` and pin `requires-python = ">=3.12"` with a
  `.python-version` file. The alternative — rewriting the port to a `TypeVar`-based
  signature — is a real choice, but PEP 695 is the reason 3.12 was chosen in §1, so pin it.
- **`UV_CACHE_DIR` must be set** where the default cache is read-only. On this host
  `/root/.cache/uv` is mounted read-only (verified: `errno 30`, *Read-only file system*),
  and `uv` fails outright until the cache points at a writable path. Putting
  `UV_CACHE_DIR=.uv-cache` in the project env (git-ignored) makes `uv` work and keeps the
  cache alongside the project. This is an environment fact, not a design preference, but
  it belongs in the doc because it is the first thing that breaks a fresh clone.

### Scheduling specifics (decision 2)

The systemd units are described in §12.5. Four properties make the schedule dependable on
a host that may be suspended or rebooted:

- `Persistent=true` on every timer — a missed window runs once at next boot. This replaces
  launchd's coalesce-on-wake and `pmset repeat wake`, neither of which exists on Linux.
- The run is **idempotent and window-based**, not clock-dependent: a 5-day lookback plus
  the `gate_results` seen-set means a missed day self-heals on the next run, and a
  duplicate run delivers nothing.
- `screener revisit` is idempotent per `(arxiv_id, rung)`, so a missed day measures the
  rung late rather than losing it (§6.6.2).
- `screener feedback` is the one timer with `Restart=on-failure`, because it spends no
  money and a stalled poll silently loses feedback.
