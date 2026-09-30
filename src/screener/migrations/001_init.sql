-- 001_init.sql — the v0 schema (DESIGN.md §10).
--
-- Three rules hold throughout, each of which the earlier draft of the design violated:
--   1. runs rows are inserted with status='running' BEFORE anything FKs to them;
--   2. history (assessments, cost_ledger) is append-only — "current" is a query, not an
--      UPDATE;
--   3. every JSON column names the model it holds.
--
-- The seen-set is gate_results keyed (arxiv_id, version). There is deliberately no
-- separate `seen` table: two tables could disagree, and dedupe is v0's only state.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS papers (
  arxiv_id          TEXT NOT NULL,
  version           INTEGER NOT NULL,
  title             TEXT NOT NULL,
  abstract          TEXT NOT NULL,
  authors           TEXT NOT NULL,          -- JSON array[str]
  categories        TEXT NOT NULL,          -- JSON array[str]
  primary_category  TEXT NOT NULL DEFAULT '',
  submitted_at      TEXT NOT NULL,          -- ISO-8601 UTC
  updated_at        TEXT,
  abs_url           TEXT NOT NULL DEFAULT '',
  pdf_url           TEXT NOT NULL DEFAULT '',
  comment           TEXT,
  code_url          TEXT,
  first_seen_at     TEXT NOT NULL,
  PRIMARY KEY (arxiv_id, version)           -- a revision is a new row, not an update
);

-- One row per (paper-version, gate evaluation). Pure input, stored for audit AND dedupe.
CREATE TABLE IF NOT EXISTS gate_results (
  arxiv_id    TEXT NOT NULL,
  version     INTEGER NOT NULL,
  run_id      TEXT NOT NULL REFERENCES runs(run_id),
  keep        INTEGER NOT NULL,
  reason      TEXT,                         -- HardFlag when keep=0, else NULL
  hint        TEXT NOT NULL,                -- JSON RelevanceHint
  created_at  TEXT NOT NULL,
  PRIMARY KEY (arxiv_id, version),
  FOREIGN KEY (arxiv_id, version) REFERENCES papers(arxiv_id, version)
);

CREATE TABLE IF NOT EXISTS enrichment (
  arxiv_id   TEXT NOT NULL,
  version    INTEGER NOT NULL,
  source     TEXT NOT NULL,
  payload    TEXT NOT NULL,
  fetched_at TEXT NOT NULL,
  PRIMARY KEY (arxiv_id, version, source, fetched_at),
  FOREIGN KEY (arxiv_id, version) REFERENCES papers(arxiv_id, version)
);

-- Append-only. A re-review adds a row; "current" is the latest created_at for
-- (arxiv_id, stage). The PK is idempotent within a run, and run_id gives per-run cost
-- attribution. Columns mirror the Assessment model one-to-one:
--   payload <- Assessment.triage | .scores | .verdict   (whichever `stage` populates)
--   review  <- Assessment.review                        (NULL unless stage='review')
--   flags   <- Assessment.soft_flags + .hard_flag
CREATE TABLE IF NOT EXISTS assessments (
  arxiv_id       TEXT NOT NULL,
  version        INTEGER NOT NULL,
  run_id         TEXT NOT NULL REFERENCES runs(run_id),
  stage          TEXT NOT NULL,
  prompt_version TEXT NOT NULL,
  model          TEXT NOT NULL,
  created_at     TEXT NOT NULL,
  payload        TEXT NOT NULL,
  review         TEXT,
  flags          TEXT NOT NULL,
  cost_usd       REAL NOT NULL DEFAULT 0,
  PRIMARY KEY (arxiv_id, version, stage, prompt_version, model, run_id),
  FOREIGN KEY (arxiv_id, version) REFERENCES papers(arxiv_id, version)
);
CREATE INDEX IF NOT EXISTS assessments_latest ON assessments(arxiv_id, stage, created_at DESC);

CREATE TABLE IF NOT EXISTS runs (
  run_id      TEXT PRIMARY KEY,
  started_at  TEXT NOT NULL,
  finished_at TEXT,
  status      TEXT NOT NULL,                -- running|ok|empty|degraded|failed
  mode        TEXT NOT NULL DEFAULT 'daily',
  stats       TEXT NOT NULL DEFAULT '{}',
  config_hash TEXT NOT NULL DEFAULT '',
  cost_usd    REAL NOT NULL DEFAULT 0
);

-- The score history and per-item audit trail (§6.5, §13.5). components +
-- effective_weights are what make a past ranking explainable months later.
CREATE TABLE IF NOT EXISTS rankings (
  run_id            TEXT NOT NULL REFERENCES runs(run_id),
  arxiv_id          TEXT NOT NULL,
  version           INTEGER NOT NULL,
  score             REAL NOT NULL,
  components        TEXT NOT NULL,          -- JSON {Dimension: weight_i * dim_i}
  effective_weights TEXT NOT NULL,          -- JSON the 6- or 8-dimension row actually used
  soft_flags        TEXT NOT NULL,
  disposition       TEXT NOT NULL,
  hard_flag         TEXT,
  topics            TEXT NOT NULL,
  lab               TEXT,
  tags              TEXT NOT NULL,
  created_at        TEXT NOT NULL,
  PRIMARY KEY (run_id, arxiv_id, version),
  FOREIGN KEY (arxiv_id, version) REFERENCES papers(arxiv_id, version)
);

CREATE TABLE IF NOT EXISTS cost_ledger (
  run_id        TEXT NOT NULL REFERENCES runs(run_id),
  seq           INTEGER NOT NULL,
  created_at    TEXT NOT NULL,
  stage         TEXT NOT NULL,
  model         TEXT NOT NULL,
  input_tokens  INTEGER NOT NULL DEFAULT 0,
  output_tokens INTEGER NOT NULL DEFAULT 0,
  usd           REAL NOT NULL DEFAULT 0,
  PRIMARY KEY (run_id, seq)
);

CREATE TABLE IF NOT EXISTS deliveries (
  run_id     TEXT NOT NULL REFERENCES runs(run_id),
  arxiv_id   TEXT NOT NULL,
  version    INTEGER NOT NULL,
  kind       TEXT NOT NULL,                 -- digest | weekly_recap | second_look
  rank       INTEGER NOT NULL,
  score      REAL NOT NULL,
  message_id TEXT,
  sent_at    TEXT,
  PRIMARY KEY (run_id, arxiv_id, version, kind),
  FOREIGN KEY (arxiv_id, version) REFERENCES papers(arxiv_id, version)
);

-- At-most-once per (paper-version, kind), across all kinds: a paper is never delivered
-- twice in a daily digest and never resurfaced twice. A recap and a second look are
-- different kinds, and a revised version is a new row, so both remain deliverable.
CREATE UNIQUE INDEX IF NOT EXISTS delivered_once
  ON deliveries(arxiv_id, version, kind)
  WHERE message_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS feedback (
  message_id TEXT NOT NULL,
  run_id     TEXT NOT NULL REFERENCES runs(run_id),
  arxiv_id   TEXT NOT NULL,
  kind       TEXT NOT NULL,                 -- rating | reply
  value      TEXT NOT NULL,
  created_at TEXT NOT NULL,
  PRIMARY KEY (message_id, arxiv_id, kind, value)
);

-- Namespaced single-row state: 'telegram.get_updates_offset', 'revisit.resurface_candidate'.
CREATE TABLE IF NOT EXISTS kv_state (
  key        TEXT PRIMARY KEY,
  value      TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

-- ---- maturity loop (§6.6) ----------------------------------------------------------
-- Keyed on arxiv_id: a paper's trajectory, not a version's.
CREATE TABLE IF NOT EXISTS watchlist (
  arxiv_id            TEXT PRIMARY KEY,
  enrolled_version    INTEGER NOT NULL,
  cohort_date         TEXT NOT NULL,
  score_band          TEXT NOT NULL,        -- delivered|above_min|mid|low|gate_only
  day0_score          REAL,
  day0_impact_forecast REAL,
  delivered           INTEGER NOT NULL DEFAULT 0,
  enrolled_at         TEXT NOT NULL,
  FOREIGN KEY (arxiv_id, enrolled_version) REFERENCES papers(arxiv_id, version)
);
CREATE INDEX IF NOT EXISTS watchlist_cohort ON watchlist(cohort_date, score_band);

CREATE TABLE IF NOT EXISTS outcomes (
  arxiv_id             TEXT NOT NULL REFERENCES watchlist(arxiv_id),
  rung_days            INTEGER NOT NULL,
  actual_age_days      INTEGER NOT NULL,    -- lateness is expected and recorded
  status               TEXT NOT NULL,       -- measured | missed
  measured_at          TEXT NOT NULL,
  citations            INTEGER,
  influential_citations INTEGER,
  stars                INTEGER,
  hf_upvotes           INTEGER,
  venue                TEXT,
  social_mentions      INTEGER,
  revisions            INTEGER,
  code_url             TEXT,
  matured_impact       REAL,
  components_present   TEXT NOT NULL DEFAULT '[]',
  raw                  TEXT NOT NULL DEFAULT '{}',
  PRIMARY KEY (arxiv_id, rung_days)
);

CREATE TABLE IF NOT EXISTS revisits (
  revisit_id        TEXT PRIMARY KEY,
  run_date          TEXT NOT NULL,
  due               INTEGER NOT NULL,
  measured          INTEGER NOT NULL,
  missed            INTEGER NOT NULL,
  per_source_errors TEXT NOT NULL DEFAULT '{}',
  cost_usd          REAL NOT NULL DEFAULT 0,
  stats             TEXT NOT NULL DEFAULT '{}'
);

-- Calibration drift series. The false-negative *ids* live in payload; the column is the
-- count, because a JSON array cannot be an INTEGER column.
CREATE TABLE IF NOT EXISTS calibration (
  report_date          TEXT NOT NULL,
  rung_days            INTEGER NOT NULL,
  n                    INTEGER NOT NULL,
  spearman             REAL,
  spearman_within_topic REAL,
  mae                  REAL,
  false_negative_count INTEGER NOT NULL DEFAULT 0,
  payload              TEXT NOT NULL DEFAULT '{}',
  PRIMARY KEY (report_date, rung_days)
);
