-- 002: feedback becomes per-user, so "most rated by users" is answerable and two people can
-- react to the same message with the same emoji without colliding.
--
-- The v1 primary key was (message_id, arxiv_id, kind, value): it had no notion of who reacted,
-- so it could not distinguish three users giving the same 👍, and could not undo one of them when
-- a reaction was removed. SQLite cannot alter a primary key, so the table is rebuilt.
--
-- `tg_user_id` is NOT NULL DEFAULT 0 rather than nullable because SQLite treats NULLs as distinct
-- in a primary key, which would silently defeat the uniqueness this table exists for. 0 means
-- "unknown", which in practice only happens for rows migrated from v1.

PRAGMA foreign_keys = OFF;

CREATE TABLE feedback_v2 (
  message_id   TEXT NOT NULL,
  run_id       TEXT NOT NULL REFERENCES runs(run_id),
  arxiv_id     TEXT NOT NULL,
  kind         TEXT NOT NULL,               -- rating | reply | reaction
  value        TEXT NOT NULL,
  tg_user_id   INTEGER NOT NULL DEFAULT 0,  -- 0 = unknown (pre-v2 rows)
  tg_user_name TEXT,
  created_at   TEXT NOT NULL,
  PRIMARY KEY (message_id, arxiv_id, kind, value, tg_user_id)
);

INSERT INTO feedback_v2
  (message_id, run_id, arxiv_id, kind, value, tg_user_id, tg_user_name, created_at)
SELECT message_id, run_id, arxiv_id, kind, value, 0, NULL, created_at FROM feedback;

DROP TABLE feedback;
ALTER TABLE feedback_v2 RENAME TO feedback;

CREATE INDEX IF NOT EXISTS feedback_arxiv ON feedback(arxiv_id, kind);
CREATE INDEX IF NOT EXISTS feedback_user ON feedback(tg_user_id, kind);
CREATE INDEX IF NOT EXISTS feedback_created ON feedback(created_at DESC);

PRAGMA foreign_keys = ON;
