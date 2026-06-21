from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path

from verified_inviter import config
from verified_inviter.models import (
    Candidate,
    DraftEmail,
    Knowledge,
    Match,
    RepoVerdict,
    TechnicalVerdict,
)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS candidates (
  canonical_id     TEXT PRIMARY KEY,
  source           TEXT,
  github_username  TEXT,
  hf_username      TEXT,
  display_name     TEXT,
  first_seen_at    TIMESTAMP,
  last_seen_at     TIMESTAMP,
  raw_payload_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_candidates_gh ON candidates(github_username);
CREATE INDEX IF NOT EXISTS idx_candidates_hf ON candidates(hf_username);

CREATE TABLE IF NOT EXISTS repo_verdicts (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  canonical_id    TEXT REFERENCES candidates(canonical_id) ON DELETE CASCADE,
  repo_name       TEXT,
  relevant        INTEGER,
  domain          TEXT,
  reasoning       TEXT,
  created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_repo_verdicts_candidate ON repo_verdicts(canonical_id);

CREATE TABLE IF NOT EXISTS knowledge (
  canonical_id    TEXT PRIMARY KEY REFERENCES candidates(canonical_id) ON DELETE CASCADE,
  summary         TEXT,
  domains_json    TEXT,
  technologies_json TEXT,
  evidence_json   TEXT,
  created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS technical_verdicts (
  canonical_id    TEXT PRIMARY KEY REFERENCES candidates(canonical_id) ON DELETE CASCADE,
  verdict         TEXT,
  criteria_met_json TEXT,
  reasoning       TEXT,
  seed_stage      INTEGER,
  confidence      TEXT,
  created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS matches (
  canonical_id    TEXT PRIMARY KEY REFERENCES candidates(canonical_id) ON DELETE CASCADE,
  match_company   TEXT,
  why             TEXT,
  confidence      TEXT,
  created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS invites (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  canonical_id    TEXT REFERENCES candidates(canonical_id) ON DELETE CASCADE,
  ref_token       TEXT UNIQUE,
  recipient_email TEXT,
  sender          TEXT,
  subject         TEXT,
  body            TEXT,
  email_path      TEXT,
  matched_company TEXT,
  status          TEXT DEFAULT 'drafted',
  sent_at         TIMESTAMP,
  bounce_count    INTEGER DEFAULT 0,
  created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_invites_status ON invites(canonical_id, status);
CREATE INDEX IF NOT EXISTS idx_invites_token ON invites(ref_token);

CREATE TABLE IF NOT EXISTS runs (
  run_date        DATE PRIMARY KEY,
  started_at      TIMESTAMP,
  finished_at     TIMESTAMP,
  candidates_seen INTEGER DEFAULT 0,
  repos_judged    INTEGER DEFAULT 0,
  invites_drafted INTEGER DEFAULT 0,
  invites_sent    INTEGER DEFAULT 0,
  dry_run         INTEGER DEFAULT 1,
  error           TEXT
);
"""


def init_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


def upsert_candidate(conn: sqlite3.Connection, candidate: Candidate) -> None:
    conn.execute(
        """
        INSERT INTO candidates (canonical_id, source, github_username, hf_username, display_name,
                                first_seen_at, last_seen_at, raw_payload_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(canonical_id) DO UPDATE SET
            last_seen_at = excluded.last_seen_at,
            raw_payload_json = excluded.raw_payload_json
        """,
        (
            candidate.canonical_id,
            candidate.source,
            candidate.github_username,
            candidate.hf_username,
            candidate.display_name,
            candidate.first_seen_at.isoformat(),
            candidate.last_seen_at.isoformat(),
            json.dumps(candidate.profile_json, default=str),
        ),
    )
    conn.commit()


def is_blocked_for_processing(
    conn: sqlite3.Connection, canonical_id: str, skip_rejudge_days: int
) -> bool:
    row = conn.execute(
        "SELECT 1 FROM invites WHERE canonical_id = ? AND status IN ('sent', 'drafted') LIMIT 1",
        (canonical_id,),
    ).fetchone()
    if row:
        return True

    row = conn.execute(
        "SELECT created_at FROM technical_verdicts WHERE canonical_id = ? AND verdict = 'skip' ORDER BY created_at DESC LIMIT 1",
        (canonical_id,),
    ).fetchone()
    if row:
        created_at = datetime.fromisoformat(row["created_at"])
        if (datetime.now() - created_at) < timedelta(days=skip_rejudge_days):
            return True

    return False


def insert_repo_verdicts(conn: sqlite3.Connection, verdicts: list[RepoVerdict]) -> None:
    if not verdicts:
        return
    conn.executemany(
        """
        INSERT INTO repo_verdicts (canonical_id, repo_name, relevant, domain, reasoning)
        VALUES (?, ?, ?, ?, ?)
        """,
        [
            (v.canonical_id, v.repo_name, 1 if v.relevant else 0, v.domain, v.reasoning)
            for v in verdicts
        ],
    )
    conn.commit()


def get_relevant_repos_for(
    conn: sqlite3.Connection, canonical_id: str
) -> list[tuple[str, str, str]]:
    rows = conn.execute(
        "SELECT repo_name, domain, reasoning FROM repo_verdicts "
        "WHERE canonical_id = ? AND relevant = 1 ORDER BY id",
        (canonical_id,),
    ).fetchall()
    return [(r["repo_name"], r["domain"], r["reasoning"]) for r in rows]


def has_any_relevant_repo(conn: sqlite3.Connection, canonical_id: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM repo_verdicts WHERE canonical_id = ? AND relevant = 1 LIMIT 1",
        (canonical_id,),
    ).fetchone()
    return row is not None


def insert_or_replace_knowledge(conn: sqlite3.Connection, knowledge: Knowledge) -> None:
    conn.execute(
        """
        REPLACE INTO knowledge (canonical_id, summary, domains_json, technologies_json, evidence_json)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            knowledge.canonical_id,
            knowledge.summary,
            json.dumps(knowledge.domains),
            json.dumps(knowledge.technologies),
            json.dumps(knowledge.evidence, default=str),
        ),
    )
    conn.commit()


def get_knowledge(conn: sqlite3.Connection, canonical_id: str) -> Knowledge | None:
    row = conn.execute(
        "SELECT summary, domains_json, technologies_json, evidence_json FROM knowledge WHERE canonical_id = ?",
        (canonical_id,),
    ).fetchone()
    if not row:
        return None
    return Knowledge(
        canonical_id=canonical_id,
        summary=row["summary"],
        domains=json.loads(row["domains_json"] or "[]"),
        technologies=json.loads(row["technologies_json"] or "[]"),
        evidence=json.loads(row["evidence_json"] or "[]"),
    )


def insert_or_replace_technical_verdict(
    conn: sqlite3.Connection, verdict: TechnicalVerdict
) -> None:
    conn.execute(
        """
        REPLACE INTO technical_verdicts (canonical_id, verdict, criteria_met_json, reasoning, seed_stage, confidence)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            verdict.canonical_id,
            verdict.verdict,
            json.dumps(verdict.criteria_met),
            verdict.reasoning,
            1 if verdict.seed_stage else 0,
            verdict.confidence,
        ),
    )
    conn.commit()


def get_technical_verdict(
    conn: sqlite3.Connection, canonical_id: str
) -> TechnicalVerdict | None:
    row = conn.execute(
        "SELECT verdict, criteria_met_json, reasoning, seed_stage, confidence FROM technical_verdicts WHERE canonical_id = ?",
        (canonical_id,),
    ).fetchone()
    if not row:
        return None
    return TechnicalVerdict(
        canonical_id=canonical_id,
        verdict=row["verdict"],
        criteria_met=json.loads(row["criteria_met_json"] or "[]"),
        reasoning=row["reasoning"],
        seed_stage=bool(row["seed_stage"]),
        confidence=row["confidence"],
    )


def insert_or_replace_match(conn: sqlite3.Connection, match: Match) -> None:
    conn.execute(
        """
        REPLACE INTO matches (canonical_id, match_company, why, confidence)
        VALUES (?, ?, ?, ?)
        """,
        (match.canonical_id, match.match_company, match.why, match.confidence),
    )
    conn.commit()


def get_match(conn: sqlite3.Connection, canonical_id: str) -> Match | None:
    row = conn.execute(
        "SELECT match_company, why, confidence FROM matches WHERE canonical_id = ?",
        (canonical_id,),
    ).fetchone()
    if not row:
        return None
    return Match(
        canonical_id=canonical_id,
        match_company=row["match_company"],
        why=row["why"],
        confidence=row["confidence"],
    )


def create_invite_draft(conn: sqlite3.Connection, draft: DraftEmail) -> int:
    conn.execute(
        """
        INSERT INTO invites (canonical_id, ref_token, sender, subject, body, email_path, matched_company, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, 'drafted')
        """,
        (
            draft.canonical_id,
            draft.ref_token,
            f"{config.SENDER_NAME} <{config.SENDER_EMAIL}>",
            draft.subject,
            draft.body,
            draft.email_path,
            draft.matched_company,
        ),
    )
    conn.commit()
    return conn.execute("SELECT last_insert_rowid()").fetchone()[0]


def list_pending_invites(conn: sqlite3.Connection, run_date: date) -> list[dict]:
    rows = conn.execute(
        """
        SELECT id, canonical_id, ref_token, recipient_email, sender, subject, body,
               email_path, matched_company, status
        FROM invites
        WHERE status = 'drafted' AND date(created_at) = ?
        ORDER BY id
        """,
        (run_date.isoformat(),),
    ).fetchall()
    return [dict(r) for r in rows]


def mark_invite_sent(conn: sqlite3.Connection, invite_id: int, sent_at: datetime) -> None:
    conn.execute(
        "UPDATE invites SET status = 'sent', sent_at = ? WHERE id = ?",
        (sent_at.isoformat(), invite_id),
    )
    conn.commit()


def mark_invite_bounced(conn: sqlite3.Connection, invite_id: int, is_soft: bool) -> None:
    row = conn.execute(
        "SELECT bounce_count FROM invites WHERE id = ?", (invite_id,)
    ).fetchone()
    if not row:
        return
    new_count = row["bounce_count"] + 1
    if is_soft and new_count <= 1:
        status = "failed"
    else:
        status = "bounced"
    conn.execute(
        "UPDATE invites SET bounce_count = ?, status = ? WHERE id = ?",
        (new_count, status, invite_id),
    )
    conn.commit()


def increment_run_stats(
    conn: sqlite3.Connection, run_date: date, **kwargs: int
) -> None:
    # Ensure a row exists
    conn.execute(
        """
        INSERT INTO runs (run_date, started_at, dry_run)
        VALUES (?, ?, ?)
        ON CONFLICT(run_date) DO NOTHING
        """,
        (run_date.isoformat(), datetime.now().isoformat(), 1 if config.DRY_RUN else 0),
    )
    if kwargs:
        columns = ", ".join(f"{k} = {k} + ?" for k in kwargs)
        values = list(kwargs.values()) + [run_date.isoformat()]
        conn.execute(
            f"UPDATE runs SET {columns} WHERE run_date = ?",  # noqa: S608
            tuple(values),
        )
    conn.commit()


def finish_run(
    conn: sqlite3.Connection, run_date: date, finished_at: datetime, error: str | None
) -> None:
    conn.execute(
        "UPDATE runs SET finished_at = ?, error = ? WHERE run_date = ?",
        (finished_at.isoformat(), error, run_date.isoformat()),
    )
    conn.commit()


def insert_run(conn: sqlite3.Connection, run_date: date, dry_run: bool) -> None:
    conn.execute(
        """
        INSERT INTO runs (run_date, started_at, dry_run)
        VALUES (?, ?, ?)
        ON CONFLICT(run_date) DO UPDATE SET
            started_at = COALESCE(runs.started_at, excluded.started_at)
        """,
        (run_date.isoformat(), datetime.now().isoformat(), 1 if dry_run else 0),
    )
    conn.commit()
