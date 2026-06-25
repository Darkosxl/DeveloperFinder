from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

from flask import (
    Flask,
    abort,
    flash,
    g,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from werkzeug.middleware.proxy_fix import ProxyFix

from verified_inviter import config
from verified_inviter.auth import configure_auth, require_auth
from verified_inviter.email_send import send_invite_by_id
from verified_inviter.scheduler import SCHEDULER

app = Flask(__name__)
app.secret_key = config.FLASK_SECRET_KEY
app.config["PREFERRED_URL_SCHEME"] = "https"
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
configure_auth(app)

TRUNCATE_LEN = 140


def _get_db() -> sqlite3.Connection:
    if "db" not in g:
        db_path = Path(config.DB_PATH)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        g.db = sqlite3.connect(db_path)
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA busy_timeout = 30000")
    return g.db


@app.teardown_appcontext
def _close_db(exception: BaseException | None) -> None:
    db = g.pop("db", None)
    if db is not None:
        db.close()


def _fmt_timestamp(value: str | None) -> str | None:
    if not value:
        return None
    # Strip trailing microseconds (e.g. "2024-06-24 20:19:16.123456" -> "2024-06-24 20:19:16")
    if "." in value:
        value = value.split(".", 1)[0]
    return value.replace("T", " ")


def _truncate(text: str | None, length: int = TRUNCATE_LEN) -> str | None:
    if not text:
        return None
    if len(text) <= length:
        return text
    return text[:length].rstrip() + "…"


def get_runs(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        """
        SELECT run_date, started_at, finished_at, candidates_seen, repos_judged,
               invites_drafted, invites_sent, dry_run, error
        FROM runs
        ORDER BY run_date DESC
        """
    ).fetchall()
    runs = []
    for row in rows:
        runs.append(
            {
                "run_date": row["run_date"],
                "started_at": _fmt_timestamp(row["started_at"]),
                "finished_at": _fmt_timestamp(row["finished_at"]),
                "candidates_seen": row["candidates_seen"] or 0,
                "repos_judged": row["repos_judged"] or 0,
                "invites_drafted": row["invites_drafted"] or 0,
                "invites_sent": row["invites_sent"] or 0,
                "dry_run": bool(row["dry_run"]),
                "error": row["error"],
            }
        )
    return runs


def get_summary_stats(conn: sqlite3.Connection) -> dict:
    total_candidates = conn.execute(
        "SELECT COUNT(*) FROM candidates"
    ).fetchone()[0]
    total_invites = conn.execute("SELECT COUNT(*) FROM invites").fetchone()[0]
    sent_invites = conn.execute(
        "SELECT COUNT(*) FROM invites WHERE status = 'sent'"
    ).fetchone()[0]
    drafted_invites = conn.execute(
        "SELECT COUNT(*) FROM invites WHERE status = 'drafted'"
    ).fetchone()[0]
    worth_a_damn = conn.execute(
        "SELECT COUNT(*) FROM technical_verdicts WHERE verdict = 'worth_a_damn'"
    ).fetchone()[0]
    skipped = conn.execute(
        "SELECT COUNT(*) FROM technical_verdicts WHERE verdict = 'skip'"
    ).fetchone()[0]
    return {
        "total_candidates": total_candidates,
        "total_invites": total_invites,
        "sent_invites": sent_invites,
        "drafted_invites": drafted_invites,
        "worth_a_damn": worth_a_damn,
        "skipped": skipped,
    }


def get_candidates(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        """
        SELECT canonical_id, source, github_username, hf_username, display_name,
               first_seen_at, last_seen_at
        FROM candidates
        ORDER BY last_seen_at DESC, first_seen_at DESC
        """
    ).fetchall()
    candidates = []
    for row in rows:
        candidates.append(
            {
                "canonical_id": row["canonical_id"],
                "source": row["source"],
                "github_username": row["github_username"],
                "hf_username": row["hf_username"],
                "display_name": row["display_name"],
                "first_seen_at": _fmt_timestamp(row["first_seen_at"]),
                "last_seen_at": _fmt_timestamp(row["last_seen_at"]),
            }
        )
    return candidates


def get_verdicts(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        """
        SELECT tv.canonical_id, tv.verdict, tv.criteria_met_json, tv.reasoning,
               tv.confidence, tv.created_at, c.display_name
        FROM technical_verdicts tv
        LEFT JOIN candidates c ON c.canonical_id = tv.canonical_id
        ORDER BY tv.created_at DESC
        """
    ).fetchall()
    verdicts = []
    for row in rows:
        criteria = json.loads(row["criteria_met_json"] or "[]")
        reasoning = row["reasoning"]
        verdicts.append(
            {
                "canonical_id": row["canonical_id"],
                "display_name": row["display_name"],
                "verdict": row["verdict"],
                "criteria_list": criteria,
                "confidence": row["confidence"],
                "reasoning": reasoning,
                "reasoning_truncated": _truncate(reasoning),
                "created_at": _fmt_timestamp(row["created_at"]),
            }
        )
    return verdicts


def get_invites(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        """
        SELECT i.id, i.canonical_id, i.recipient_email, i.sender, i.subject,
               i.matched_company, i.status, i.sent_at, i.bounce_count, i.created_at,
               c.display_name
        FROM invites i
        LEFT JOIN candidates c ON c.canonical_id = i.canonical_id
        ORDER BY i.created_at DESC
        """
    ).fetchall()
    invites = []
    for row in rows:
        invites.append(
            {
                "id": row["id"],
                "canonical_id": row["canonical_id"],
                "display_name": row["display_name"],
                "recipient_email": row["recipient_email"],
                "sender": row["sender"],
                "subject": row["subject"],
                "matched_company": row["matched_company"],
                "status": row["status"],
                "sent_at": _fmt_timestamp(row["sent_at"]),
                "bounce_count": row["bounce_count"] or 0,
                "created_at": _fmt_timestamp(row["created_at"]),
            }
        )
    return invites


def get_invite(conn: sqlite3.Connection, invite_id: int) -> dict | None:
    row = conn.execute(
        """
        SELECT i.id, i.canonical_id, i.recipient_email, i.sender, i.subject, i.body,
               i.matched_company, i.status, i.sent_at, i.bounce_count, i.created_at,
               c.display_name
        FROM invites i
        LEFT JOIN candidates c ON c.canonical_id = i.canonical_id
        WHERE i.id = ?
        """,
        (invite_id,),
    ).fetchone()
    if row is None:
        return None
    return {
        "id": row["id"],
        "canonical_id": row["canonical_id"],
        "display_name": row["display_name"],
        "recipient_email": row["recipient_email"],
        "sender": row["sender"],
        "subject": row["subject"],
        "body": row["body"],
        "matched_company": row["matched_company"],
        "status": row["status"],
        "sent_at": _fmt_timestamp(row["sent_at"]),
        "bounce_count": row["bounce_count"] or 0,
        "created_at": _fmt_timestamp(row["created_at"]),
    }


def get_matches(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        """
        SELECT m.canonical_id, m.match_company, m.why, m.confidence, m.created_at,
               c.display_name
        FROM matches m
        LEFT JOIN candidates c ON c.canonical_id = m.canonical_id
        ORDER BY m.created_at DESC
        """
    ).fetchall()
    matches = []
    for row in rows:
        why = row["why"]
        matches.append(
            {
                "canonical_id": row["canonical_id"],
                "display_name": row["display_name"],
                "match_company": row["match_company"],
                "confidence": row["confidence"],
                "why": why,
                "why_truncated": _truncate(why),
                "created_at": _fmt_timestamp(row["created_at"]),
            }
        )
    return matches


@app.route("/")
@require_auth
def overview() -> str:
    conn = _get_db()
    runs = get_runs(conn)
    stats = get_summary_stats(conn)
    return render_template("overview.html", active_tab="overview", runs=runs, stats=stats)


@app.route("/candidates")
@require_auth
def candidates() -> str:
    conn = _get_db()
    rows = get_candidates(conn)
    return render_template("candidates.html", active_tab="candidates", candidates=rows)


@app.route("/verdicts")
@require_auth
def verdicts() -> str:
    conn = _get_db()
    rows = get_verdicts(conn)
    return render_template("verdicts.html", active_tab="verdicts", verdicts=rows)


@app.route("/emails")
@require_auth
def emails() -> str:
    conn = _get_db()
    rows = get_invites(conn)
    return render_template("emails.html", active_tab="emails", invites=rows)


@app.route("/emails/<int:invite_id>")
@require_auth
def email_detail(invite_id: int) -> str:
    conn = _get_db()
    invite = get_invite(conn, invite_id)
    if invite is None:
        abort(404)
    return render_template("email_detail.html", active_tab="emails", invite=invite)


@app.route("/emails/<int:invite_id>/send", methods=["POST"])
@require_auth
def send_email(invite_id: int):
    conn = _get_db()
    success, message = send_invite_by_id(conn, invite_id)
    if success:
        flash(message, "success")
    else:
        flash(message, "error")
    return redirect(url_for("email_detail", invite_id=invite_id))


@app.route("/matches")
@require_auth
def matches() -> str:
    conn = _get_db()
    rows = get_matches(conn)
    return render_template("matches.html", active_tab="matches", matches=rows)


@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/api/scheduler/status")
@require_auth
def scheduler_status():
    return jsonify(SCHEDULER.status())


@app.route("/api/scheduler/start", methods=["POST"])
@require_auth
def scheduler_start():
    SCHEDULER.start()
    return jsonify(SCHEDULER.status())


@app.route("/api/scheduler/stop", methods=["POST"])
@require_auth
def scheduler_stop():
    SCHEDULER.stop()
    return jsonify(SCHEDULER.status())


@app.route("/api/scheduler/run-now", methods=["POST"])
@require_auth
def scheduler_run_now():
    SCHEDULER.run_now()
    return jsonify({"status": "triggered"}), 202


def main() -> None:
    port = int(os.getenv("DASHBOARD_PORT", "5005"))
    app.run(host="127.0.0.1", port=port, debug=True)


if __name__ == "__main__":
    main()
