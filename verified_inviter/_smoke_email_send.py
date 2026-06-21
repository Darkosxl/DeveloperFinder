"""Smoke test for verified_inviter.email_send."""

from __future__ import annotations

import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

from verified_inviter.email_send import (
    render_template,
    send_invite_resend,
    parse_bounce,
    write_dry_run_draft,
    send_pending_invites,
)
from verified_inviter.models import Candidate, DraftEmail
from verified_inviter import config, store


def make_invite(row_id: int, recipient: str | None = "test@example.com") -> dict:
    return {
        "id": row_id,
        "canonical_id": f"gh:smokeuser{row_id}",
        "ref_token": f"tok{row_id}",
        "recipient_email": recipient,
        "sender": f"{config.SENDER_NAME} <{config.SENDER_EMAIL}>",
        "subject": "Smoke test invite",
        "body": "Merhaba! We liked your work on low-level systems.",
        "email_path": "personalized",
        "matched_company": "Example Co",
    }


def test_render_template() -> None:
    candidate = Candidate(
        canonical_id="gh:smokeuser",
        source="github",
        github_username="smokeuser",
        hf_username=None,
        display_name="Smoke User",
        profile_json={},
        first_seen_at=datetime.now(),
        last_seen_at=datetime.now(),
    )
    text = render_template("Hello!", candidate, "abc123")
    assert "https://exposureai.org/verified?ref=abc123" in text
    assert "Buraya başvur:" in text
    assert "Apply here:" in text
    assert config.SENDER_NAME in text
    assert "Exposure Verified" in text
    print("[OK] render_template")


def test_send_invite_resend_mock() -> None:
    mock_client = MagicMock()
    mock_client.Emails.send.return_value = {"id": "re_123"}
    invite = make_invite(1)
    result = send_invite_resend(mock_client, invite)
    assert result["id"] == "re_123"
    call_args = mock_client.Emails.send.call_args[0][0]
    assert call_args["from"] == f"{config.SENDER_NAME} <{config.SENDER_EMAIL}>"
    assert call_args["to"] == ["test@example.com"]
    assert call_args["subject"] == invite["subject"]
    assert call_args["reply_to"] == config.SENDER_EMAIL
    assert "Apply here:" in call_args["text"]
    print("[OK] send_invite_resend mock")


def test_parse_bounce() -> None:
    soft, reason = parse_bounce(Exception("rate limit exceeded"))
    assert soft is True

    hard, reason = parse_bounce(Exception("invalid_email: not a valid email"))
    assert hard is False

    hard2, _ = parse_bounce(Exception("HTTP 422: recipient rejected"))
    assert hard2 is False

    soft2, _ = parse_bounce(Exception("HTTP 500: internal server error"))
    assert soft2 is True

    print("[OK] parse_bounce")


def test_write_dry_run_draft(tmp_path: Path) -> None:
    import email

    invite = make_invite(2)
    outbox = tmp_path / "outbox" / "draft-2026-01-01"
    write_dry_run_draft(invite, outbox)

    eml_path = outbox / "draft_0002.eml"
    assert eml_path.exists()
    eml_text = eml_path.read_text(encoding="utf-8")
    assert "From:" in eml_text
    assert "To:" in eml_text
    assert "Subject:" in eml_text
    # Valid .eml requires a blank line between headers and body.
    assert "\n\n" in eml_text

    msg = email.message_from_string(eml_text)
    body = msg.get_payload(decode=True).decode("utf-8")
    assert "Apply here:" in body
    assert "Buraya başvur:" in body
    assert config.SENDER_NAME in body

    summary_path = outbox / "summary.md"
    assert summary_path.exists()
    summary = summary_path.read_text(encoding="utf-8")
    assert "tok2" in summary
    assert "Example Co" in summary
    print("[OK] write_dry_run_draft")


def test_send_pending_invites(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    conn = store.init_db(db_path)

    candidate = Candidate(
        canonical_id="gh:smokeuser",
        source="github",
        github_username="smokeuser",
        hf_username=None,
        display_name="Smoke User",
        profile_json={},
        first_seen_at=datetime.now(),
        last_seen_at=datetime.now(),
    )
    store.upsert_candidate(conn, candidate)

    draft1 = DraftEmail(
        canonical_id="gh:smokeuser",
        subject="Test 1",
        body="Body 1",
        email_path="personalized",
        matched_company="Acme",
        ref_token="ref1",
    )
    draft2 = DraftEmail(
        canonical_id="gh:smokeuser",
        subject="Test 2",
        body="Body 2",
        email_path="generic",
        matched_company=None,
        ref_token="ref2",
    )
    draft3 = DraftEmail(
        canonical_id="gh:smokeuser",
        subject="Test 3 no email",
        body="Body 3",
        email_path="generic",
        matched_company=None,
        ref_token="ref3",
    )

    store.create_invite_draft(conn, draft1)
    store.create_invite_draft(conn, draft2)
    store.create_invite_draft(conn, draft3)

    # Set emails on the first two drafts; leave the third as NULL to test the skip path.
    conn.execute(
        "UPDATE invites SET recipient_email = ? WHERE id = 1", ("a@example.com",)
    )
    conn.execute(
        "UPDATE invites SET recipient_email = ? WHERE id = 2", ("b@example.com",)
    )
    conn.commit()

    outbox = tmp_path / "outbox"
    sent, failed = send_pending_invites(
        conn, dry_run=True, outbox_dir=outbox, daily_cap=10
    )
    assert sent == 2
    assert failed == 1

    rows = conn.execute(
        "SELECT id, status FROM invites ORDER BY id"
    ).fetchall()
    assert rows[0]["status"] == "sent"
    assert rows[1]["status"] == "sent"
    assert rows[2]["status"] == "bounced"

    assert (outbox / f"draft-{datetime.now().date().isoformat()}" / "draft_0001.eml").exists()
    assert (outbox / f"draft-{datetime.now().date().isoformat()}" / "draft_0002.eml").exists()
    print("[OK] send_pending_invites")


if __name__ == "__main__":
    test_render_template()
    test_send_invite_resend_mock()
    test_parse_bounce()

    tmp = Path("/tmp/verified_inviter_email_send_smoke")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    test_write_dry_run_draft(tmp)
    test_send_pending_invites(tmp)
    shutil.rmtree(tmp, ignore_errors=True)
    print("\nAll email_send smoke tests passed.")
