"""Resend email sending, dry-run draft writer, and bounce parsing."""

from __future__ import annotations

import email.message
import logging
import random
import re
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

import resend

from verified_inviter import config
from verified_inviter.models import Candidate
from verified_inviter.store import (
    get_invite_by_id,
    list_pending_invites,
    mark_invite_bounced,
    mark_invite_sent,
)

logger = logging.getLogger(__name__)

APPLY_LINK = "https://exposureai.org/verified?ref={ref_token}"


def render_template(body: str, candidate: Candidate | None, ref_token: str) -> str:
    """Wrap the LLM-generated body with the tracking link and signature.

    Parameters
    ----------
    body:
        LLM-generated email body (no link or signature).
    candidate:
        Candidate the invite is for; currently used for any future personalization,
        e.g. greeting by display name.
    ref_token:
        Unique reference token for the apply link.

    Returns
    -------
    The full text of the email, ready for the provider or a .eml file.
    """
    link = APPLY_LINK.format(ref_token=ref_token)
    return (
        f"{body}\n\n"
        "---\n"
        f"Buraya başvur: {link}\n"
        f"Apply here: {link}\n"
        "—\n"
        f"{config.SENDER_NAME}\n"
        "Exposure Verified\n"
    )


def send_invite_resend(client: resend.Client, invite: dict[str, Any]) -> dict[str, Any]:
    """Send an invite using the Resend Python SDK.

    Parameters
    ----------
    client:
        Configured Resend client.
    invite:
        Invite row dict from ``store.list_pending_invites``. Must contain
        ``recipient_email``, ``subject``, ``body``, and ``ref_token``.

    Returns
    -------
    The response dict from ``resend.Emails.send``.
    """
    body = render_template(invite["body"], candidate=None, ref_token=invite["ref_token"])
    params: dict[str, Any] = {
        "from": f"{config.SENDER_NAME} <{config.SENDER_EMAIL}>",
        "to": [invite["recipient_email"]],
        "subject": invite["subject"],
        "reply_to": config.SENDER_EMAIL,
        "text": body,
    }
    logger.info("sending invite via resend", extra={"invite_id": invite["id"], "to": params["to"]})
    return client.Emails.send(params)


def _write_eml(invite: dict[str, Any], outbox_dir: Path) -> Path:
    """Write a single .eml file for an invite draft."""
    body = render_template(invite["body"], candidate=None, ref_token=invite["ref_token"])

    msg = email.message.EmailMessage()
    msg["From"] = invite["sender"] or f"{config.SENDER_NAME} <{config.SENDER_EMAIL}>"
    msg["To"] = invite["recipient_email"] or "undiscovered@example.com"
    msg["Subject"] = invite["subject"]
    msg["Reply-To"] = config.SENDER_EMAIL
    msg.set_content(body)

    filename = f"draft_{invite['id']:04d}.eml"
    eml_path = outbox_dir / filename
    eml_path.write_text(msg.as_string(), encoding="utf-8")
    return eml_path


def write_dry_run_draft(invite: dict[str, Any], outbox_dir: Path) -> None:
    """Write a dry-run draft .eml and append to the run summary.

    Parameters
    ----------
    invite:
        Invite row dict from ``store.list_pending_invites``.
    outbox_dir:
        Directory for the dry run (e.g. ``outbox/draft-YYYY-MM-DD``). Created if
        it does not exist.
    """
    outbox_dir.mkdir(parents=True, exist_ok=True)

    eml_path = _write_eml(invite, outbox_dir)
    link = APPLY_LINK.format(ref_token=invite["ref_token"])

    summary_path = outbox_dir / "summary.md"
    recipient = invite["recipient_email"] or "(no email yet)"
    with summary_path.open("a", encoding="utf-8") as f:
        f.write(
            f"- **id:** {invite['id']} | **canonical_id:** {invite['canonical_id']}\n"
            f"  - **to:** {recipient}\n"
            f"  - **path:** {invite['email_path']}\n"
            f"  - **company:** {invite['matched_company'] or 'none'}\n"
            f"  - **subject:** {invite['subject']}\n"
            f"  - **apply link:** {link}\n"
            f"  - **eml file:** {eml_path.name}\n\n"
        )

    logger.info("dry-run draft written", extra={"invite_id": invite["id"], "eml": str(eml_path)})


def parse_bounce(send_error: Exception) -> tuple[bool, str]:
    """Inspect a Resend send error and classify it as soft or hard bounce.

    Parameters
    ----------
    send_error:
        Exception raised by ``resend.Emails.send`` or an HTTP client.

    Returns
    -------
    ``(is_soft, reason)``. Soft bounces can be retried once; hard bounces are
    permanent.
    """
    text = str(send_error).lower()

    # Soft bounce patterns: transient failures, rate limits, upstream 5xx.
    soft_patterns = [
        "rate limit",
        "too many requests",
        "500",
        "502",
        "503",
        "504",
        "timeout",
        "temporary",
        "unavailable",
        "try again",
        "retry",
    ]

    # Hard bounce patterns: invalid recipient, rejected address, bad domain.
    hard_patterns = [
        "invalid_email",
        "invalid email",
        "not a valid email",
        "recipient rejected",
        "domain not found",
        "no mx record",
        "bad mailbox",
        "user does not exist",
        "address rejected",
    ]

    reason = text

    # Resend's SDK may raise an exception with a JSON payload or a plain string.
    # Check for hard-bounce signals first; those are permanent.
    if any(p in text for p in hard_patterns):
        return False, reason

    if any(p in text for p in soft_patterns):
        return True, reason

    # Default heuristic: any HTTP 4xx (except 429) is hard; 5xx / 429 are soft.
    status_codes = re.findall(r"\b(http\s+)?(\d{3})\b", text)
    for _, code in status_codes:
        if code in {"429"}:
            return True, reason
        if code.startswith("5"):
            return True, reason
        if code.startswith("4"):
            return False, reason

    # Fallback: unknown errors are treated as soft so we retry once.
    return True, reason


def send_invite_by_id(conn, invite_id: int) -> tuple[bool, str]:
    """Send a single drafted invite by ID. Returns (success, message)."""
    invite = get_invite_by_id(conn, invite_id)
    if invite is None:
        return False, "Invite not found"

    status = invite.get("status")
    if status != "drafted":
        return False, f"Cannot send invite with status '{status}'"

    if not invite.get("recipient_email"):
        mark_invite_bounced(conn, invite_id, is_soft=False)
        return False, "No recipient email on file"

    if config.DRY_RUN:
        outbox_dir = config.OUTBOX_DIR / f"draft-{date.today().isoformat()}"
        write_dry_run_draft(invite, outbox_dir)
        mark_invite_sent(conn, invite_id, datetime.now())
        return True, "Dry-run draft written"

    try:
        client = resend.Client(api_key=config.RESEND_API_KEY)
        send_invite_resend(client, invite)
        mark_invite_sent(conn, invite_id, datetime.now())
        return True, "Email sent"
    except Exception as exc:
        is_soft, reason = parse_bounce(exc)
        mark_invite_bounced(conn, invite_id, is_soft)
        return False, f"Send failed: {reason}"


def send_pending_invites(
    conn,
    dry_run: bool,
    outbox_dir: Path,
    daily_cap: int,
) -> tuple[int, int]:
    """Send (or dry-run) up to ``daily_cap`` pending invites for today.

    Parameters
    ----------
    conn:
        SQLite connection.
    dry_run:
        If True, write .eml files and a summary instead of sending.
    outbox_dir:
        Base directory for dry-run outputs. Daily subdirectories are created under
        this path.
    daily_cap:
        Maximum number of sends/drafts to process.

    Returns
    -------
    ``(sent_count, failed_count)``.
    """
    run_date = date.today()
    invites = list_pending_invites(conn, run_date)
    sent_count = 0
    failed_count = 0

    if not invites:
        logger.info("no pending invites for today", extra={"run_date": run_date.isoformat()})
        return sent_count, failed_count

    client = resend.Client(api_key=config.RESEND_API_KEY) if not dry_run else None
    today_dir = outbox_dir / f"draft-{run_date.isoformat()}"

    for invite in invites[:daily_cap]:
        if not invite.get("recipient_email"):
            logger.warning(
                "skipping invite with missing email",
                extra={"invite_id": invite["id"], "canonical_id": invite["canonical_id"]},
            )
            mark_invite_bounced(conn, invite["id"], is_soft=False)
            failed_count += 1
            continue

        if dry_run:
            write_dry_run_draft(invite, today_dir)
            # Mark as sent-ish so the same batch is not re-drafted on the next run.
            mark_invite_sent(conn, invite["id"], datetime.now())
            sent_count += 1
            continue

        try:
            response = send_invite_resend(client, invite)
            logger.info(
                "invite sent",
                extra={"invite_id": invite["id"], "resend_id": response.get("id")},
            )
            mark_invite_sent(conn, invite["id"], datetime.now())
            sent_count += 1
        except Exception as exc:
            is_soft, reason = parse_bounce(exc)
            logger.warning(
                "invite send failed",
                extra={
                    "invite_id": invite["id"],
                    "is_soft": is_soft,
                    "reason": reason,
                },
            )
            mark_invite_bounced(conn, invite["id"], is_soft)
            failed_count += 1

        if not dry_run and invite != invites[:daily_cap][-1]:
            sleep_seconds = random.uniform(60, 180)
            logger.info(
                "jittering before next send",
                extra={"sleep_seconds": round(sleep_seconds, 1)},
            )
            time.sleep(sleep_seconds)

    return sent_count, failed_count
