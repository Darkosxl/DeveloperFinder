from __future__ import annotations

import json
import secrets
import sqlite3

from verified_inviter import config
from verified_inviter.llm_client import SambaNovaClient
from verified_inviter.models import Candidate, DraftEmail, Knowledge, Match
from verified_inviter.store import create_invite_draft, get_knowledge, get_match


_PATH_A_SYSTEM = """You are a warm, personalized outreach writer for Exposure Verified, a curated network of top Turkish technologists. You write a single email in Turkish and English mixed. The tone is human, respectful, and specific. You reference the developer's actual work and the matched company."""

_PATH_A_USER_TEMPLATE = """Developer:
Username: {github_username}
Display name: {display_name}
Bio: {bio}

What they actually know:
{summary}

Domains: {domains}
Technologies: {technologies}

Evidence of their work:
{evidence_lines}

Matched company:
Name: {company_name}
What they are building: {what_they_are_building}
Website: {website}

Why this is a match: {match_why}

Write a cold-outreach email inviting this developer to apply to Exposure Verified (exposureai.org/verified). The body should be in Turkish and English mixed: open with a warm Turkish sentence, then continue in English with the technical details, and close with a friendly Turkish sign-off.

Mention 1–2 specific things from their work that caught your attention. Mention the matched company naturally as one place in the Exposure network where they could find peers.

Do not include the apply link; the system will append it. Do not use overly sales-y language. Keep it under 180 words.

Return JSON in this exact schema:
{{
  "subject": "email subject line, one line, no markdown",
  "body": "the full email body text, Turkish+English mixed"
}}

Respond ONLY with a single valid JSON object."""

_PATH_B_SYSTEM = """You are a warm outreach writer for Exposure Verified, a curated network of top Turkish technologists. You write a single email in Turkish and English mixed. The tone is human, respectful, and domain-aware even without a specific company match."""

_PATH_B_USER_TEMPLATE = """Developer:
Username: {github_username}
Display name: {display_name}
Bio: {bio}

What they actually know:
{summary}

Domains: {domains}
Technologies: {technologies}

Evidence of their work:
{evidence_lines}

Write a cold-outreach email inviting this developer to apply to Exposure Verified (exposureai.org/verified). No specific company matched, so keep the email warm and generic: mention that we are a curated network of top Turkish technologists and that their domain stands out.

The body should be in Turkish and English mixed: open with a warm Turkish sentence, continue in English, close with a friendly Turkish sign-off.

Do not include the apply link; the system will append it. Keep it under 150 words.

Return JSON in this exact schema:
{{
  "subject": "email subject line, one line, no markdown",
  "body": "the full email body text, Turkish+English mixed"
}}

Respond ONLY with a single valid JSON object."""


def generate_ref_token() -> str:
    """Return a 16-character URL-safe random token."""
    return secrets.token_urlsafe(12)


def _build_evidence_lines(knowledge: Knowledge) -> str:
    lines: list[str] = []
    for item in knowledge.evidence:
        repo = item.get("repo", "(unknown)")
        demonstrates = item.get("demonstrates", "(no description)")
        lines.append(f"- {repo}: {demonstrates}")
    return "\n".join(lines) if lines else "(no evidence)"


def _find_company_entry(companies: list[dict], match_company: str) -> dict:
    for company in companies:
        if company.get("name") == match_company:
            return company
    return {"name": match_company, "what_they_are_building": "", "website": ""}


def draft_personalized_email(
    llm: SambaNovaClient,
    candidate: Candidate,
    knowledge: Knowledge,
    match: Match,
    companies: list[dict],
    ref_token: str,
) -> DraftEmail:
    """Path A: draft a personalized email using the matched company."""
    company = _find_company_entry(companies, match.match_company or "")
    profile = candidate.profile_json

    user_prompt = _PATH_A_USER_TEMPLATE.format(
        github_username=candidate.github_username or "(unknown)",
        display_name=candidate.display_name or "(unknown)",
        bio=profile.get("bio") or "(none)",
        summary=knowledge.summary,
        domains=", ".join(knowledge.domains) if knowledge.domains else "(none)",
        technologies=", ".join(knowledge.technologies) if knowledge.technologies else "(none)",
        evidence_lines=_build_evidence_lines(knowledge),
        company_name=company.get("name", "(unknown)"),
        what_they_are_building=company.get("what_they_are_building", "(unknown)"),
        website=company.get("website", "(none)"),
        match_why=match.why,
    )

    response = llm.chat_json(
        system=_PATH_A_SYSTEM,
        user=user_prompt,
        temperature=config.EMAIL_TEMP,
    )

    return DraftEmail(
        canonical_id=candidate.canonical_id,
        subject=response.get("subject", "") or "Invitation to Exposure Verified",
        body=response.get("body", "") or "(no body generated)",
        email_path="personalized",
        matched_company=match.match_company,
        ref_token=ref_token,
    )


def draft_generic_email(
    llm: SambaNovaClient,
    candidate: Candidate,
    knowledge: Knowledge,
    ref_token: str,
) -> DraftEmail:
    """Path B: draft a generic but domain-aware email."""
    profile = candidate.profile_json

    user_prompt = _PATH_B_USER_TEMPLATE.format(
        github_username=candidate.github_username or "(unknown)",
        display_name=candidate.display_name or "(unknown)",
        bio=profile.get("bio") or "(none)",
        summary=knowledge.summary,
        domains=", ".join(knowledge.domains) if knowledge.domains else "(none)",
        technologies=", ".join(knowledge.technologies) if knowledge.technologies else "(none)",
        evidence_lines=_build_evidence_lines(knowledge),
    )

    response = llm.chat_json(
        system=_PATH_B_SYSTEM,
        user=user_prompt,
        temperature=config.EMAIL_TEMP,
    )

    return DraftEmail(
        canonical_id=candidate.canonical_id,
        subject=response.get("subject", "") or "Invitation to Exposure Verified",
        body=response.get("body", "") or "(no body generated)",
        email_path="generic",
        matched_company=None,
        ref_token=ref_token,
    )


def draft_email_for_candidate(
    llm: SambaNovaClient,
    store: sqlite3.Connection,
    candidate: Candidate,
    companies: list[dict],
) -> DraftEmail:
    """Determine path A/B from the stored Match and draft the email."""
    knowledge = get_knowledge(store, candidate.canonical_id)
    if knowledge is None:
        raise ValueError(f"No knowledge found for candidate {candidate.canonical_id}")

    match = get_match(store, candidate.canonical_id)
    if match is None:
        # If no match record exists, treat as generic path.
        match = Match(
            canonical_id=candidate.canonical_id,
            match_company=None,
            why="",
            confidence="low",
        )

    ref_token = generate_ref_token()

    if match.match_company:
        draft = draft_personalized_email(
            llm,
            candidate,
            knowledge,
            match,
            companies,
            ref_token,
        )
    else:
        draft = draft_generic_email(
            llm,
            candidate,
            knowledge,
            ref_token,
        )

    create_invite_draft(store, draft)
    return draft
