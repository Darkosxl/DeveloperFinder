from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from verified_inviter import config
from verified_inviter.llm_client import SambaNovaClient
from verified_inviter.models import Candidate, Knowledge, Match
from verified_inviter.store import get_knowledge, insert_or_replace_match


_SYSTEM = """You are a talent matcher for a curated talent network. Given what a developer actually knows, choose at most one company from the provided list that is a strong fit. Do not force a match. If none fit well, say "none"."""

_USER_TEMPLATE = """Developer knowledge summary:
{summary}

Domains: {domains}
Technologies: {technologies}

Evidence of their work:
{evidence_lines}

Exposure companies:
{companies_text}

Each company has: name, what they are building, website. Choose the one best match for this developer's demonstrated expertise, or choose none if nothing genuinely fits.

Return JSON in this exact schema:
{{
  "match": "Company Name" | "none",
  "why": "one sentence tying the developer's evidence to the company's focus",
  "confidence": "high" | "medium" | "low"
}}

Respond ONLY with a single valid JSON object."""


def load_companies(path: Path) -> list[dict]:
    """Load the Exposure companies list from a JSON file."""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Companies file {path} must contain a JSON list")
    return [dict(c) for c in data]


def _build_evidence_lines(knowledge: Knowledge) -> str:
    lines: list[str] = []
    for item in knowledge.evidence:
        repo = item.get("repo", "(unknown)")
        demonstrates = item.get("demonstrates", "(no description)")
        lines.append(f"- {repo}: {demonstrates}")
    return "\n".join(lines) if lines else "(no evidence)"


def _build_companies_text(companies: list[dict]) -> str:
    lines: list[str] = []
    for company in companies:
        name = company.get("name", "(unknown)")
        what = company.get("what_they_are_building", "(unknown)")
        website = company.get("website", "(none)")
        lines.append(f"- {name}: {what} (website: {website})")
    return "\n".join(lines) if lines else "(no companies)"


def _build_matching_prompt(knowledge: Knowledge, companies: list[dict]) -> str:
    return _USER_TEMPLATE.format(
        summary=knowledge.summary,
        domains=", ".join(knowledge.domains) if knowledge.domains else "(none)",
        technologies=", ".join(knowledge.technologies) if knowledge.technologies else "(none)",
        evidence_lines=_build_evidence_lines(knowledge),
        companies_text=_build_companies_text(companies),
    )


def pick_matching_company(
    llm: SambaNovaClient,
    knowledge: Knowledge,
    companies: list[dict],
) -> Match:
    """Call the LLM to pick the best matching company or none."""
    response = llm.chat_json(
        system=_SYSTEM,
        user=_build_matching_prompt(knowledge, companies),
        temperature=config.MATCHING_TEMP,
    )

    raw_match = response.get("match", "none")
    match_company: str | None = None if str(raw_match).lower() == "none" else str(raw_match)
    why = response.get("why", "") or "(no reasoning provided)"
    confidence = response.get("confidence", "low") or "low"

    return Match(
        canonical_id=knowledge.canonical_id,
        match_company=match_company,
        why=why,
        confidence=confidence if confidence in ("high", "medium", "low") else "low",
    )


def run_match_for_candidate(
    llm: SambaNovaClient,
    store: sqlite3.Connection,
    candidate: Candidate,
    companies_path: Path,
) -> Match:
    """Load the candidate's knowledge and the companies list, pick a match, and persist it."""
    knowledge = get_knowledge(store, candidate.canonical_id)
    if knowledge is None:
        raise ValueError(f"No knowledge found for candidate {candidate.canonical_id}")

    companies = load_companies(companies_path)
    match = pick_matching_company(llm, knowledge, companies)
    insert_or_replace_match(store, match)
    return match
