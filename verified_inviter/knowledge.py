from __future__ import annotations

import json
import sqlite3

from verified_inviter import config
from verified_inviter.llm_client import SambaNovaClient
from verified_inviter.models import Candidate, Knowledge, Repo
from verified_inviter.store import insert_or_replace_knowledge


_SYSTEM = """You are a senior technical recruiter who reads code. You read a developer's GitHub repositories and HuggingFace profile and write a concrete, evidence-based summary of what the developer actually knows. Be specific and avoid hype."""

_USER_TEMPLATE = """GitHub profile:
Username: {github_username}
Display name: {display_name}
Bio: {bio}
Company: {company}
Location: {location}
Public repos: {public_repos}
Followers: {followers}

HuggingFace signal (if any):
{hf_text}

Repositories and their fetched contents (README + key files):
{repos_contents_text}

For each repo, the LLM has already judged whether it is relevant. Only relevant repos are shown above.

Return a JSON object in this exact schema:
{{
  "summary": "2-4 sentences describing the developer's real capabilities and depth",
  "domains": ["webrtc", "systems programming", ...],
  "technologies": ["rust", "pion", "ffmpeg", ...],
  "evidence": [
    {{"repo": "owner/repo-name", "demonstrates": "concrete evidence of a skill"}}
  ]
}}

Do not inflate. Do not hallucinate. If the repositories are shallow, say so. Use repo names and actual content from the provided text. Respond ONLY with a single valid JSON object."""


def _build_repos_contents_text(
    candidate: Candidate,
    relevant_repos: list[Repo],
    contents: list[dict],
) -> str:
    """Build the repos_contents_text section for the knowledge prompt."""
    repo_texts: list[str] = []
    for repo, exa_result in contents:
        exa_text = exa_result.get("_truncated_text", "")
        repo_text = f"""Repo: {repo.owner}/{repo.name}
Language: {repo.language or '(unknown)'}
Stars: {repo.stars}
Fetched content (truncated):
{exa_text}"""
        repo_texts.append(repo_text)

    return "\n\n---\n\n".join(repo_texts)


def _build_hf_text(candidate: Candidate) -> str:
    """Extract a HuggingFace signal string from the candidate profile JSON."""
    profile = candidate.profile_json
    hf_profile = profile.get("hf_profile")
    if not hf_profile:
        return "(none)"
    return json.dumps(hf_profile, default=str, indent=2)


def _build_profile_prompt(
    candidate: Candidate,
    relevant_repos: list[Repo],
    contents: list[dict],
) -> str:
    profile = candidate.profile_json
    return _USER_TEMPLATE.format(
        github_username=candidate.github_username or "(unknown)",
        display_name=candidate.display_name or "(unknown)",
        bio=profile.get("bio") or "(none)",
        company=profile.get("company") or "(none)",
        location=profile.get("location") or "(none)",
        public_repos=profile.get("public_repos", "(unknown)"),
        followers=profile.get("followers", "(unknown)"),
        hf_text=_build_hf_text(candidate),
        repos_contents_text=_build_repos_contents_text(candidate, relevant_repos, contents),
    )


def extract_knowledge(
    llm: SambaNovaClient,
    candidate: Candidate,
    relevant_repos: list[Repo],
    contents: list[dict],
) -> Knowledge:
    """Call the LLM to extract knowledge from the candidate's work."""
    response = llm.chat_json(
        system=_SYSTEM,
        user=_build_profile_prompt(candidate, relevant_repos, contents),
        temperature=config.KNOWLEDGE_TEMP,
    )

    summary = response.get("summary", "") or "(no summary provided)"
    domains = response.get("domains") or []
    technologies = response.get("technologies") or []
    evidence = response.get("evidence") or []

    return Knowledge(
        canonical_id=candidate.canonical_id,
        summary=summary,
        domains=[str(d) for d in domains],
        technologies=[str(t) for t in technologies],
        evidence=[dict(e) for e in evidence],
    )


def extract_knowledge_for_candidate(
    llm: SambaNovaClient,
    store: sqlite3.Connection,
    candidate: Candidate,
    relevant_repos: list[Repo],
    contents: list[dict],
) -> Knowledge:
    """Extract knowledge and persist it to the store."""
    knowledge = extract_knowledge(llm, candidate, relevant_repos, contents)
    insert_or_replace_knowledge(store, knowledge)
    return knowledge
