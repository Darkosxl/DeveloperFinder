from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

from verified_inviter import config
from verified_inviter.llm_client import SambaNovaClient
from verified_inviter.models import Candidate, Knowledge, Repo, RepoVerdict, TechnicalVerdict
from verified_inviter.store import (
    get_knowledge,
    get_relevant_repos_for,
    insert_or_replace_technical_verdict,
)


_SYSTEM = """You are a senior talent partner for Exposure Verified, a curated top-talent platform. You decide whether a developer is a high-potential seed-stage talent worth inviting. Be conservative but fair. Reject mediocrity. We want people who are genuinely good but not yet famous."""

_USER_TEMPLATE = """Candidate profile:
Username: {github_username}
Display name: {display_name}
Bio: {bio}
Followers: {followers}
Public repos: {public_repos}

Extracted knowledge from their actual work:
{summary}
Domains: {domains}
Technologies: {technologies}
Evidence:
{evidence_lines}

All repository metadata (relevant repos marked):
{all_repo_lines}

HuggingFace signal (if any):
{hf_text}

Evaluate against these four criteria. A strong pass on any one is enough; mediocre-tiling across all four is not enough:
1. Open source contributions that matter — meaningful PRs/commits to notable upstream projects (not typo fixes or README edits).
2. Personal projects that are quite technical and ambitious — real engineering depth, not boilerplate or AI-generated slop.
3. Personal projects that have garnered stars / people are actually using — community traction signals real utility.
4. HuggingFace models / fine-tunes / datasets that many people have downloaded or favorited — clear adoption signal.

Also consider the seed-stage lens: someone already famous (e.g. >~1000 followers or viral repos, broad internet recognition) is hard to recruit — we prefer emerging talent who are still building leverage.

Return JSON in this exact schema:
{{
  "verdict": "worth_a_damn" | "skip",
  "criteria_met": ["1", "3"],
  "reasoning": "2-4 sentences explaining the verdict and citing specific evidence or which criteria failed",
  "seed_stage": true | false,
  "confidence": "high" | "medium" | "low"
}}

Respond ONLY with a single valid JSON object."""


def _build_evidence_lines(knowledge: Knowledge) -> str:
    lines: list[str] = []
    for item in knowledge.evidence:
        repo = item.get("repo", "(unknown)")
        demonstrates = item.get("demonstrates", "(no description)")
        lines.append(f"- {repo}: {demonstrates}")
    return "\n".join(lines) if lines else "(no evidence)"


def _build_all_repo_lines(
    all_repos_metadata: list[Repo],
    repo_verdicts: list[RepoVerdict],
) -> str:
    verdict_by_name = {v.repo_name: v for v in repo_verdicts}
    lines: list[str] = []
    for repo in all_repos_metadata:
        name = f"{repo.owner}/{repo.name}"
        verdict = verdict_by_name.get(name)
        relevant_marker = "RELEVANT" if verdict and verdict.relevant else "irrelevant"
        domain = f" [{verdict.domain}]" if verdict and verdict.relevant else ""
        lines.append(
            f"- {name} | {relevant_marker}{domain} | lang={repo.language or '?'} | stars={repo.stars} | forks={repo.forks} | pushed={repo.pushed_at.isoformat()}"
        )
    return "\n".join(lines) if lines else "(no repos)"


def _build_hf_text(candidate: Candidate) -> str:
    hf_profile = candidate.profile_json.get("hf_profile")
    if not hf_profile:
        return "(none)"
    import json

    return json.dumps(hf_profile, default=str, indent=2)


def _build_judge_prompt(
    candidate: Candidate,
    knowledge: Knowledge,
    repo_verdicts: list[RepoVerdict],
    all_repos_metadata: list[Repo],
) -> str:
    profile = candidate.profile_json
    return _USER_TEMPLATE.format(
        github_username=candidate.github_username or "(unknown)",
        display_name=candidate.display_name or "(unknown)",
        bio=profile.get("bio") or "(none)",
        followers=profile.get("followers", "(unknown)"),
        public_repos=profile.get("public_repos", "(unknown)"),
        summary=knowledge.summary,
        domains=", ".join(knowledge.domains) if knowledge.domains else "(none)",
        technologies=", ".join(knowledge.technologies) if knowledge.technologies else "(none)",
        evidence_lines=_build_evidence_lines(knowledge),
        all_repo_lines=_build_all_repo_lines(all_repos_metadata, repo_verdicts),
        hf_text=_build_hf_text(candidate),
    )


def judge_technical_quality(
    llm: SambaNovaClient,
    candidate: Candidate,
    knowledge: Knowledge,
    repo_verdicts: list[RepoVerdict],
    all_repos_metadata: list[Repo],
) -> TechnicalVerdict:
    """Call the technical judge LLM and return a TechnicalVerdict."""
    response = llm.chat_json(
        system=_SYSTEM,
        user=_build_judge_prompt(candidate, knowledge, repo_verdicts, all_repos_metadata),
        temperature=config.TECH_JUDGE_TEMP,
    )

    verdict = response.get("verdict", "skip")
    criteria_met = response.get("criteria_met") or []
    reasoning = response.get("reasoning", "") or "(no reasoning provided)"
    seed_stage = bool(response.get("seed_stage", False))
    confidence = response.get("confidence", "low") or "low"

    return TechnicalVerdict(
        canonical_id=candidate.canonical_id,
        verdict=verdict if verdict in ("worth_a_damn", "skip") else "skip",
        criteria_met=[str(c) for c in criteria_met],
        reasoning=reasoning,
        seed_stage=seed_stage,
        confidence=confidence if confidence in ("high", "medium", "low") else "low",
    )


def run_technical_judge_for_candidate(
    llm: SambaNovaClient,
    store: sqlite3.Connection,
    candidate: Candidate,
    all_repos_metadata: list[Repo],
) -> TechnicalVerdict:
    """Fetch artifacts, run the technical judge, and persist the verdict."""
    knowledge = get_knowledge(store, candidate.canonical_id)
    if knowledge is None:
        raise ValueError(f"No knowledge found for candidate {candidate.canonical_id}")

    repo_verdicts = [
        RepoVerdict(
            canonical_id=candidate.canonical_id,
            repo_name=repo_name,
            relevant=True,
            domain=domain,
            reasoning=reasoning,
        )
        for repo_name, domain, reasoning in get_relevant_repos_for(store, candidate.canonical_id)
    ]

    verdict = judge_technical_quality(
        llm,
        candidate,
        knowledge,
        repo_verdicts,
        all_repos_metadata,
    )
    insert_or_replace_technical_verdict(store, verdict)
    return verdict


def _outbox_rejects_path() -> Path:
    today = datetime.now().strftime("%Y-%m-%d")
    outbox = config.OUTBOX_DIR / f"draft-{today}"
    outbox.mkdir(parents=True, exist_ok=True)
    return outbox / "rejects.md"


def append_reject_to_outbox(
    canonical_id: str,
    github_username: str | None,
    verdict: TechnicalVerdict,
) -> None:
    """Append a reject line to the dry-run rejects.md log."""
    path = _outbox_rejects_path()
    criteria = ", ".join(verdict.criteria_met) if verdict.criteria_met else "(none)"
    line = f"- {canonical_id} | {github_username or '(unknown)'} | verdict: {verdict.verdict} | criteria: {criteria} | {verdict.reasoning}\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(line)
