from __future__ import annotations

from datetime import datetime, timedelta, timezone

from verified_inviter import config
from verified_inviter.llm_client import SambaNovaClient
from verified_inviter.models import Candidate, Repo, RepoVerdict


_SYSTEM = """You are a strict technical reviewer for a curated talent platform. You inspect a single GitHub repository and decide whether it represents genuine, domain-expert engineering work. You are skeptical of low-effort projects."""

_USER_TEMPLATE = """Repo name: {repo_name}
Owner: {owner}
Description: {description}
Primary language: {language}
Stars: {stars}
Forks: {forks}
Topics: {topics}
Has README: {has_readme}
Homepage: {homepage}
Last pushed: {pushed_at}
Is fork: {is_fork}

Is this repository genuine technical work showing real domain expertise? Examples of genuine technical work: low-level programming, systems programming, WebRTC, infrastructure/devops, cybersecurity, compilers, distributed systems, embedded/IoT, graphics, databases, machine learning engineering, high-performance networking, reverse engineering, etc.

Examples of non-genuine / low-effort work that should be marked irrelevant: AI-generated FastAPI/Next.js landing pages, tutorial forks, boilerplate clones, dotfiles, pure static marketing sites, NFT/crypto hype pages, one-script experiments with no engineering depth, README-only repos with no code, etc.

Return JSON in this schema:
{{
  "relevant": true | false,
  "domain": "one short phrase: e.g. 'webrtc media server', 'kernel development', 'malware analysis'",
  "reasoning": "one concise sentence explaining the decision"
}}

Respond ONLY with a single valid JSON object."""


def _repo_to_prompt(repo: Repo) -> str:
    return _USER_TEMPLATE.format(
        repo_name=repo.name,
        owner=repo.owner,
        description=repo.description or "(none)",
        language=repo.language or "(unknown)",
        stars=repo.stars,
        forks=repo.forks,
        topics=", ".join(repo.topics) if repo.topics else "(none)",
        has_readme="yes" if repo.has_readme else "no",
        homepage=repo.homepage or "(none)",
        pushed_at=repo.pushed_at.isoformat(),
        is_fork="yes" if repo.is_fork else "no",
    )


def judge_repo(
    llm: SambaNovaClient,
    candidate: Candidate,
    repo: Repo,
) -> RepoVerdict:
    """Call the LLM for a single repo and return a RepoVerdict."""
    response = llm.chat_json(
        system=_SYSTEM,
        user=_repo_to_prompt(repo),
        temperature=config.REPO_JUDGE_TEMP,
    )

    relevant = bool(response.get("relevant", False))
    domain = response.get("domain", "") or "(unknown)"
    reasoning = response.get("reasoning", "") or "(no reasoning provided)"

    return RepoVerdict(
        canonical_id=candidate.canonical_id,
        repo_name=f"{repo.owner}/{repo.name}",
        relevant=relevant,
        domain=domain,
        reasoning=reasoning,
    )


def judge_repos_for_candidate(
    llm: SambaNovaClient,
    candidate: Candidate,
    repos: list[Repo],
) -> list[RepoVerdict]:
    """Filter repos by recency and judge each one individually."""
    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=config.REPO_RECENCY_DAYS)
    fresh_repos = [repo for repo in repos if repo.pushed_at >= cutoff]

    verdicts: list[RepoVerdict] = []
    for repo in fresh_repos:
        verdicts.append(judge_repo(llm, candidate, repo))
    return verdicts
