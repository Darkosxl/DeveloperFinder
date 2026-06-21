from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class Candidate:
    canonical_id: str
    source: str
    github_username: str | None
    hf_username: str | None
    display_name: str | None
    profile_json: dict[str, Any]
    first_seen_at: datetime
    last_seen_at: datetime


@dataclass
class Repo:
    owner: str
    name: str
    description: str | None
    language: str | None
    stars: int
    forks: int
    topics: list[str]
    has_readme: bool
    homepage: str | None
    pushed_at: datetime
    created_at: datetime
    is_fork: bool


@dataclass
class RepoVerdict:
    canonical_id: str
    repo_name: str
    relevant: bool
    domain: str
    reasoning: str


@dataclass
class Knowledge:
    canonical_id: str
    summary: str
    domains: list[str]
    technologies: list[str]
    evidence: list[dict]


@dataclass
class TechnicalVerdict:
    canonical_id: str
    verdict: str
    criteria_met: list[str]
    reasoning: str
    seed_stage: bool
    confidence: str


@dataclass
class Match:
    canonical_id: str
    match_company: str | None
    why: str
    confidence: str


@dataclass
class DraftEmail:
    canonical_id: str
    subject: str
    body: str
    email_path: str
    matched_company: str | None
    ref_token: str
