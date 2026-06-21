from __future__ import annotations

import datetime
import inspect
import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import verified_inviter.relevance
import verified_inviter.content_fetch
import verified_inviter.knowledge
import verified_inviter.technical_judge
import verified_inviter.matching
import verified_inviter.email_draft
from verified_inviter import config
from verified_inviter.models import Candidate, DraftEmail, Knowledge, Match, Repo, RepoVerdict, TechnicalVerdict
from verified_inviter.store import init_db


class FakeSambaNovaClient:
    """Mock LLM client that returns preconfigured JSON responses."""

    def __init__(self, responses: list[dict] | None = None) -> None:
        self.responses = responses or []
        self.calls: list[tuple[str, str, float]] = []
        self._index = 0

    def chat_json(self, system: str, user: str, temperature: float, max_retries: int = 2) -> dict:
        self.calls.append((system, user, temperature))
        response = self.responses[self._index % len(self.responses)]
        self._index += 1
        return response


def _make_candidate() -> Candidate:
    return Candidate(
        canonical_id="gh:testdev",
        source="github",
        github_username="testdev",
        hf_username=None,
        display_name="Test Dev",
        profile_json={
            "bio": "builder",
            "company": "none",
            "location": "Istanbul",
            "public_repos": 12,
            "followers": 42,
        },
        first_seen_at=datetime.datetime.now(),
        last_seen_at=datetime.datetime.now(),
    )


def _make_repo(name: str = "cool-repo", stars: int = 10, pushed_days_ago: int = 1) -> Repo:
    return Repo(
        owner="testdev",
        name=name,
        description="a repo",
        language="rust",
        stars=stars,
        forks=1,
        topics=["webrtc"],
        has_readme=True,
        homepage=None,
        pushed_at=datetime.datetime.now(datetime.timezone.utc)
        - datetime.timedelta(days=pushed_days_ago),
        created_at=datetime.datetime.now() - datetime.timedelta(days=365),
        is_fork=False,
    )


def test_relevance_function_signatures() -> None:
    assert callable(verified_inviter.relevance.judge_repo)
    assert callable(verified_inviter.relevance.judge_repos_for_candidate)
    sig = inspect.signature(verified_inviter.relevance.judge_repos_for_candidate)
    assert list(sig.parameters.keys()) == ["llm", "candidate", "repos"]


def test_judge_repo_maps_response() -> None:
    llm = FakeSambaNovaClient([{"relevant": True, "domain": "webrtc", "reasoning": "real"}])
    candidate = _make_candidate()
    repo = _make_repo()
    verdict = verified_inviter.relevance.judge_repo(llm, candidate, repo)
    assert isinstance(verdict, RepoVerdict)
    assert verdict.relevant is True
    assert verdict.domain == "webrtc"
    assert verdict.repo_name == "testdev/cool-repo"
    assert "examples of genuine" in llm.calls[0][1].lower()


def test_judge_repos_for_candidate_filters_recency() -> None:
    llm = FakeSambaNovaClient([{"relevant": True, "domain": "x", "reasoning": "y"}])
    candidate = _make_candidate()
    repos = [
        _make_repo("fresh", pushed_days_ago=1),
        _make_repo("old", pushed_days_ago=999),
    ]
    verdicts = verified_inviter.relevance.judge_repos_for_candidate(llm, candidate, repos)
    assert len(verdicts) == 1
    assert verdicts[0].repo_name == "testdev/fresh"


def test_content_fetch_function_signatures() -> None:
    assert callable(verified_inviter.content_fetch.fetch_repo_contents)
    assert callable(verified_inviter.content_fetch.fetch_contents_for_relevant_repos)


def test_truncate_text_preserves_sentence() -> None:
    text = "A. B. C. D. E."
    truncated = verified_inviter.content_fetch._truncate_text(text, max_chars=10)
    assert truncated in ("A.", "A. B.", "A. B. C.")


def test_knowledge_function_signatures() -> None:
    assert callable(verified_inviter.knowledge.extract_knowledge)
    assert callable(verified_inviter.knowledge.extract_knowledge_for_candidate)


def test_extract_knowledge_maps_response() -> None:
    llm = FakeSambaNovaClient([
        {
            "summary": " knows webrtc",
            "domains": ["webrtc"],
            "technologies": ["rust"],
            "evidence": [{"repo": "testdev/cool-repo", "demonstrates": "media"}],
        }
    ])
    candidate = _make_candidate()
    repo = _make_repo()
    contents = [(repo, {"_truncated_text": "README\nwebrtc stuff"})]
    knowledge = verified_inviter.knowledge.extract_knowledge(llm, candidate, [repo], contents)
    assert isinstance(knowledge, Knowledge)
    assert knowledge.canonical_id == "gh:testdev"
    assert knowledge.domains == ["webrtc"]
    assert knowledge.technologies == ["rust"]
    assert knowledge.evidence[0]["repo"] == "testdev/cool-repo"


def test_extract_knowledge_for_candidate_persists() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        conn = init_db(Path(tmp) / "test.db")
        llm = FakeSambaNovaClient([
            {
                "summary": "s",
                "domains": ["d"],
                "technologies": ["t"],
                "evidence": [],
            }
        ])
        candidate = _make_candidate()
        repo = _make_repo()
        contents = [(repo, {"_truncated_text": "x"})]
        knowledge = verified_inviter.knowledge.extract_knowledge_for_candidate(
            llm, conn, candidate, [repo], contents
        )
        assert knowledge.canonical_id == "gh:testdev"


def test_technical_judge_function_signatures() -> None:
    assert callable(verified_inviter.technical_judge.judge_technical_quality)
    assert callable(verified_inviter.technical_judge.run_technical_judge_for_candidate)
    assert callable(verified_inviter.technical_judge.append_reject_to_outbox)


def test_judge_technical_quality_maps_response() -> None:
    llm = FakeSambaNovaClient([
        {
            "verdict": "worth_a_damn",
            "criteria_met": ["2", "3"],
            "reasoning": "real depth",
            "seed_stage": True,
            "confidence": "high",
        }
    ])
    candidate = _make_candidate()
    knowledge = Knowledge(
        canonical_id="gh:testdev",
        summary="good",
        domains=["webrtc"],
        technologies=["rust"],
        evidence=[{"repo": "testdev/cool-repo", "demonstrates": "media"}],
    )
    repo_verdicts = [
        RepoVerdict(canonical_id="gh:testdev", repo_name="testdev/cool-repo", relevant=True, domain="webrtc", reasoning="real")
    ]
    repos = [_make_repo()]
    verdict = verified_inviter.technical_judge.judge_technical_quality(
        llm, candidate, knowledge, repo_verdicts, repos
    )
    assert isinstance(verdict, TechnicalVerdict)
    assert verdict.verdict == "worth_a_damn"
    assert verdict.seed_stage is True
    assert verdict.criteria_met == ["2", "3"]
    assert "four criteria" in llm.calls[0][1].lower()


def test_run_technical_judge_for_candidate_persists() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        conn = init_db(Path(tmp) / "test.db")
        llm = FakeSambaNovaClient([
            {
                "verdict": "skip",
                "criteria_met": [],
                "reasoning": "slop",
                "seed_stage": False,
                "confidence": "medium",
            }
        ])
        candidate = _make_candidate()
        knowledge = Knowledge(
            canonical_id="gh:testdev",
            summary="ok",
            domains=["x"],
            technologies=["y"],
            evidence=[],
        )
        verified_inviter.store.insert_or_replace_knowledge(conn, knowledge)
        verified_inviter.store.insert_repo_verdicts(conn, [
            RepoVerdict(canonical_id="gh:testdev", repo_name="testdev/cool-repo", relevant=True, domain="x", reasoning="r")
        ])
        repos = [_make_repo()]
        verdict = verified_inviter.technical_judge.run_technical_judge_for_candidate(
            llm, conn, candidate, repos
        )
        assert verdict.verdict == "skip"


def test_matching_function_signatures() -> None:
    assert callable(verified_inviter.matching.load_companies)
    assert callable(verified_inviter.matching.pick_matching_company)
    assert callable(verified_inviter.matching.run_match_for_candidate)


def test_load_companies(tmp_path: Path) -> None:
    path = tmp_path / "companies.json"
    path.write_text(json.dumps([{"name": "A", "what_they_are_building": "x", "website": "https://a.co"}]))
    companies = verified_inviter.matching.load_companies(path)
    assert companies[0]["name"] == "A"


def test_pick_matching_company_maps_none() -> None:
    llm = FakeSambaNovaClient([{"match": "none", "why": "", "confidence": "low"}])
    knowledge = Knowledge(
        canonical_id="gh:testdev",
        summary="s",
        domains=["d"],
        technologies=["t"],
        evidence=[],
    )
    match = verified_inviter.matching.pick_matching_company(llm, knowledge, [])
    assert isinstance(match, Match)
    assert match.match_company is None


def test_pick_matching_company_maps_company() -> None:
    llm = FakeSambaNovaClient([{"match": "Pragma", "why": "fit", "confidence": "high"}])
    knowledge = Knowledge(
        canonical_id="gh:testdev",
        summary="s",
        domains=["d"],
        technologies=["t"],
        evidence=[],
    )
    match = verified_inviter.matching.pick_matching_company(llm, knowledge, [])
    assert match.match_company == "Pragma"


def test_run_match_for_candidate_persists() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        conn = init_db(Path(tmp) / "test.db")
        companies_path = Path(tmp) / "companies.json"
        companies_path.write_text(json.dumps([{"name": "A", "what_they_are_building": "x", "website": "https://a.co"}]))
        llm = FakeSambaNovaClient([{"match": "none", "why": "", "confidence": "low"}])
        candidate = _make_candidate()
        knowledge = Knowledge(
            canonical_id="gh:testdev",
            summary="s",
            domains=["d"],
            technologies=["t"],
            evidence=[],
        )
        verified_inviter.store.insert_or_replace_knowledge(conn, knowledge)
        match = verified_inviter.matching.run_match_for_candidate(llm, conn, candidate, companies_path)
        assert match.match_company is None


def test_email_draft_function_signatures() -> None:
    assert callable(verified_inviter.email_draft.generate_ref_token)
    assert callable(verified_inviter.email_draft.draft_personalized_email)
    assert callable(verified_inviter.email_draft.draft_generic_email)
    assert callable(verified_inviter.email_draft.draft_email_for_candidate)


def test_generate_ref_token() -> None:
    token = verified_inviter.email_draft.generate_ref_token()
    assert isinstance(token, str)
    assert len(token) >= 16


def test_draft_personalized_email() -> None:
    llm = FakeSambaNovaClient([{"subject": "hi", "body": "hello"}])
    candidate = _make_candidate()
    knowledge = Knowledge(
        canonical_id="gh:testdev",
        summary="s",
        domains=["d"],
        technologies=["t"],
        evidence=[{"repo": "r", "demonstrates": "x"}],
    )
    match = Match(canonical_id="gh:testdev", match_company="A", why="fit", confidence="high")
    companies = [{"name": "A", "what_they_are_building": "x", "website": "https://a.co"}]
    draft = verified_inviter.email_draft.draft_personalized_email(
        llm, candidate, knowledge, match, companies, "tok123"
    )
    assert isinstance(draft, DraftEmail)
    assert draft.email_path == "personalized"
    assert draft.matched_company == "A"
    assert draft.ref_token == "tok123"


def test_draft_generic_email() -> None:
    llm = FakeSambaNovaClient([{"subject": "hi", "body": "hello"}])
    candidate = _make_candidate()
    knowledge = Knowledge(
        canonical_id="gh:testdev",
        summary="s",
        domains=["d"],
        technologies=["t"],
        evidence=[],
    )
    draft = verified_inviter.email_draft.draft_generic_email(llm, candidate, knowledge, "tok")
    assert isinstance(draft, DraftEmail)
    assert draft.email_path == "generic"
    assert draft.matched_company is None


def test_draft_email_for_candidate_path_a() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        conn = init_db(Path(tmp) / "test.db")
        llm = FakeSambaNovaClient([{"subject": "hi", "body": "hello"}])
        candidate = _make_candidate()
        knowledge = Knowledge(
            canonical_id="gh:testdev",
            summary="s",
            domains=["d"],
            technologies=["t"],
            evidence=[],
        )
        verified_inviter.store.insert_or_replace_knowledge(conn, knowledge)
        verified_inviter.store.insert_or_replace_match(conn, Match(
            canonical_id="gh:testdev", match_company="A", why="fit", confidence="high"
        ))
        companies = [{"name": "A", "what_they_are_building": "x", "website": "https://a.co"}]
        draft = verified_inviter.email_draft.draft_email_for_candidate(llm, conn, candidate, companies)
        assert draft.email_path == "personalized"
        assert draft.matched_company == "A"


def test_draft_email_for_candidate_path_b() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        conn = init_db(Path(tmp) / "test.db")
        llm = FakeSambaNovaClient([{"subject": "hi", "body": "hello"}])
        candidate = _make_candidate()
        knowledge = Knowledge(
            canonical_id="gh:testdev",
            summary="s",
            domains=["d"],
            technologies=["t"],
            evidence=[],
        )
        verified_inviter.store.insert_or_replace_knowledge(conn, knowledge)
        verified_inviter.store.insert_or_replace_match(conn, Match(
            canonical_id="gh:testdev", match_company=None, why="", confidence="low"
        ))
        draft = verified_inviter.email_draft.draft_email_for_candidate(llm, conn, candidate, [])
        assert draft.email_path == "generic"
        assert draft.matched_company is None
