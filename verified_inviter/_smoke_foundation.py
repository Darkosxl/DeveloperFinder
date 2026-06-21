from __future__ import annotations

import tempfile
from datetime import datetime, date
from pathlib import Path

from verified_inviter import store
from verified_inviter.models import (
    Candidate,
    DraftEmail,
    Knowledge,
    Match,
    RepoVerdict,
    TechnicalVerdict,
)
from verified_inviter.llm_client import SambaNovaClient


def _candidate() -> Candidate:
    return Candidate(
        canonical_id="gh:testuser",
        source="github",
        github_username="testuser",
        hf_username=None,
        display_name="Test User",
        profile_json={"login": "testuser", "bio": "test"},
        first_seen_at=datetime.now(),
        last_seen_at=datetime.now(),
    )


def _run() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        conn = store.init_db(db_path)

        # candidates
        candidate = _candidate()
        store.upsert_candidate(conn, candidate)
        blocked = store.is_blocked_for_processing(conn, candidate.canonical_id, 30)
        assert not blocked, "fresh candidate should not be blocked"

        # repo verdicts
        verdicts = [
            RepoVerdict(
                canonical_id=candidate.canonical_id,
                repo_name="cool/repo",
                relevant=True,
                domain="webrtc",
                reasoning="real media server",
            ),
            RepoVerdict(
                canonical_id=candidate.canonical_id,
                repo_name="lame/landing",
                relevant=False,
                domain="",
                reasoning="landing page",
            ),
        ]
        store.insert_repo_verdicts(conn, verdicts)
        assert store.has_any_relevant_repo(conn, candidate.canonical_id)
        relevant = store.get_relevant_repos_for(conn, candidate.canonical_id)
        assert len(relevant) == 1
        assert relevant[0][0] == "cool/repo"

        # knowledge
        knowledge = Knowledge(
            canonical_id=candidate.canonical_id,
            summary="Knows WebRTC and Rust.",
            domains=["webrtc", "systems"],
            technologies=["rust", "pion"],
            evidence=[{"repo": "cool/repo", "demonstrates": "media server"}],
        )
        store.insert_or_replace_knowledge(conn, knowledge)
        fetched = store.get_knowledge(conn, candidate.canonical_id)
        assert fetched is not None
        assert fetched.summary == knowledge.summary
        assert fetched.domains == ["webrtc", "systems"]

        # technical verdict
        tv = TechnicalVerdict(
            canonical_id=candidate.canonical_id,
            verdict="worth_a_damn",
            criteria_met=["1", "3"],
            reasoning="Strong repo work and traction.",
            seed_stage=True,
            confidence="high",
        )
        store.insert_or_replace_technical_verdict(conn, tv)
        fetched_tv = store.get_technical_verdict(conn, candidate.canonical_id)
        assert fetched_tv is not None
        assert fetched_tv.verdict == "worth_a_damn"
        assert fetched_tv.seed_stage is True

        # match
        match = Match(
            canonical_id=candidate.canonical_id,
            match_company="Pragma",
            why="Great fit for infra work.",
            confidence="high",
        )
        store.insert_or_replace_match(conn, match)
        fetched_match = store.get_match(conn, candidate.canonical_id)
        assert fetched_match is not None
        assert fetched_match.match_company == "Pragma"

        # invite draft
        draft = DraftEmail(
            canonical_id=candidate.canonical_id,
            subject="You're invited",
            body="Hi, apply here.",
            email_path="personalized",
            matched_company="Pragma",
            ref_token="abc123",
        )
        invite_id = store.create_invite_draft(conn, draft)
        assert invite_id > 0
        pending = store.list_pending_invites(conn, date.today())
        assert len(pending) == 1
        assert pending[0]["canonical_id"] == candidate.canonical_id

        store.mark_invite_sent(conn, invite_id, datetime.now())
        assert store.is_blocked_for_processing(conn, candidate.canonical_id, 30)

        # run stats
        store.insert_run(conn, date.today(), dry_run=True)
        store.increment_run_stats(conn, date.today(), candidates_seen=1, repos_judged=2, invites_drafted=1)
        store.finish_run(conn, date.today(), datetime.now(), error=None)

        # SambaNovaClient instantiates without network call
        client = SambaNovaClient(api_key="fake", base_url="https://example.com", model="test")
        assert client.model == "test"
        client.close()

        print("OK: foundation smoke test passed")


if __name__ == "__main__":
    _run()
