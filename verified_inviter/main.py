from __future__ import annotations

import argparse
import json
import logging
import logging.handlers
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from verified_inviter import config, email_send
from verified_inviter.discovery import github, huggingface
from verified_inviter.email_draft import draft_email_for_candidate
from verified_inviter.knowledge import extract_knowledge_for_candidate
from verified_inviter.llm_client import SambaNovaClient
from verified_inviter.matching import load_companies, run_match_for_candidate
from verified_inviter.models import Candidate, DraftEmail
from verified_inviter.relevance import judge_repos_for_candidate
from verified_inviter.store import (
    finish_run,
    increment_run_stats,
    init_db,
    insert_repo_verdicts,
    insert_run,
    is_blocked_for_processing,
    upsert_candidate,
)
from verified_inviter.technical_judge import (
    append_reject_to_outbox,
    run_technical_judge_for_candidate,
)
from verified_inviter.content_fetch import fetch_contents_for_relevant_repos

logger = logging.getLogger(__name__)


def setup_logging(log_dir: Path, level: str) -> None:
    """Configure stdout + rotating file logs + one-line runs.log."""
    log_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    stdout = logging.StreamHandler(sys.stdout)
    stdout.setLevel(logging.DEBUG)
    stdout.setFormatter(fmt)
    root.addHandler(stdout)

    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "verified_inviter.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    runs_handler = logging.FileHandler(
        log_dir / "runs.log",
        encoding="utf-8",
    )
    runs_handler.setLevel(logging.INFO)
    runs_handler.setFormatter(fmt)
    root.addHandler(runs_handler)


def build_http_clients() -> dict[str, httpx.Client]:
    """Return named httpx clients for GitHub, HuggingFace, and Exa."""
    github_client = httpx.Client(
        base_url="https://api.github.com",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {config.GITHUB_TOKEN}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        timeout=30.0,
    )
    hf_client = httpx.Client(
        headers={"User-Agent": "Mozilla/5.0 (compatible; ExposureBot/1.0)"},
        timeout=30.0,
    )
    exa_client = httpx.Client(
        base_url="https://api.exa.ai",
        headers={
            "Authorization": f"Bearer {config.EXA_API_KEY}",
            "Content-Type": "application/json",
        },
        timeout=120.0,
    )
    return {
        "github": github_client,
        "hf": hf_client,
        "exa": exa_client,
    }


def discover_candidates(
    github_client: httpx.Client,
    hf_client: httpx.Client,
) -> list[Candidate]:
    """Run GitHub + HuggingFace discovery and cross-match."""
    gh = github.infer_candidates_from_github(github_client)
    hf = huggingface.infer_candidates_from_huggingface(hf_client, config.TURKISH_NAMES_PATH)
    return huggingface.cross_match_github_hf(gh, hf)


def process_candidate(
    conn: Any,
    llm: SambaNovaClient,
    github_client: httpx.Client,
    exa_client: httpx.Client,
    candidate: Candidate,
    companies: list[dict],
) -> DraftEmail | None:
    """Run one candidate through stages 3→8 and return the drafted email if any."""
    upsert_candidate(conn, candidate)

    # Stage 3: repo relevance judging
    repos = github.list_user_repos(github_client, candidate.github_username or "")
    if not repos:
        logger.info("no recent repos for candidate", extra={"canonical_id": candidate.canonical_id})
        return None

    verdicts = judge_repos_for_candidate(llm, candidate, repos)
    insert_repo_verdicts(conn, verdicts)
    increment_run_stats(conn, date.today(), repos_judged=len(verdicts))

    relevant_repos = [repo for repo in repos if any(
        v.repo_name == f"{repo.owner}/{repo.name}" and v.relevant for v in verdicts
    )]
    if not relevant_repos:
        logger.info("no relevant repos for candidate", extra={"canonical_id": candidate.canonical_id})
        return None

    # Stage 4: fetch contents for relevant repos
    contents = fetch_contents_for_relevant_repos(
        exa_client,
        relevant_repos,
        cap=config.MAX_RELEVANT_REPOS_PER_CANDIDATE,
    )

    # Stage 5: knowledge extraction
    try:
        knowledge = extract_knowledge_for_candidate(llm, conn, candidate, relevant_repos, contents)
    except Exception as exc:
        logger.exception("knowledge extraction failed", extra={"canonical_id": candidate.canonical_id})
        return None

    # Stage 6: technical judge
    try:
        verdict = run_technical_judge_for_candidate(llm, conn, candidate, repos)
    except Exception as exc:
        logger.exception("technical judge failed", extra={"canonical_id": candidate.canonical_id})
        return None

    if verdict.verdict != "worth_a_damn" or not verdict.seed_stage:
        logger.info(
            "technical judge rejected candidate",
            extra={"canonical_id": candidate.canonical_id, "reasoning": verdict.reasoning},
        )
        append_reject_to_outbox(candidate.canonical_id, candidate.github_username, verdict)
        return None

    # Stage 7: company matching
    try:
        match = run_match_for_candidate(llm, conn, candidate, config.COMPANIES_PATH)
    except Exception as exc:
        logger.exception("company matching failed", extra={"canonical_id": candidate.canonical_id})
        return None

    # Stage 8: email drafting
    try:
        draft = draft_email_for_candidate(llm, conn, candidate, companies)
    except Exception as exc:
        logger.exception("email drafting failed", extra={"canonical_id": candidate.canonical_id})
        return None

    return draft


def run_daily(dry_run: bool) -> None:
    """Main entry point for a daily run."""
    run_date = date.today()
    conn = init_db(config.DB_PATH)

    try:
        insert_run(conn, run_date, dry_run)
        setup_logging(config.LOG_DIR, config.LOG_LEVEL)
        logger.info("run started", extra={"dry_run": dry_run})

        clients = build_http_clients()
        try:
            llm = SambaNovaClient(
                config.SAMBA_API_KEY,
                config.SAMBA_BASE_URL,
                config.SAMBA_MODEL,
            )
            try:
                companies = load_companies(config.COMPANIES_PATH)
                candidates = discover_candidates(clients["github"], clients["hf"])
                increment_run_stats(conn, run_date, candidates_seen=len(candidates))

                accepted: list[DraftEmail] = []
                for candidate in candidates[:config.MAX_CANDIDATES_PER_RUN]:
                    if is_blocked_for_processing(conn, candidate.canonical_id, config.SKIP_REJUDGE_DAYS):
                        logger.info("candidate blocked", extra={"canonical_id": candidate.canonical_id})
                        continue

                    try:
                        draft = process_candidate(
                            conn, llm, clients["github"], clients["exa"], candidate, companies
                        )
                    except Exception as exc:
                        logger.exception(
                            "candidate processing failed",
                            extra={"canonical_id": candidate.canonical_id},
                        )
                        continue

                    if draft:
                        # Note: DraftEmail does not carry a recipient_email field in the
                        # current model; email discovery is a v1 placeholder. The actual
                        # send function skips invites with NULL recipient_email.
                        accepted.append(draft)
                        if len(accepted) >= config.DAILY_INVITE_CAP:
                            break

                increment_run_stats(conn, run_date, invites_drafted=len(accepted))
                sent, failed = email_send.send_pending_invites(
                    conn, dry_run, config.OUTBOX_DIR, config.DAILY_INVITE_CAP
                )
                increment_run_stats(conn, run_date, invites_sent=sent)

                logger.info("run finished", extra={"sent": sent, "failed": failed})
                finish_run(conn, run_date, datetime.now(), error=None)
            finally:
                llm.close()
        finally:
            for client in clients.values():
                client.close()
    except Exception as e:
        logger.exception("run failed")
        finish_run(conn, run_date, datetime.now(), error=str(e))
        raise


def run_self_test(username: str) -> None:
    """Run the pipeline on a single GitHub username without discovery or sending."""
    conn = init_db(config.DB_PATH)
    setup_logging(config.LOG_DIR, config.LOG_LEVEL)

    clients = build_http_clients()
    try:
        llm = SambaNovaClient(
            config.SAMBA_API_KEY,
            config.SAMBA_BASE_URL,
            config.SAMBA_MODEL,
        )
        try:
            companies = load_companies(config.COMPANIES_PATH)
            profile = github.get_user_profile(clients["github"], username)

            now = datetime.now(tz=timezone.utc)
            candidate = Candidate(
                canonical_id=f"self-test:{username}",
                source="self-test",
                github_username=username,
                hf_username=None,
                display_name=profile.get("name") or profile.get("login"),
                profile_json={"github": profile},
                first_seen_at=now,
                last_seen_at=now,
            )
            upsert_candidate(conn, candidate)

            draft = process_candidate(
                conn, llm, clients["github"], clients["exa"], candidate, companies
            )

            stamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
            out_dir = config.OUTBOX_DIR / f"self-test-{stamp}" / username
            out_dir.mkdir(parents=True, exist_ok=True)

            from verified_inviter.store import get_knowledge, get_match, get_technical_verdict

            knowledge = get_knowledge(conn, candidate.canonical_id)
            verdict = get_technical_verdict(conn, candidate.canonical_id)
            match = get_match(conn, candidate.canonical_id)

            if knowledge:
                (out_dir / "knowledge.json").write_text(
                    json.dumps(
                        {
                            "canonical_id": knowledge.canonical_id,
                            "summary": knowledge.summary,
                            "domains": knowledge.domains,
                            "technologies": knowledge.technologies,
                            "evidence": knowledge.evidence,
                        },
                        indent=2,
                        default=str,
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
            if verdict:
                (out_dir / "technical_verdict.json").write_text(
                    json.dumps(
                        {
                            "canonical_id": verdict.canonical_id,
                            "verdict": verdict.verdict,
                            "criteria_met": verdict.criteria_met,
                            "reasoning": verdict.reasoning,
                            "seed_stage": verdict.seed_stage,
                            "confidence": verdict.confidence,
                        },
                        indent=2,
                        default=str,
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
            if match:
                (out_dir / "match.json").write_text(
                    json.dumps(
                        {
                            "canonical_id": match.canonical_id,
                            "match_company": match.match_company,
                            "why": match.why,
                            "confidence": match.confidence,
                        },
                        indent=2,
                        default=str,
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
            if draft:
                (out_dir / "draft_email.txt").write_text(
                    f"Subject: {draft.subject}\n\n{draft.body}\n",
                    encoding="utf-8",
                )

            print("\n=== Self-test summary ===")
            print(f"Username: {username}")
            print(f"Canonical ID: {candidate.canonical_id}")
            if knowledge:
                print(f"\nKnowledge summary:\n{knowledge.summary}")
            if verdict:
                print(
                    f"\nTechnical verdict: {verdict.verdict} "
                    f"(seed_stage={verdict.seed_stage}, confidence={verdict.confidence})"
                )
                print(f"Reasoning: {verdict.reasoning}")
            if match:
                print(f"\nMatch: {match.match_company or 'none'}")
                print(f"Why: {match.why}")
            if draft:
                print(f"\nEmail subject: {draft.subject}")
                print(f"Email body preview:\n{draft.body[:200]}...")
            else:
                print("\nNo draft produced (candidate dropped earlier in the pipeline).")
            print(f"\nArtifacts written to: {out_dir}")
        finally:
            llm.close()
    finally:
        for client in clients.values():
            client.close()


def main() -> None:
    """CLI dispatcher."""
    parser = argparse.ArgumentParser(description="Verified Inviter Agent")
    parser.add_argument(
        "--self-test",
        dest="self_test",
        metavar="GITHUB_USERNAME",
        help="Run the pipeline on a single GitHub username without discovery or sending.",
    )
    args = parser.parse_args()

    if args.self_test:
        run_self_test(args.self_test)
    else:
        run_daily(config.DRY_RUN)


if __name__ == "__main__":
    main()
