from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from verified_inviter import config
from verified_inviter.models import Candidate, Repo

logger = logging.getLogger(__name__)

_GITHUB_API_BASE = "https://api.github.com"


def _github_headers() -> dict[str, str]:
    return {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {config.GITHUB_TOKEN}",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _parse_github_timestamp(value: str | None) -> datetime:
    """Parse an ISO 8601 timestamp from the GitHub API into a UTC datetime."""
    if not value:
        return datetime.min.replace(tzinfo=timezone.utc)
    # GitHub returns ISO 8601 with a trailing Z (e.g. 2024-01-01T00:00:00Z)
    normalized = value.replace("Z", "+00:00")
    dt = datetime.fromisoformat(normalized)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _check_rate_limit_headroom(response: httpx.Response) -> bool:
    """Return True if we still have more than the configured headroom remaining."""
    remaining = response.headers.get("x-ratelimit-remaining")
    if remaining is None:
        return True
    try:
        return int(remaining) > config.GITHUB_RATE_LIMIT_HEADROOM
    except ValueError:
        return True


def _wait_for_rate_limit_reset(response: httpx.Response) -> None:
    """Sleep until the GitHub rate limit resets, if a reset header is present."""
    reset_epoch = response.headers.get("x-ratelimit-reset")
    if reset_epoch is None:
        return
    try:
        reset_at = datetime.fromtimestamp(int(reset_epoch), tz=timezone.utc)
        now = datetime.now(tz=timezone.utc)
        wait_seconds = (reset_at - now).total_seconds()
        if wait_seconds > 0:
            logger.warning(
                "GitHub rate limit exhausted; sleeping until reset",
                extra={"wait_seconds": int(wait_seconds)},
            )
            import time

            time.sleep(wait_seconds + 1)
    except (ValueError, OSError):
        pass


def search_turkey_users(
    client: httpx.Client,
    per_page: int = 100,
    max_pages: int = 5,
) -> list[dict[str, Any]]:
    """Search GitHub users located in Turkey.

    Uses the GitHub Search API with a disjunction of location queries. Returns
    the raw ``items`` dicts. Stops early if rate-limit headroom is exhausted or
    if a page comes back empty / incomplete.
    """
    query = (
        "location:Turkey OR location:Türkey OR location:Istanbul "
        "OR location:Ankara OR location:Izmir"
    )
    items: list[dict[str, Any]] = []
    for page in range(1, max_pages + 1):
        response = client.get(
            f"{_GITHUB_API_BASE}/search/users",
            headers=_github_headers(),
            params={
                "q": query,
                "per_page": per_page,
                "page": page,
            },
            timeout=30.0,
        )
        if response.status_code == 403 and not _check_rate_limit_headroom(response):
            _wait_for_rate_limit_reset(response)
            continue
        response.raise_for_status()
        data = response.json()
        page_items = data.get("items", [])
        if not page_items or data.get("incomplete_results"):
            logger.warning(
                "GitHub search stopped early",
                extra={"page": page, "reason": "incomplete_results" if data.get("incomplete_results") else "empty"},
            )
            break
        items.extend(page_items)
        if not _check_rate_limit_headroom(response):
            logger.warning(
                "GitHub search stopping early due to rate-limit headroom",
                extra={"page": page, "remaining": response.headers.get("x-ratelimit-remaining")},
            )
            break
    return items


def get_user_profile(client: httpx.Client, username: str) -> dict[str, Any]:
    """Fetch a full GitHub user profile."""
    response = client.get(
        f"{_GITHUB_API_BASE}/users/{username}",
        headers=_github_headers(),
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()


def has_recent_activity(client: httpx.Client, username: str, days: int) -> bool:
    """Return True if the user has a public event within the last ``days`` days.

    Conservatively returns True on 404 or rate-limit errors so that we do not
    drop candidates just because the events endpoint is unavailable.
    """
    try:
        response = client.get(
            f"{_GITHUB_API_BASE}/users/{username}/events/public",
            headers=_github_headers(),
            params={"per_page": 30, "page": 1},
            timeout=30.0,
        )
    except httpx.HTTPError as exc:
        logger.warning(
            "Could not fetch GitHub events; assuming recent activity",
            extra={"username": username, "error": str(exc)},
        )
        return True

    if response.status_code in {404, 403}:
        return True

    if response.status_code >= 500:
        return True

    try:
        response.raise_for_status()
    except httpx.HTTPError:
        return True

    events = response.json()
    if not events:
        return False

    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=days)
    for event in events:
        created_at = event.get("created_at")
        if not created_at:
            continue
        event_time = _parse_github_timestamp(created_at)
        if event_time >= cutoff:
            return True
    return False


def repo_from_dict(repo: dict[str, Any]) -> Repo:
    """Map a GitHub repo JSON dict to a ``Repo`` dataclass."""
    owner = (repo.get("owner") or {}).get("login", "")
    return Repo(
        owner=owner,
        name=repo.get("name", ""),
        description=repo.get("description"),
        language=repo.get("language"),
        stars=repo.get("stargazers_count", 0) or 0,
        forks=repo.get("forks_count", 0) or 0,
        topics=repo.get("topics", []) or [],
        has_readme=bool(repo.get("has_readme", True)),
        homepage=repo.get("homepage") or None,
        pushed_at=_parse_github_timestamp(repo.get("pushed_at")),
        created_at=_parse_github_timestamp(repo.get("created_at")),
        is_fork=repo.get("fork", False),
    )


def list_user_repos(client: httpx.Client, username: str) -> list[Repo]:
    """Enumerate a user's owned repositories, newest first, filtered by recency.

    Repos whose ``pushed_at`` is older than ``config.REPO_RECENCY_DAYS`` are
    skipped. Pagination stops when a page contains no repos or all repos on the
    page are older than the cutoff.
    """
    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=config.REPO_RECENCY_DAYS)
    repos: list[Repo] = []
    page = 1
    while True:
        response = client.get(
            f"{_GITHUB_API_BASE}/users/{username}/repos",
            headers=_github_headers(),
            params={
                "type": "owner",
                "sort": "pushed",
                "direction": "desc",
                "per_page": 100,
                "page": page,
            },
            timeout=30.0,
        )
        if response.status_code == 403 and not _check_rate_limit_headroom(response):
            _wait_for_rate_limit_reset(response)
            continue
        response.raise_for_status()
        page_items = response.json()
        if not page_items:
            break

        all_old = True
        for repo in page_items:
            mapped = repo_from_dict(repo)
            if mapped.pushed_at >= cutoff:
                repos.append(mapped)
                all_old = False

        if all_old:
            break
        page += 1
    return repos


def infer_candidates_from_github(client: httpx.Client) -> list[Candidate]:
    """Discover Turkey-based GitHub users and build ``Candidate`` objects.

    The returned list is de-duplicated by GitHub username.
    """
    search_items = search_turkey_users(client)
    candidates: list[Candidate] = []
    seen: set[str] = set()
    now = datetime.now(tz=timezone.utc)

    for item in search_items:
        username = item.get("login")
        if not username or username in seen:
            continue
        seen.add(username)

        try:
            profile = get_user_profile(client, username)
        except httpx.HTTPError as exc:
            logger.warning(
                "Skipping GitHub user; profile fetch failed",
                extra={"username": username, "error": str(exc)},
            )
            continue

        if not has_recent_activity(client, username, config.REPO_RECENCY_DAYS):
            logger.info(
                "Skipping GitHub user; no recent activity",
                extra={"username": username},
            )
            continue

        display_name = profile.get("name") or profile.get("login")
        candidate = Candidate(
            canonical_id=f"gh:{username}",
            source="github",
            github_username=username,
            hf_username=None,
            display_name=display_name,
            profile_json={"github": profile},
            first_seen_at=now,
            last_seen_at=now,
        )
        candidates.append(candidate)

    return candidates
