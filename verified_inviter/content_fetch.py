from __future__ import annotations

import time

import httpx

from verified_inviter import config
from verified_inviter.models import Repo


EXA_CONTENTS_URL = "https://api.exa.ai/contents"


def fetch_repo_contents(
    client: httpx.Client,
    repo: Repo,
    max_retries: int = 2,
) -> dict:
    """Fetch Exa contents for a single GitHub repo URL.

    Retries with exponential backoff on 429 errors.
    """
    url = f"https://github.com/{repo.owner}/{repo.name}"
    headers = {
        "Authorization": f"Bearer {config.EXA_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {"urls": [url]}

    last_exception: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = client.post(EXA_CONTENTS_URL, headers=headers, json=body)
            if response.status_code == 429:
                backoff = 2**attempt
                time.sleep(backoff)
                continue
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            last_exception = exc
            if exc.response.status_code == 429 and attempt < max_retries:
                backoff = 2**attempt
                time.sleep(backoff)
                continue
            raise

    if last_exception is not None:
        raise last_exception
    return {}


def _recency_score(repo: Repo) -> float:
    """Higher score for repos pushed recently."""
    from datetime import datetime, timezone

    pushed = repo.pushed_at
    if pushed.tzinfo is None:
        pushed = pushed.replace(tzinfo=timezone.utc)
    days_since_push = max(1, (datetime.now(tz=timezone.utc) - pushed).days)
    return repo.stars / days_since_push


def _concat_exa_text(exa_result: dict) -> str:
    """Concatenate Exa result texts into a single string."""
    parts: list[str] = []
    for result in exa_result.get("results", []):
        text = result.get("text", "")
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def _truncate_text(text: str, max_chars: int = 15000) -> str:
    """Truncate text to max_chars while preserving complete sentences."""
    if len(text) <= max_chars:
        return text

    # Find the last sentence boundary before the limit
    truncated = text[:max_chars]
    for marker in ".\n!?":
        idx = truncated.rfind(marker)
        if idx > 0:
            return truncated[: idx + 1].strip()
    return truncated.strip()


def fetch_contents_for_relevant_repos(
    client: httpx.Client,
    relevant_repos: list[Repo],
    cap: int,
) -> list[tuple[Repo, dict]]:
    """Sort relevant repos by stars * recency, take the top cap, and fetch Exa contents.

    Returns a list of (Repo, exa_result) tuples. The exa_result dict is augmented
    with a key `_truncated_text` containing the concatenated, truncated text.
    """
    sorted_repos = sorted(relevant_repos, key=_recency_score, reverse=True)
    capped = sorted_repos[:cap]

    results: list[tuple[Repo, dict]] = []
    for repo in capped:
        exa_result = fetch_repo_contents(client, repo)
        full_text = _concat_exa_text(exa_result)
        truncated = _truncate_text(full_text, max_chars=15000)
        exa_result["_truncated_text"] = truncated
        results.append((repo, exa_result))

    return results
