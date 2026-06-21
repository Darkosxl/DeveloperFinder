from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from selectolax.parser import HTMLParser

from verified_inviter import config
from verified_inviter.discovery.turkish_names import (
    contains_turkish_diacritic,
    default_turkish_names,
    load_turkish_names,
    matches_turkish_name,
)
from verified_inviter.models import Candidate

logger = logging.getLogger(__name__)

_HF_BASE = "https://huggingface.co"
_HF_USER_AGENT = "Mozilla/5.0 (compatible; ExposureBot/1.0)"

DEFAULT_LEADERBOARD_URLS = [
    "https://huggingface.co/models?sort=downloads",
    "https://huggingface.co/spaces?sort=likes",
    "https://huggingface.co/datasets?sort=downloads",
    "https://huggingface.co/papers",
    "https://huggingface.co/trending",
]


class HFCache:
    """Simple disk cache for HuggingFace HTML pages with a TTL."""

    def __init__(self, cache_dir: Path, ttl_seconds: int) -> None:
        self.cache_dir = cache_dir
        self.ttl_seconds = ttl_seconds
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, url: str) -> str:
        return hashlib.sha256(url.encode("utf-8")).hexdigest()[:32]

    def _path(self, url: str) -> Path:
        return self.cache_dir / f"{self._key(url)}.html"

    def get(self, url: str) -> str | None:
        path = self._path(url)
        if not path.exists():
            return None
        try:
            mtime = path.stat().st_mtime
            age = time.time() - mtime
            if age > self.ttl_seconds:
                return None
            return path.read_text(encoding="utf-8")
        except OSError:
            return None

    def set(self, url: str, html: str) -> None:
        path = self._path(url)
        try:
            path.write_text(html, encoding="utf-8")
        except OSError as exc:
            logger.warning("HF cache write failed", extra={"url": url, "error": str(exc)})


def _hf_headers() -> dict[str, str]:
    return {"User-Agent": _HF_USER_AGENT}


def fetch_leaderboard_page(
    client: httpx.Client,
    cache: HFCache,
    url: str,
    max_retries: int = 3,
) -> str:
    """Fetch an HF leaderboard page, using the disk cache when available.

    Retries on 429 or 5xx with exponential backoff (2^attempt seconds). Returns
    the raw HTML.
    """
    cached = cache.get(url)
    if cached is not None:
        return cached

    for attempt in range(max_retries):
        try:
            response = client.get(url, headers=_hf_headers(), timeout=30.0)
        except httpx.HTTPError as exc:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt
            logger.warning(
                "HF request error; retrying",
                extra={"url": url, "attempt": attempt + 1, "wait": wait, "error": str(exc)},
            )
            time.sleep(wait)
            continue

        if response.status_code in {429, 502, 503, 504}:
            if attempt == max_retries - 1:
                response.raise_for_status()
            wait = 2 ** attempt
            logger.warning(
                "HF rate limit / server error; retrying",
                extra={"url": url, "status": response.status_code, "attempt": attempt + 1, "wait": wait},
            )
            time.sleep(wait)
            continue

        response.raise_for_status()
        html = response.text
        cache.set(url, html)
        return html

    raise RuntimeError(f"Failed to fetch HF leaderboard page: {url}")


def extract_usernames_from_html(html: str) -> list[tuple[str, str | None]]:
    """Heuristic extraction of HF usernames from a leaderboard HTML page.

    Returns a list of ``(username, display_name)``. The parsing is intentionally
    loose: HF markup changes over time, so we collect any ``a[href^="/"]`` whose
    href looks like a user profile and whose text contains an ``@`` prefix or a
    recognizable display name.
    """
    parser = HTMLParser(html)
    results: list[tuple[str, str | None]] = []
    seen: set[str] = set()

    for node in parser.css("a"):
        href = node.attributes.get("href", "")
        if not href:
            continue

        # HF user profile links are /{username}. Ignore /datasets, /models, etc.
        path = urlparse(href).path.strip("/")
        if not path or "/" in path:
            continue

        username = path.lower()
        if username in {"login", "join", "logout", "settings", "api", "docs", "blog", "about"}:
            continue
        if username in seen:
            continue

        text = node.text(deep=True) or ""
        text = text.strip()
        display_name: str | None = None
        if text:
            # If the link text contains "@username", the other part is likely the display name
            if f"@{username}" in text.lower():
                parts = text.split("@")
                if parts and parts[0].strip():
                    display_name = parts[0].strip()
            else:
                display_name = text

        seen.add(username)
        results.append((username, display_name))

    return results


def fetch_profile_page(client: httpx.Client, username: str) -> dict[str, Any]:
    """Fetch a HuggingFace user's profile page and extract structured info.

    Tries the public JSON API first, falls back to HTML parsing if that fails.
    """
    url = f"{_HF_BASE}/{username}"
    api_url = f"{_HF_BASE}/api/users/{username}"

    try:
        response = client.get(api_url, headers=_hf_headers(), timeout=30.0)
        if response.status_code == 200:
            return response.json()
    except httpx.HTTPError as exc:
        logger.warning(
            "HF API profile fetch failed; falling back to HTML",
            extra={"username": username, "error": str(exc)},
        )

    try:
        response = client.get(url, headers=_hf_headers(), timeout=30.0)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        logger.warning(
            "HF HTML profile fetch failed",
            extra={"username": username, "error": str(exc)},
        )
        return {"username": username, "error": str(exc)}

    html = response.text
    parser = HTMLParser(html)
    title = parser.css_first("title")
    display_name = title.text().split("(")[0].strip() if title and title.text() else None

    bio = ""
    for selector in ['[data-target="ProfileHeaderBio"]'], [".profile-bio"], ["main .prose"]:
        el = parser.css_first(selector[0])
        if el:
            bio = el.text(deep=True) or ""
            break

    location = ""
    for selector in ['[data-target="ProfileHeaderLocation"]'], [".profile-location"]:
        el = parser.css_first(selector[0])
        if el:
            location = el.text(deep=True) or ""
            break

    organization = ""
    for selector in ['[data-target="ProfileHeaderOrganization"]'], [".profile-organization"]:
        el = parser.css_first(selector[0])
        if el:
            organization = el.text(deep=True) or ""
            break

    links: list[str] = []
    for node in parser.css("a"):
        href = node.attributes.get("href", "")
        if href.startswith("http") and "huggingface.co" not in href:
            links.append(href)

    return {
        "username": username,
        "display_name": display_name,
        "bio": bio,
        "organization": organization,
        "location": location,
        "linked_sites": links,
    }


def looks_turkish(
    username: str,
    display_name: str | None,
    names: dict,
) -> bool:
    """Return True if the username or display name signals Turkish origin."""
    text = " ".join(part for part in (username, display_name or "") if part)

    if contains_turkish_diacritic(text, names.get("diacritic_chars", "")):
        return True

    if matches_turkish_name(text, names):
        return True

    return False


def infer_candidates_from_huggingface(
    client: httpx.Client,
    names_path: Path | None = None,
    leaderboard_urls: list[str] | None = None,
    pages: int = 3,
    delay_seconds: float = 1.0,
) -> list[Candidate]:
    """Scrape HF leaderboards, filter by Turkish signals, and build Candidates.

    Uses a 24h disk cache. Returns ``Candidate`` objects with ``source='huggingface'``.
    """
    names_path = names_path or config.TURKISH_NAMES_PATH
    try:
        names = load_turkish_names(names_path)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        logger.warning(
            "Could not load Turkish names; using fallback",
            extra={"path": str(names_path), "error": str(exc)},
        )
        names = default_turkish_names()

    cache = HFCache(
        cache_dir=Path("data/.hf_cache"),
        ttl_seconds=config.HF_CACHE_TTL_SECONDS,
    )
    urls = leaderboard_urls or DEFAULT_LEADERBOARD_URLS
    candidates: list[Candidate] = []
    seen_usernames: set[str] = set()
    now = datetime.now(tz=timezone.utc)

    for base_url in urls:
        for page in range(1, pages + 1):
            if page == 1:
                url = base_url
            else:
                # HF uses ?p=2, ?p=3, ... for pagination
                separator = "&" if "?" in base_url else "?"
                url = f"{base_url}{separator}p={page}"

            try:
                html = fetch_leaderboard_page(client, cache, url)
            except (httpx.HTTPError, RuntimeError) as exc:
                logger.warning(
                    "Skipping HF leaderboard page",
                    extra={"url": url, "error": str(exc)},
                )
                continue

            for username, display_name in extract_usernames_from_html(html):
                if username in seen_usernames:
                    continue

                if not looks_turkish(username, display_name, names):
                    continue

                seen_usernames.add(username)
                profile = fetch_profile_page(client, username)
                hf_username = profile.get("username", username)
                display_name = profile.get("display_name") or display_name or hf_username

                candidate = Candidate(
                    canonical_id=f"hf:{hf_username}",
                    source="huggingface",
                    github_username=None,
                    hf_username=hf_username,
                    display_name=display_name,
                    profile_json={"huggingface": profile},
                    first_seen_at=now,
                    last_seen_at=now,
                )
                candidates.append(candidate)

            if delay_seconds > 0:
                time.sleep(delay_seconds)

    return candidates


def cross_match_github_hf(
    github_candidates: list[Candidate],
    hf_candidates: list[Candidate],
) -> list[Candidate]:
    """Merge GitHub and HuggingFace candidates by username or linked sites.

    Returns a new list containing cross-matched candidates with
    ``source='cross'``. GitHub-only and HuggingFace-only candidates are left
    untouched. The ``canonical_id`` for cross-matched candidates is
    ``cross:gh:{github_username}``.
    """
    # Index HF candidates by username and by linked sites
    hf_by_username: dict[str, Candidate] = {}
    hf_by_link: dict[str, Candidate] = {}

    for hf in hf_candidates:
        if not hf.hf_username:
            continue
        hf_by_username[hf.hf_username.lower()] = hf
        profile = hf.profile_json.get("huggingface", {})
        for link in profile.get("linked_sites", []):
            parsed = urlparse(link)
            key = f"{parsed.netloc}{parsed.path}".lower().rstrip("/")
            if key:
                hf_by_link[key] = hf

    merged: list[Candidate] = []
    cross_gh_usernames: set[str] = set()

    for gh in github_candidates:
        if not gh.github_username:
            merged.append(gh)
            continue

        gh_username = gh.github_username.lower()
        match: Candidate | None = hf_by_username.get(gh_username)

        # Try matching by linked sites in the GitHub profile / homepage
        if not match:
            profile = gh.profile_json.get("github", {})
            links: list[str] = []
            if profile.get("blog"):
                links.append(str(profile["blog"]))
            if profile.get("html_url"):
                links.append(str(profile["html_url"]))
            if profile.get("twitter_username"):
                links.append(f"https://twitter.com/{profile['twitter_username']}")
            homepage = gh.profile_json.get("github", {}).get("homepage")
            if homepage:
                links.append(str(homepage))

            for link in links:
                parsed = urlparse(link)
                key = f"{parsed.netloc}{parsed.path}".lower().rstrip("/")
                if key in hf_by_link:
                    match = hf_by_link[key]
                    break

        if match:
            cross_gh_usernames.add(gh_username)
            hf_username = match.hf_username
            display_name = gh.display_name or match.display_name
            merged_profile = dict(gh.profile_json)
            merged_profile.setdefault("huggingface", match.profile_json.get("huggingface", {}))
            merged_profile.setdefault("github", gh.profile_json.get("github", {}))
            cross = Candidate(
                canonical_id=f"cross:gh:{gh.github_username}",
                source="cross",
                github_username=gh.github_username,
                hf_username=hf_username,
                display_name=display_name,
                profile_json=merged_profile,
                first_seen_at=gh.first_seen_at,
                last_seen_at=gh.last_seen_at,
            )
            merged.append(cross)
        else:
            merged.append(gh)

    # Append HF-only candidates that were not cross-matched
    for hf in hf_candidates:
        if not hf.hf_username:
            merged.append(hf)
            continue
        if hf.hf_username.lower() not in cross_gh_usernames:
            merged.append(hf)

    return merged
