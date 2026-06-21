from __future__ import annotations

import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx

from verified_inviter.discovery import github, huggingface, turkish_names
from verified_inviter.models import Candidate, Repo


def test_load_turkish_names(tmp_path: Path) -> None:
    path = tmp_path / "names.json"
    path.write_text(
        '{"given_names": ["ahmet", "ayşe"], "surname_suffixes": ["oğlu"], "diacritic_chars": "ışçğüö"}',
        encoding="utf-8",
    )
    data = turkish_names.load_turkish_names(path)
    assert data["given_names"] == ["ahmet", "ayşe"]
    assert data["surname_suffixes"] == ["oğlu"]
    assert data["diacritic_chars"] == "ışçğüö"


def test_default_turkish_names() -> None:
    data = turkish_names.default_turkish_names()
    assert len(data["given_names"]) >= 200
    assert "oğlu" in data["surname_suffixes"]
    assert "ı" in data["diacritic_chars"]


def test_contains_turkish_diacritic() -> None:
    chars = "ı ş ç ğ ü ö İ Ş Ç Ğ Ü Ö"
    assert turkish_names.contains_turkish_diacritic("Istanbul", chars) is False
    assert turkish_names.contains_turkish_diacritic("İstanbul", chars) is True
    assert turkish_names.contains_turkish_diacritic("çağrı", chars) is True
    assert turkish_names.contains_turkish_diacritic("", chars) is False


def test_matches_turkish_name() -> None:
    names = turkish_names.default_turkish_names()
    assert turkish_names.matches_turkish_name("Ahmet Yılmaz", names) is True
    assert turkish_names.matches_turkish_name("Oğuz Oğuzoğlu", names) is True
    assert turkish_names.matches_turkish_name("John Smith", names) is False
    # "ali" as a substring should not match
    assert turkish_names.matches_turkish_name("calibrate", names) is False


def test_repo_from_dict() -> None:
    repo = github.repo_from_dict(
        {
            "owner": {"login": "octocat"},
            "name": "hello-world",
            "description": "My first repo",
            "language": "Python",
            "stargazers_count": 42,
            "forks_count": 3,
            "topics": ["demo"],
            "has_readme": True,
            "homepage": "https://example.com",
            "pushed_at": "2024-06-01T12:00:00Z",
            "created_at": "2023-01-01T00:00:00Z",
            "fork": False,
        }
    )
    assert isinstance(repo, Repo)
    assert repo.owner == "octocat"
    assert repo.name == "hello-world"
    assert repo.stars == 42
    assert repo.pushed_at.year == 2024


def test_list_user_repos_pagination() -> None:
    """list_user_repos should stop after the cutoff and paginate correctly."""
    now = datetime.now(tz=timezone.utc)
    recent_iso = now.isoformat().replace("+00:00", "Z")
    old_iso = "2023-01-01T00:00:00Z"

    def fake_get(url, **kwargs):
        mock = MagicMock()
        mock.status_code = 200
        mock.headers = {"x-ratelimit-remaining": "5000"}
        params = kwargs.get("params", {})
        if params.get("page") == 1:
            mock.json.return_value = [
                {
                    "owner": {"login": "u"},
                    "name": "repo1",
                    "description": None,
                    "language": "Rust",
                    "stargazers_count": 10,
                    "forks_count": 0,
                    "topics": [],
                    "has_readme": True,
                    "homepage": None,
                    "pushed_at": recent_iso,
                    "created_at": old_iso,
                    "fork": False,
                }
            ]
        else:
            mock.json.return_value = []
        return mock

    with patch.object(httpx.Client, "get", side_effect=fake_get):
        with httpx.Client() as client:
            repos = github.list_user_repos(client, "u")
    assert len(repos) == 1
    assert repos[0].name == "repo1"


def test_infer_candidates_from_github() -> None:
    now = datetime.now(tz=timezone.utc)
    recent_iso = now.isoformat().replace("+00:00", "Z")

    search_response = {
        "items": [{"login": "turkish_dev"}],
        "incomplete_results": False,
    }
    profile = {
        "login": "turkish_dev",
        "name": "Turkish Dev",
        "location": "Istanbul",
    }
    events = [{"created_at": recent_iso}]

    def fake_get(url, **kwargs):
        mock = MagicMock()
        mock.status_code = 200
        mock.headers = {"x-ratelimit-remaining": "5000"}
        if "search/users" in url:
            mock.json.return_value = search_response
        elif "/users/turkish_dev/events" in url:
            mock.json.return_value = events
        elif "/users/turkish_dev" in url:
            mock.json.return_value = profile
        else:
            mock.json.return_value = []
        return mock

    with patch.object(httpx.Client, "get", side_effect=fake_get):
        with httpx.Client() as client:
            candidates = github.infer_candidates_from_github(client)

    assert len(candidates) == 1
    c = candidates[0]
    assert c.canonical_id == "gh:turkish_dev"
    assert c.source == "github"
    assert c.github_username == "turkish_dev"
    assert c.display_name == "Turkish Dev"


def test_cross_match_github_hf() -> None:
    now = datetime.now(tz=timezone.utc)
    gh = Candidate(
        canonical_id="gh:ali",
        source="github",
        github_username="ali",
        hf_username=None,
        display_name="Ali V",
        profile_json={"github": {"html_url": "https://github.com/ali"}},
        first_seen_at=now,
        last_seen_at=now,
    )
    hf = Candidate(
        canonical_id="hf:ali",
        source="huggingface",
        github_username=None,
        hf_username="ali",
        display_name="Ali V",
        profile_json={"huggingface": {"linked_sites": ["https://github.com/ali"]}},
        first_seen_at=now,
        last_seen_at=now,
    )
    merged = huggingface.cross_match_github_hf([gh], [hf])
    assert len(merged) == 1
    assert merged[0].canonical_id == "cross:gh:ali"
    assert merged[0].source == "cross"
    assert merged[0].github_username == "ali"
    assert merged[0].hf_username == "ali"


def test_looks_turkish() -> None:
    names = turkish_names.default_turkish_names()
    assert huggingface.looks_turkish("ahmet_tr", "Ahmet Yılmaz", names) is True
    assert huggingface.looks_turkish("random_dev", "John Doe", names) is False
    assert huggingface.looks_turkish("cagri", "Çağrı", names) is True


def test_extract_usernames_from_html() -> None:
    html = """
    <html><body>
    <a href="/ahmet_tr">Ahmet @ahmet_tr</a>
    <a href="/models/foo">foo model</a>
    <a href="/random">random</a>
    </body></html>
    """
    found = huggingface.extract_usernames_from_html(html)
    usernames = [u for u, _ in found]
    assert "ahmet_tr" in usernames
    assert "models" not in usernames


if __name__ == "__main__":
    test_load_turkish_names(Path.cwd())
    test_default_turkish_names()
    test_contains_turkish_diacritic()
    test_matches_turkish_name()
    test_repo_from_dict()
    test_list_user_repos_pagination()
    test_infer_candidates_from_github()
    test_cross_match_github_hf()
    test_looks_turkish()
    test_extract_usernames_from_html()
    print("discovery smoke tests passed")
