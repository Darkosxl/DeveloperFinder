from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest

from verified_inviter import main
from verified_inviter import config as config_module


def test_main_module_imports() -> None:
    """Smoke test: the main module imports and exposes all required functions."""
    assert callable(main.setup_logging)
    assert callable(main.build_http_clients)
    assert callable(main.discover_candidates)
    assert callable(main.process_candidate)
    assert callable(main.run_daily)
    assert callable(main.run_self_test)
    assert callable(main.main)


def test_build_http_clients_returns_three_clients() -> None:
    """build_http_clients should return github, hf, and exa clients."""
    with mock.patch.object(config_module, "GITHUB_TOKEN", "fake_gh_token"), \
         mock.patch.object(config_module, "EXA_API_KEY", "fake_exa_key"):
        clients = main.build_http_clients()
    assert set(clients.keys()) == {"github", "hf", "exa"}
    for client in clients.values():
        assert hasattr(client, "get")
        assert hasattr(client, "post")
        client.close()


def test_argparse_daily_dispatch() -> None:
    """Calling main with no args should dispatch to run_daily."""
    with mock.patch.object(sys, "argv", ["verified_inviter.main"]), \
         mock.patch.object(main, "run_daily") as mock_run_daily:
        main.main()
    mock_run_daily.assert_called_once()


def test_argparse_self_test_dispatch() -> None:
    """Calling main with --self-test should dispatch to run_self_test."""
    with mock.patch.object(sys, "argv", ["verified_inviter.main", "--self-test", "torvalds"]), \
         mock.patch.object(main, "run_self_test") as mock_run_self_test:
        main.main()
    mock_run_self_test.assert_called_once_with("torvalds")


def test_run_self_test_does_not_send(monkeypatch, tmp_path) -> None:
    """Self-test should write artifacts and never call send_pending_invites."""
    monkeypatch.setattr(config_module, "OUTBOX_DIR", tmp_path / "outbox")
    monkeypatch.setattr(config_module, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(config_module, "DB_PATH", tmp_path / "verified_inviter.db")
    monkeypatch.setattr(config_module, "COMPANIES_PATH", tmp_path / "exposure_companies.json")
    (tmp_path / "exposure_companies.json").write_text("[]")

    fake_profile = {"name": "Linus T", "login": "torvalds"}
    fake_repos = []

    with mock.patch("verified_inviter.main.github.get_user_profile", return_value=fake_profile), \
         mock.patch("verified_inviter.main.github.list_user_repos", return_value=fake_repos), \
         mock.patch("verified_inviter.email_send.send_pending_invites") as mock_send, \
         mock.patch("verified_inviter.main.SambaNovaClient") as MockLLM:
        MockLLM.return_value.close = mock.Mock()
        main.run_self_test("torvalds")

    mock_send.assert_not_called()
