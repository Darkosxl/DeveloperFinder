from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _as_bool(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _as_int(value: str | None, default: int) -> int:
    if value is None or value.strip() == "":
        return default
    return int(value)


def _as_float(value: str | None, default: float) -> float:
    if value is None or value.strip() == "":
        return default
    return float(value)


def _as_path(value: str | None, default: str) -> Path:
    if value is None or value.strip() == "":
        return Path(default)
    return Path(value)


GITHUB_TOKEN: str = os.getenv("GITHUB_TOKEN", "")
SAMBA_API_KEY: str = os.getenv("SAMBA_API_KEY", "")
SAMBA_MODEL: str = os.getenv("SAMBA_MODEL", "gemma-4-31b-it")
SAMBA_BASE_URL: str = os.getenv("SAMBA_BASE_URL", "https://api.sambanova.ai/v1")
EXA_API_KEY: str = os.getenv("EXA_API_KEY", "")
RESEND_API_KEY: str = os.getenv("RESEND_API_KEY", "")

SENDER_EMAIL: str = os.getenv("SENDER_EMAIL", "bscemarslan@gmail.com")
SENDER_NAME: str = os.getenv("SENDER_NAME", "Exposure Verified")

DRY_RUN: bool = _as_bool(os.getenv("DRY_RUN", "1"))
DB_PATH: Path = _as_path(os.getenv("DB_PATH"), "data/verified_inviter.db")
COMPANIES_PATH: Path = _as_path(os.getenv("COMPANIES_PATH"), "data/exposure_companies.json")
TURKISH_NAMES_PATH: Path = _as_path(os.getenv("TURKISH_NAMES_PATH"), "data/turkish_names.json")

DAILY_INVITE_CAP: int = _as_int(os.getenv("DAILY_INVITE_CAP"), 5)
REPO_RECENCY_DAYS: int = _as_int(os.getenv("REPO_RECENCY_DAYS"), 180)
SKIP_REJUDGE_DAYS: int = _as_int(os.getenv("SKIP_REJUDGE_DAYS"), 30)
MAX_RELEVANT_REPOS_PER_CANDIDATE: int = _as_int(os.getenv("MAX_RELEVANT_REPOS_PER_CANDIDATE"), 5)
MAX_CANDIDATES_PER_RUN: int = _as_int(os.getenv("MAX_CANDIDATES_PER_RUN"), 30)

REPO_JUDGE_TEMP: float = _as_float(os.getenv("REPO_JUDGE_TEMP"), 0.1)
KNOWLEDGE_TEMP: float = _as_float(os.getenv("KNOWLEDGE_TEMP"), 0.3)
TECH_JUDGE_TEMP: float = _as_float(os.getenv("TECH_JUDGE_TEMP"), 0.2)
MATCHING_TEMP: float = _as_float(os.getenv("MATCHING_TEMP"), 0.2)
EMAIL_TEMP: float = _as_float(os.getenv("EMAIL_TEMP"), 0.7)

LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
LOG_DIR: Path = _as_path(os.getenv("LOG_DIR"), "logs")
OUTBOX_DIR: Path = _as_path(os.getenv("OUTBOX_DIR"), "outbox")

HF_CACHE_TTL_SECONDS: int = _as_int(os.getenv("HF_CACHE_TTL_SECONDS"), 24 * 3600)
GITHUB_RATE_LIMIT_HEADROOM: int = _as_int(os.getenv("GITHUB_RATE_LIMIT_HEADROOM"), 100)

# --- Dashboard / Server ---
DASHBOARD_PORT: int = _as_int(os.getenv("DASHBOARD_PORT"), 8000)
FLASK_SECRET_KEY: str = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-me")

# --- Auth (Google OAuth) ---
GOOGLE_CLIENT_ID: str = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET: str = os.getenv("GOOGLE_CLIENT_SECRET", "")
ALLOWED_EMAIL: str = os.getenv("ALLOWED_EMAIL", "bscemarslan@gmail.com")

# --- Scheduler ---
SCHEDULER_INTERVAL_MINUTES: int = _as_int(os.getenv("SCHEDULER_INTERVAL_MINUTES"), 1440)
SCHEDULER_AUTOSTART: bool = _as_bool(os.getenv("SCHEDULER_AUTOSTART", "1"))
