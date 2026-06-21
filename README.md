# Verified Inviter Agent

A daily agent that discovers high-potential Turkish developers, analyzes their real technical work via Exa AI and SambaNova LLMs, matches them to companies in the Exposure network, and sends personalized invitations to apply to [Exposure Verified](https://exposureai.org/verified). It keeps full state in SQLite, never re-invites the same person, and runs on a VPS via cron.

---

## Setup

1. Copy the environment file and fill in your keys:

   ```bash
   cp .env.example .env
   ```

2. Install dependencies with `uv`:

   ```bash
   uv sync
   ```

3. Create the required data files:

   - `data/exposure_companies.json` — user-supplied list of Exposure companies:

     ```json
     [
       {
         "name": "Company Name",
         "what_they_are_building": "One-line description of what they build.",
         "website": "https://example.com"
       }
     ]
     ```

   - `data/turkish_names.json` — list of Turkish names/suffixes for filtering HuggingFace leaderboards (a fallback is built in if the file is missing).

---

## Self-test (run this first)

Before running discovery or sending any email, run the pipeline on a single GitHub profile to validate the LLM stages and inspect the artifacts:

```bash
uv run python -m verified_inviter.main --self-test <your_github_username>
```

Artifacts are written to `outbox/self-test-YYYY-MM-DD-HHMMSS/<username>/`:

- `knowledge.json`
- `technical_verdict.json`
- `match.json`
- `draft_email.txt`

No emails are sent in self-test mode.

---

## Daily run (dry-run by default)

```bash
uv run python -m verified_inviter.main
```

With `DRY_RUN=1` (the default), the agent discovers candidates, runs all LLM stages, drafts up to 5 emails, and writes them to `outbox/draft-YYYY-MM-DD/` as `.eml` files plus a `summary.md`. Nothing is sent.

Rejected candidates are logged in `outbox/draft-YYYY-MM-DD/rejects.md` with the technical judge's reasoning.

---

## Switching to live sends

Set `DRY_RUN=0` in `.env`:

```text
DRY_RUN=0
```

Live sends go through Resend using the sender configured in `.env`. Note: public email discovery is not implemented in v1, so emails are only sent when a recipient address is available. Candidates with no discovered email are recorded with `status='failed'` and skipped.

---

## Cron example

Run once a day at 09:00 Europe/Istanbul:

```cron
0 9 * * * cd /path/to/DeveloperFinder && /path/to/uv run python -m verified_inviter.main >> logs/$(date +\%F).log 2>&1
```

Make sure the cron environment has the same `.env` loaded or source it before the command.

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GITHUB_TOKEN` | *(required)* | GitHub personal access token for search and repo enumeration. |
| `SAMBA_API_KEY` | *(required)* | SambaNova API key. |
| `SAMBA_MODEL` | `gemma-4-31b-it` | SambaNova chat model id. |
| `SAMBA_BASE_URL` | `https://api.sambanova.ai/v1` | SambaNova API base URL. |
| `EXA_API_KEY` | *(required)* | Exa AI API key for fetching repo contents. |
| `RESEND_API_KEY` | *(required)* | Resend API key for live email sending. |
| `SENDER_EMAIL` | `bscemarslan@gmail.com` | From/reply-to address. |
| `SENDER_NAME` | `Exposure Verified` | Display name used in the email sender field. |
| `DRY_RUN` | `1` | `1` / `true` / `yes` drafts only; `0` / `false` / `no` sends live. |
| `DB_PATH` | `data/verified_inviter.db` | SQLite database path. |
| `COMPANIES_PATH` | `data/exposure_companies.json` | Exposure companies JSON. |
| `TURKISH_NAMES_PATH` | `data/turkish_names.json` | Turkish name filter JSON. |
| `DAILY_INVITE_CAP` | `5` | Max emails per run. |
| `REPO_RECENCY_DAYS` | `180` | Ignore repos/candidates with no activity in this many days. |
| `SKIP_REJUDGE_DAYS` | `30` | Do not re-judge a skipped candidate for this many days. |
| `MAX_RELEVANT_REPOS_PER_CANDIDATE` | `5` | Max Exa-fetched repos per candidate. |
| `MAX_CANDIDATES_PER_RUN` | `20` | Max candidates to process per run. |
| `LOG_LEVEL` | `INFO` | Logging level. |
| `LOG_DIR` | `logs` | Directory for `verified_inviter.log` and `runs.log`. |
| `OUTBOX_DIR` | `outbox` | Directory for dry-run drafts and self-test artifacts. |

---

## Data files

- `data/exposure_companies.json` — user-supplied list of Exposure companies. Shape: `[{name, what_they_are_building, website}]`.
- `data/turkish_names.json` — user-supplied or generated Turkish name filter.
- `data/verified_inviter.db` — SQLite state database with `candidates`, `repo_verdicts`, `knowledge`, `technical_verdicts`, `matches`, `invites`, and `runs` tables.

---

## Output

- **Dry-run drafts:** `outbox/draft-YYYY-MM-DD/draft_{id}.eml` + `summary.md`
- **Rejects:** `outbox/draft-YYYY-MM-DD/rejects.md`
- **Self-test artifacts:** `outbox/self-test-YYYY-MM-DD-HHMMSS/<username>/`
- **Logs:** `logs/verified_inviter.log` (rotating, 10 MB × 5 backups) + `logs/runs.log`
- **State:** `data/verified_inviter.db`

---

## Pipeline overview

1. **Discovery** — GitHub users in Turkey + HuggingFace leaderboard scraping, cross-matched.
2. **Repo enumeration** — list every recent repo owned by the candidate.
3. **Repo relevance** — LLM judges each repo as genuine technical work vs. slop.
4. **Content fetch** — Exa AI fetches the contents of relevant repos.
5. **Knowledge extraction** — LLM writes what the candidate actually knows.
6. **Technical judge** — LLM decides if the candidate is worth inviting and still seed-stage.
7. **Company matching** — LLM picks one matching Exposure company or `none`.
8. **Email drafting** — LLM writes a personalized or generic Turkish+English email.
9. **Send / dry-run** — Resend live sends or `.eml` drafts, capped at 5 per day.

---

## License

Internal project for Exposure AI.
