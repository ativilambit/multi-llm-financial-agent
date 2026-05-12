# Multi-LLM equity analyst

Python CLI that renders a Jinja2 equity/options prompt from YAML config, fans out to multiple LLM providers in parallel, and synthesizes a consensus report. Optional **iterative** mode runs a LangGraph loop: multi-provider fan-out, synthesis with per-round confidence parsing, web-grounded verification, routing, and final packaging with SQLite checkpointing for resume.

See [CHANGELOG.md](CHANGELOG.md) for the change history.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### PDF output (optional)

Standard and iterative runs still write **Markdown** (`.md`) for every primary artifact. When **`pdf_output_enabled`** is true (the default), the tool also writes a sibling **`.pdf`** next to the same path—for example `outputs/<run>/synthesis.pdf` beside `synthesis.md`, and under `iterations/` for per-round `iteration_<n>_synthesis.pdf`, `iteration_<n>_verify.pdf`, and the consolidated `iteration_<n>.pdf` next to `iteration_<n>.md`.

Rendering uses **WeasyPrint** (HTML via the Python **markdown** library) so results look like formatted documents without bundling Chromium. WeasyPrint relies on native **Pango/Cairo** stacks:

```bash
brew install pango cairo gdk-pixbuf libffi
```

If WeasyPrint fails to import or render (missing system libraries, broken install), the run **logs a warning** and **continues** without PDFs; Markdown outputs are unchanged.

- **Disable for one run:** `python -m equity_analyst run ... --no-pdf` (or `--pdf` to force on).
- **Disable via environment:** `PDF_OUTPUT_ENABLED=false` (same truthy/falsy rules as other env flags: `1` / `true` / `yes` / `on` enable; anything else disables when the variable is set).

**Google Drive:** the uploader walks the run output directory with `os.walk` and uploads every non-dotfile, preserving subpaths—**`.pdf` files in `iterations/` are included** automatically with the Markdown sources.

## API keys

Copy `.env.example` to `.env` and set keys for the providers you enable in config:

- `ANTHROPIC_API_KEY`
- `OPENAI_API_KEY`
- `GEMINI_API_KEY`
- `XAI_API_KEY` (Grok)

## Database (Postgres)

The Postgres DB layer is **additive**: it stores structured metadata for querying/calibration, but it **does not replace** file artifacts. The human-readable source of truth remains:

- `outputs/<run-id>/run.json`
- `outputs/<run-id>/synthesis.md` (and optional PDFs)
- `outputs/<run-id>/outcome.json`
- `outputs/<run-id>/predictions_extract.json` (optional fallback when Postgres is down or `db_enabled` is false; see prediction extraction below)
- `outputs/outcomes_registry.jsonl`

### Connection

Set `DATABASE_URL` in `.env` at the repo root. It is loaded automatically by the `equity_analyst` CLI, Alembic (`migrations/env.py`), and `scripts/setup_db.sh`, so you do not need to `source .env` manually for those entry points.

`python-dotenv` is used with `override=False` (shell exports still win if you set them explicitly).

```bash
DATABASE_URL=postgresql+psycopg://college_brain:college_brain@localhost:5432/multi_llm_equity_runs
```

### Bootstrap

Create the DB (if missing) and apply migrations:

```bash
scripts/setup_db.sh
```

This script requires `psql` on your `PATH` (Postgres client).

For schema bumps:

```bash
alembic upgrade head
```

### Sample queries

Hit rate by symbol (direction labels recorded):

```sql
SELECT
  r.symbol,
  COUNT(*) AS outcomes
FROM runs r
JOIN outcomes o ON o.run_id = r.run_id
GROUP BY 1
ORDER BY outcomes DESC;
```

Recent runs missing outcomes:

```sql
SELECT r.run_id, r.symbol, r.started_at_utc, r.earnings_date
FROM runs r
LEFT JOIN outcomes o ON o.run_id = r.run_id
WHERE o.run_id IS NULL
ORDER BY r.started_at_utc DESC
LIMIT 50;
```

Calibration buckets (after running **prediction extraction** so `predictions` rows exist):

```sql
SELECT
  width_bucket(p.predicted_probability_up, 0.0, 1.0, 10) AS bucket,
  COUNT(*) AS n
FROM predictions p
GROUP BY 1
ORDER BY 1;
```

## Google Drive auto-upload

After each standard or iterative run finishes writing `outputs/<run-id>/` (including `run.json`), you can optionally mirror that folder to Google Drive using a **Google Cloud service account** JSON key. The CLI resolves your configured `drive_root_folder_id` / `DRIVE_ROOT_FOLDER_ID`, then uploads under a **child folder** named exactly **`prod`** (production runs, the default) or **`test`** (test runs), creating that child folder on first upload if it does not exist. Under that environment folder it creates a subfolder named after the run id, uploads every file (preserving paths such as `iterations/`), skips dotfiles, skips iterative checkpoint basenames (`checkpoint.sqlite`, `checkpoint.sqlite-wal`, `checkpoint.sqlite-shm`, `checkpoint.sqlite-journal`), and appends `drive_folder_url` plus `run_environment`, `drive_upload_parent_folder_id`, and `drive_upload_parent_folder_name` to `run.json` on success. Iterative runs also append the link to the footer of `synthesis.md`. Upload failures are logged and never fail the analysis run.

At startup the CLI **preflights** the configured root folder via the Drive API: the folder must live on a **Google Shared Drive** (Workspace “Team Drive”). If it does not, uploads are disabled for that run and a clear warning is printed so you do not spend a long model run only to hit `storageQuotaExceeded` on upload.

### Google Drive setup (service account)

1. In [Google Cloud Console](https://console.cloud.google.com/), create or pick a project, then **IAM & Admin → Service Accounts → Create service account** (any name).
2. **Keys → Add key → Create new key → JSON** and download the key file (keep it private; do not commit it). Point `drive_credentials_path` / `DRIVE_CREDENTIALS_PATH` at this file.
3. **APIs & Services → Library → Google Drive API → Enable**.
4. **Use a folder on a Shared Drive, not personal “My Drive”.** Service accounts have **no personal Drive storage quota**; uploads into a personal folder fail with HTTP 403 `storageQuotaExceeded` even if the folder is shared with the SA. Create or move your destination folder inside a [Shared Drive](https://developers.google.com/workspace/drive/api/guides/about-shareddrives) (requires Google Workspace).
5. **Share** that Shared Drive folder (or grant membership on the drive) with the service account’s `client_email` as **Content Manager** (or a role that includes **create/upload** and **add files**). Editor on a personal folder is **not** enough if the folder is not on a Shared Drive.
6. Set `drive_root_folder_id` / `DRIVE_ROOT_FOLDER_ID` to the folder id from the URL: `https://drive.google.com/drive/folders/<this-part>`.

If you cannot use Shared Drives, use **OAuth user mode** below so uploads run as your personal Gmail account (files count against your own Drive quota).

### Google Drive setup (OAuth user flow — for personal Gmail)

Use this when you are **not** on Google Workspace / Shared Drives. The CLI uploads with **your** Google account after a one-time browser consent.

1. In [Google Cloud Console](https://console.cloud.google.com/), pick a project → **APIs & Services → Library** → enable **Google Drive API**.
2. **APIs & Services → OAuth consent screen**: configure a consent screen (External user type is fine for personal use). Add scope **`https://www.googleapis.com/auth/drive`** (see scope note below).
3. **APIs & Services → Credentials → Create credentials → OAuth client ID** → Application type: **Desktop app** → create → **Download JSON** and save it to your configured path (default below).
4. Install the CLI deps, set `drive_auth_mode: oauth_user` (or `DRIVE_AUTH_MODE=oauth_user`), and run once:

   ```bash
   python -m equity_analyst.drive_oauth_setup
   ```

   (Optionally pass `--config path/to.yaml` so paths come from YAML.) A browser opens; sign in with the Gmail account that should own the uploads and grant the Drive scope. The refresh token is saved to your token path (default `~/.config/multi-llm-equity-analyst/oauth_token.json`, overridable with `drive_oauth_token_path` / `DRIVE_OAUTH_TOKEN_PATH`). After upgrading from a prior version, delete your existing `oauth_token.json` and re-run the setup command — old tokens have the narrower scope.

5. Set **`drive_root_folder_id`** / **`DRIVE_ROOT_FOLDER_ID`** to a folder id from your normal Drive URL (`https://drive.google.com/drive/folders/<id>`). No Shared Drive is required; the signed-in user must own or have write access to that folder.

6. On later runs the tool **silently refreshes** the access token. If you revoke the app in Google Account settings or the refresh fails, run `python -m equity_analyst.drive_oauth_setup` again.

**OAuth scope (`drive` vs `drive.file`):** OAuth user uploads use **`https://www.googleapis.com/auth/drive`**, so the app can resolve and write into **folders you created manually** in Drive (not only folders or files the app created, which is all **`drive.file`** can see). That is appropriate for a **single-user personal tool** on your own Gmail. Full `drive` means the app can read and write **any** file in that account’s Drive the API allows; it is **not** ideal for multi-tenant or broadly distributed apps, where you would prefer a narrower scope and Shared-Drive–style isolation.

### Configuration

The usual way to enable Drive upload without editing every YAML file is to add the variables to **`.env`** next to your API keys (copy from [`.env.example`](.env.example)). The CLI calls `python-dotenv` at startup with `override=False`, so anything you already exported in the shell still wins over `.env`.

```bash
# .env (optional) — service account (Shared Drive)
DRIVE_UPLOAD_ENABLED=true
DRIVE_CREDENTIALS_PATH=/Users/you/secrets/equity-analyst-drive-sa.json
DRIVE_ROOT_FOLDER_ID=1AbCdEf...

# .env (optional) — OAuth user (personal Gmail)
# DRIVE_UPLOAD_ENABLED=true
# DRIVE_AUTH_MODE=oauth_user
# DRIVE_OAUTH_CLIENT_SECRETS_PATH=/Users/you/secrets/google-oauth-desktop.json
# DRIVE_OAUTH_TOKEN_PATH=/Users/you/.config/multi-llm-equity-analyst/oauth_token.json
# DRIVE_ROOT_FOLDER_ID=1AbCdEf...
```

**Precedence:** CLI flags (`--upload-to-drive` / `--no-upload-to-drive`, `--drive-folder-id`) override the resolved config. After that: if `DRIVE_UPLOAD_ENABLED` is set in the environment (shell **or** values loaded from `.env`), it overrides the YAML boolean for `drive_upload_enabled`. For `drive_credentials_path` and `drive_root_folder_id`, non-empty YAML entries win; otherwise the environment supplies them. Between shell and `.env`, **`load_dotenv(override=False)`** keeps existing shell variables and only fills names that are not already set—so **shell > `.env`** for the same variable name.

You can still set the same fields in YAML (paths can use shell-style expansion such as `"${HOME}/secrets/..."` when your shell expands them before load, or use absolute paths):

```yaml
drive_upload_enabled: true
drive_credentials_path: "${HOME}/secrets/equity-analyst-drive-sa.json"
drive_root_folder_id: "1AbCdEf...your-folder-id..."
```

OAuth user mode (YAML example):

```yaml
drive_upload_enabled: true
drive_auth_mode: oauth_user
drive_oauth_client_secrets_path: "${HOME}/secrets/google-oauth-desktop.json"
drive_oauth_token_path: "${HOME}/.config/multi-llm-equity-analyst/oauth_token.json"
drive_root_folder_id: "1AbCdEf...your-personal-folder-id..."
```

Environment keys (optional; typically set in `.env` or the shell):

- `DRIVE_UPLOAD_ENABLED=true|false`
- `DRIVE_CREDENTIALS_PATH`
- `DRIVE_ROOT_FOLDER_ID`
- `DRIVE_AUTH_MODE=service_account|oauth_user`
- `DRIVE_OAUTH_CLIENT_SECRETS_PATH` (path to Desktop OAuth client JSON; used by `drive_oauth_setup`)
- `DRIVE_OAUTH_TOKEN_PATH` (saved refresh token JSON)
- `RUN_ENVIRONMENT=production|test` (optional; routes uploads under the `prod` or `test` subfolder; overrides YAML `run_environment` when set)

Per-run CLI overrides:

```bash
python -m equity_analyst run --config ... --upload-to-drive --drive-folder-id <folder-id>
python -m equity_analyst run --config ... --no-upload-to-drive
python -m equity_analyst run --config ... --drive-auth-mode oauth_user
python -m equity_analyst run --config ... --environment test
python -m equity_analyst run --config ... --env production
```

### Caveats

- With a service account, file bytes are stored against **Shared Drive** (or delegated-user) quota—not “free” personal My Drive space for the SA.
- Files land in the folder you configure; anyone who can **view** that Drive folder can see uploaded runs. Treat the folder and sharing like sensitive storage.

## Configs

Stock-specific YAML lives under `configs/`. **Workflow:** pick the **MNDY** or **CRCL** pair below as a **template**, copy both YAMLs to new filenames (`<symbol>_YYYY_MM_DD.yaml` and `_fast.yaml`), then edit symbol, company name, session labels, dates, optional price hints, and any symbol-specific lookbacks. **Price fields** (`today_low` / `today_high` / `current_price`, or the aliases `reference_session_*` / `reference_last_price`) are **optional, unverified hints** for orientation only—the rendered prompt tells models to **fetch and cite** the **last regular-session official closing price** (and session high/low) via **web_search** and not to treat YAML numbers as ground truth. **`earnings_timing`** is also **optional**: if you omit it, the equity prompt requires **web_search** verification of the actual **BMO / AMC / during-hours** schedule for `earnings_date` (with URL citation); if you set it, that label is treated as the stated schedule while still allowing cross-checks.

| File | Use case |
|------|----------|
| `configs/mndy_2026_05_08.yaml` | Default **MNDY** run: all four fan-out providers (**Anthropic**, **OpenAI**, **Grok**, **Gemini Flash**) use long timeouts; web search follows each provider’s default (typically on). **Synthesis** runs on **Gemini 3.1 Pro Preview**. Best for highest-quality grounded research. |
| `configs/mndy_2026_05_08_fast.yaml` | **MNDY** hybrid speed: **OpenAI** alone runs deep `web_search`; **Anthropic**, **Grok**, and **Gemini Flash** reason without search; **Gemini 3.1 Pro Preview** synthesizer has no extra search. Shorter wall-clock for iteration. |
| `configs/crcl_2026_05_08.yaml` | Same layout as the MNDY standard config for **CRCL** (Circle Internet Group, NYSE: **CRCL**), aligned to the **May 11, 2026** earnings cycle. |
| `configs/crcl_2026_05_08_fast.yaml` | **CRCL** hybrid fast config (mirrors MNDY `_fast`: one grounded **OpenAI** search, fast fan-out otherwise). |

### Running multiple symbols

The 2026-05-10 batch ships ten standard configs that all mirror **CRCL**’s provider structure (Anthropic Opus → OpenAI → Grok → Gemini Flash fan-out, Gemini Pro synthesizer, Gemini verifier, 600 s timeouts, `historical_quarters: 6`). Price fields are `null` on purpose — every provider must source the last regular-session close via `web_search`. **`earnings_timing`** is omitted so **call timing is verified in the prompt** via **web_search** (same pattern as the May-12 batch and the checked-in **MNDY** / **CRCL** examples).

| Symbol | Company | Config |
|--------|---------|--------|
| ASTS | AST SpaceMobile, Inc. (Nasdaq: ASTS) | `configs/asts_2026_05_10.yaml` |
| FIGR | Figure Technology Solutions, Inc. (Nasdaq: FIGR) | `configs/figr_2026_05_10.yaml` |
| HIMS | Hims & Hers Health, Inc. (NYSE: HIMS) | `configs/hims_2026_05_10.yaml` |
| RGTI | Rigetti Computing, Inc. (Nasdaq: RGTI) | `configs/rgti_2026_05_10.yaml` |
| GTM | ZoomInfo Technologies Inc. (Nasdaq: GTM, formerly ZI) | `configs/gtm_2026_05_10.yaml` |
| PLUG | Plug Power Inc. (Nasdaq: PLUG) | `configs/plug_2026_05_10.yaml` |
| STE | STERIS plc (NYSE: STE) | `configs/ste_2026_05_10.yaml` |
| ACHR | Archer Aviation Inc. (NYSE: ACHR) | `configs/achr_2026_05_10.yaml` |
| IX | ORIX Corporation (NYSE ADR: IX) | `configs/ix_2026_05_10.yaml` |
| QUBT | Quantum Computing Inc. (Nasdaq: QUBT) | `configs/qubt_2026_05_10.yaml` |

The **2026-05-12 earnings batch** adds ten configs keyed to **Tue May 12, 2026** (same provider stack as above: Anthropic Opus, **OpenAI `gpt-5.5`**, Grok, Gemini Flash fan-out; Gemini Pro synthesizer with `web_search: true`; Gemini verifier; **600 s** timeouts on every step; `historical_quarters: 6`; `drive_upload_enabled: true`; `pdf_output_enabled: true`; price fields `null` with the comment *reference-only; LLMs fetch via web_search*). These configs omit **`earnings_timing`** so each run’s prompt directs models to **verify** the release/call timing for that symbol and date via **web_search** rather than baking in a guessed BMO/AMC string.

| Symbol | Company | Config |
|--------|---------|--------|
| SE | Sea Limited (NYSE ADR: SE) | `configs/se_2026_05_12.yaml` |
| ZBRA | Zebra Technologies Corporation (Nasdaq: ZBRA) | `configs/zbra_2026_05_12.yaml` |
| ONON | On Holding AG (NYSE: ONON) | `configs/onon_2026_05_12.yaml` |
| QBTS | D-Wave Quantum Inc. (NYSE: QBTS) | `configs/qbts_2026_05_12.yaml` |
| LIF | Life360, Inc. (Nasdaq: LIF) | `configs/lif_2026_05_12.yaml` |
| ETOR | eToro Group Ltd. (Nasdaq: ETOR) | `configs/etor_2026_05_12.yaml` |
| JD | JD.com, Inc. (Nasdaq ADR: JD) | `configs/jd_2026_05_12.yaml` |
| VOD | Vodafone Group Public Limited Company (Nasdaq ADR: VOD) | `configs/vod_2026_05_12.yaml` |
| TME | Tencent Music Entertainment Group (NYSE ADR: TME) | `configs/tme_2026_05_12.yaml` |
| RDY | Dr. Reddy's Laboratories Limited (NYSE ADR: RDY) | `configs/rdy_2026_05_12.yaml` |

The **2026-05-13 earnings batch** adds six configs keyed to **Wed May 13, 2026** (same provider stack and options as the May-12 batch above; `earnings_timing` omitted).

| Symbol | Company | Config |
|--------|---------|--------|
| NBIS | Nebius Group N.V. (Nasdaq: NBIS) | `configs/nbis_2026_05_13.yaml` |
| BABA | Alibaba Group Holding Limited (NYSE ADR: BABA) | `configs/baba_2026_05_13.yaml` |
| WIX | Wix.com Ltd. (Nasdaq: WIX) | `configs/wix_2026_05_13.yaml` |
| DT | Dynatrace Inc. (NYSE: DT) | `configs/dt_2026_05_13.yaml` |
| VSH | Vishay Intertechnology, Inc. (NYSE: VSH) | `configs/vsh_2026_05_13.yaml` |
| BIRK | Birkenstock Holding plc (NYSE: BIRK) | `configs/birk_2026_05_13.yaml` |

`scripts/run_all_symbols.sh` wraps `python -m equity_analyst run` and is **Bash 3.2-compatible** (no `mapfile`, no `${var,,}`, no associative arrays) so it works with macOS `/bin/bash`. By default it runs the **2026-05-10** symbol set above. Use **`--date YYYY-MM-DD`** (or `YYYY_MM_DD`) so config paths resolve as `configs/<symbol_lower>_<suffix>.yaml`. If you set **`--date`** (or pass a leading **`DATE`** positional) and omit **`--symbols`** / **`--symbols-file`**, the script **auto-discovers** every matching `configs/*_<suffix>.yaml` and runs those tickers in **sorted** order. Use **`--symbols A,B,C`** or **`--symbols-file path`** to pin a subset; with either, every expected config must exist **before** the batch starts (missing files are listed and the script exits non-zero). **`--symbols` wins** if both `--symbols` and `--symbols-file` are passed.

```bash
# Sequential (default): one symbol at a time, --iterative --max-iterations 3 --log-level INFO.
# Live Python logs stream to your terminal and are also copied to outputs/batch_<ts>/<symbol>.log.
scripts/run_all_symbols.sh

# Auto-discover every config for that earnings date (sorted tickers):
scripts/run_all_symbols.sh 2026_05_12
scripts/run_all_symbols.sh 2026_05_13
scripts/run_all_symbols.sh --date 2026-05-12

# Positional date + explicit subset (NBIS and BABA only):
scripts/run_all_symbols.sh 2026_05_13 NBIS BABA

# May 12, 2026 batch (all configs for that date, explicit comma list — same set as discovery):
scripts/run_all_symbols.sh --date 2026-05-12 --symbols SE,ZBRA,ONON,QBTS,LIF,ETOR,JD,VOD,TME,RDY

# May 13, 2026 batch (six symbols):
scripts/run_all_symbols.sh --date 2026-05-13 --symbols NBIS,BABA,WIX,DT,VSH,BIRK

# Same batch in parallel (example: three concurrent symbols):
scripts/run_all_symbols.sh --date 2026-05-12 --symbols SE,ZBRA,ONON,QBTS,LIF,ETOR,JD,VOD,TME,RDY --parallel --jobs 3

# Common overrides:
scripts/run_all_symbols.sh --max-iterations 2
scripts/run_all_symbols.sh --no-iterative
scripts/run_all_symbols.sh --log-level DEBUG

# Optional parallel mode: up to 2 symbols run concurrently by default (override with
# `--jobs N` or `-j N`, allowed range 1–10). Example: `--parallel --jobs 3`.
# WARNING: every symbol shares one API key per provider. Higher concurrency increases
# the chance of HTTP 429 rate-limit errors from Anthropic/OpenAI/Gemini/Grok; sequential
# mode is safer for production runs.
scripts/run_all_symbols.sh --parallel
```

Each run writes one combined log per symbol plus a `batch_summary.txt` under `outputs/batch_<UTC-timestamp>/`. In **sequential** mode (default), output is **tee’d**: you see each symbol’s run live in the terminal while the same lines are saved to that symbol’s log file. In **`--parallel`** mode, per-symbol output goes to the log files only (no interleaved live streams); follow one job with `tail -f outputs/batch_*/<SYMBOL>.log` (symbol names are lowercased in filenames, e.g. `asts.log`). Passing **`--jobs N`** (or **`-j N`**) without **`--parallel`** enables parallel mode when **N > 1**; **`--jobs 1`** alone keeps sequential mode. Successful symbols append `[OK]   <SYM>  duration=<sec>s  output_dir=<abs path>` lines; failures append `[FAIL] <SYM>  duration=...  exit=...  log=...`. The script exits non-zero if any symbol failed.

**Wall-clock expectations.** Sequential iterative runs typically take several minutes per symbol — for ten symbols with `--max-iterations 3` and grounded web search on every provider, plan on **multiple hours** end-to-end (often **4–10+ hours** depending on provider latency, web-search retries, and Drive upload). Use `--max-iterations 2` or `--no-iterative` for faster passes, and let the batch run unattended overnight. `--parallel` can reduce overall wall time when your provider rate limits allow it; raising `--jobs` speeds the batch only if APIs tolerate the extra concurrency—otherwise expect more 429s and retries.

### Diagnostics

- **`scripts/test_openai_cache.sh`** — quick way to verify OpenAI prompt caching is working. Runs `configs/cache_test_openai.yaml` (OpenAI-only fan-out + synthesizer, no web search, no Drive, no PDF) twice at `--log-level DEBUG`, then prints **`[PASS]`** when the two runs share the same `instructions_sha16` and run 2 reports `cache_read > 0`, or **`[FAIL]`** with the offending values.

## Standard mode

```bash
python -m equity_analyst run --config configs/mndy_2026_05_08.yaml
```

Dry-run (no API calls; writes preview under `outputs/`):

```bash
python -m equity_analyst run --config configs/mndy_2026_05_08.yaml --dry-run
```

### Web search and performance

Provider web search tools are **on by default** and are often the dominant source of latency. For faster runs, disable them:

```bash
python -m equity_analyst run --config configs/mndy_2026_05_08.yaml --no-web-search
```

The same flag works in iterative mode. Legacy aliases **`--enable-web-search` / `--no-enable-web-search`** still map to the same setting as `--web-search` / `--no-web-search`.

### Prompt caching (Anthropic fan-out)

Anthropic **Messages API** [prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching) is **on by default** for equity fan-out: the static persona lives in the `system` turn with an explicit cache breakpoint, and the **last** tool definition (web search, when enabled) is marked so tools + system reuse the same cached prefix on repeated runs with the same template and tool shape. Breakpoints use **`{"type": "ephemeral", "ttl": "1h"}`** so iterative re-runs and back-to-back jobs against one template keep a warm cache longer than the default 5-minute ephemeral TTL.

**Why:** Cached prefix reads are billed at a fraction of full input tokens and skip re-processing that prefix server-side, which usually improves **time-to-first-token** and cuts **input cost** on the stable portion of the prompt (often on the order of **~80%** savings on those cached tokens vs uncached input, depending on model and pricing tier).

**Disable:** `python -m equity_analyst run --config ... --no-prompt-cache` or set `prompt_cache_enabled: false` in YAML on `RunConfig`.

**Minimum cache size:** Anthropic only applies caching when the marked prefix meets a **model-specific minimum** (for example **4096 tokens** for Claude Opus 4.7 / 4.6 / 4.5 per current docs). Shorter prefixes are accepted but **not** cached; watch **INFO** logs from `equity_analyst.providers.anthropic_provider` for `Anthropic cache stats cache_read=...` — non-zero `cache_read` confirms a hit.

### Gemini context caching (fan-out)

When **`prompt_cache_enabled`** is true and **Gemini** is in the fan-out provider list, the CLI uses the Gemini API [**explicit context cache**](https://ai.google.dev/gemini-api/docs/caching): the static equity **persona** is stored as `system_instruction` in a server-side cache, and each request sends only the dynamic body (template output plus, in iterative mode, follow-up questions) while referencing that cache by name. Cached input tokens are billed at a **large discount** versus normal input tokens (often on the order of **~75%** lower on those cached tokens for many Gemini 2.x models—see the current [pricing](https://ai.google.dev/pricing) page), which matters most in **iterative** mode where the same persona prefix is replayed every round.

- **Index file:** `outputs/.gemini_cache_index.json` maps `(sha256(persona), model id)` → `cachedContents/...` names so repeat runs can reuse a cache still within TTL.
- **TTL:** set **`gemini_cache_ttl_s`** on `RunConfig` (default **3600**, allowed range **60–86400** seconds). Storage is billed for how long the cache lives; short TTLs avoid paying for unused cache time.
- **Disable:** same as Anthropic—`--no-prompt-cache` or `prompt_cache_enabled: false` turns off Gemini explicit caching as well.
- **Logs:** `equity_analyst.providers.gemini_provider` emits **`Gemini cache hit`** / **`Gemini cache miss creating new entry`** at **INFO** when caching applies.
- **Minimum size:** Gemini enforces a **model-dependent** minimum token count for context caching (see the [Context caching](https://ai.google.dev/gemini-api/docs/caching) doc table—**Gemini 3 Flash Preview** and **Gemini 2.5 Flash** require **1024** tokens; **Gemini 3.1 Pro Preview** and **Gemini 2.5 Pro** require **4096**). Smaller personas skip caching and use the normal uncached request path.
- **Choosing the fan-out Gemini model:** the MNDY configs use **`gemini-3-flash-preview`** for fan-out (lower caching threshold, cheaper, fast reasoner) and reserve **`gemini-3.1-pro-preview`** for synthesis (where the much larger synthesizer system prompt easily clears the 4,096-token Pro minimum). If you put Pro in the fan-out list with a small persona, the per-request cache will silently skip and you'll pay full input price each call.

YAML may set **per-provider** overrides: optional `model` (API model id for that backend), optional **`max_output_tokens`** (completion budget for that fan-out provider only; falls back to global `max_output_tokens`), `web_search: true|false`, and optional **per-provider timeouts** (`request_timeout_s`). Global defaults: `request_timeout_s` (default **180** seconds), `max_output_tokens` (default **16000** for fan-out — each parallel provider gets this unless overridden per provider), `synthesizer_max_output_tokens` (default **24000** for the final synthesis pass — larger because the synthesizer must weave every provider’s output across all 11 sections), `synthesizer_max_input_tokens` (default **100000** estimated tokens for the full synthesis prompt after pre-compression; override in YAML or with **`--synthesizer-max-input-tokens`** — large synthesis models such as **Gemini 3.1 Pro** support much larger contexts), and `verifier_max_output_tokens` (default **8192**, iterative verifier only). **OpenAI** long web-search runs use the **streaming** Responses API so the HTTP connection stays alive across multi-minute tool loops; if you still hit `asyncio` timeout errors, raise **`request_timeout_s`** globally or on specific providers (for example `900` or `1500` seconds on `openai`).

Default fan-out budget is **16,000** tokens per provider. For an 11-section deep-research prompt this is usually enough to avoid truncation in `claude.md` / `openai.md` / `grok.md` / `gemini.md`; if a run still cuts off, raise the global `max_output_tokens` or set a higher **`max_output_tokens`** on specific providers in YAML. Override from the CLI with **`--max-output-tokens`** (fan-out), **`--synthesizer-max-output-tokens`** and **`--synthesizer-max-input-tokens`** (synthesis), and **`--verifier-max-output-tokens`** (iterative verify step).

**Oversized provider bodies (pre-synthesis compression):** Before the synthesizer builds its prompt, **healthy** fan-out bodies are candidates for Gemini Flash summarization when **either** (a) a single body’s size estimate `len(text)//4` is at least **`summarize_threshold_input_tokens`** (default **8000**), **or** (b) the **sum** of those estimates across healthy bodies exceeds **`max(8000, synthesizer_max_input_tokens - 3000)`** (a rough budget for provider bodies after reserving ~3000 tokens for system/persona/headers). The implementation summarizes the **largest** bodies first until the aggregate is under that target. Models: **`oversized_summarize_model`** (default **`gemini-3-flash-preview`**, no web search). The summarizer is instructed to keep numbered sections, markdown tables, all quantitative figures, and disagreement signals, and to shorten prose only. **`oversized_summarize_max_output_tokens`** (default **8192**) caps that completion; **`oversized_summarize_max_input_tokens`** (default **100000** estimated tokens) bounds how much text is sent to the summarizer (larger bodies are head/tail shrunk first). If summarization fails or returns empty output, the **original** body is kept and the existing **`synthesizer_max_input_tokens`** head/tail trim path still runs. Toggle with **`summarize_oversized_providers`** in YAML or **`--no-summarize-oversized`** on the CLI; override the per-body cutoff with **`--summarize-threshold-tokens`**. This adds Flash API calls per summarized provider per synthesis round (including each iterative refinement round), trading a small amount of extra latency and token cost for a shorter, safer synthesis prompt.

**Anthropic** defaults to **Opus** (`claude-opus-4-7` unless you set `model`), which has substantially higher input-token rate limits on the standard tier (on the order of **~500k** tokens per minute) than Sonnet (often around **~30k** tokens per minute). Override with `model` under `providers` if you need a different snapshot. See [Anthropic model IDs](https://docs.anthropic.com/en/docs/about-claude/models/model-ids-and-versions). Long-running calls (including web search) use the **streaming** Messages API, as required by the Anthropic Python SDK for requests that may exceed ~10 minutes.

**OpenAI** uses the **streaming** Responses API (`stream=True`) for all fan-out calls so long web-search tool loops stay connected instead of timing out on a single blocking HTTP response.

### OpenAI automatic prompt caching (fan-out)

OpenAI [**automatic prompt caching**](https://platform.openai.com/docs/guides/prompt-caching) applies when the combined prefix (messages + tools) is at least **1024 tokens** and the **leading** bytes match a recent request on the routed worker. Fan-out calls send structured **`input`**: a **`system`** message with the static equity persona first, then a **`user`** message with the rendered template body, plus a stable **`prompt_cache_key`** so similar jobs route consistently. Usage reports cache hits on **`usage.input_tokens_details.cached_tokens`** (logged as **`OpenAI cache stats cache_read=...`**).

**Important:** `cache_read` is **usually zero on the first** request after a cold start (nothing to reuse yet). Run the same symbol/config twice within the model’s retention window ( **`gpt-5.5`** defaults to **24h** extended retention per OpenAI’s docs), or use two back-to-back calls, to see non-zero **`cached_tokens`** on the second completion.

**Diagnostic:** with **`OPENAI_API_KEY`** set (and optional **`PROBE_OPENAI_MODEL`** override, default **`gpt-5.5`**), run:

```bash
.venv/bin/python scripts/probe_openai_cache.py
```

It issues two minimal completions with the same static/user split and prints **`cached_tokens`** for each call.

**Synthesizer** defaults to **Gemini** (`gemini-3.1-pro-preview` per `GeminiProvider.DEFAULT_GEMINI_MODEL`) so synthesis runs on a different model than typical Anthropic fan-out, which avoids stacking the same provider’s rate limits on both parallel answers and the long synthesis prompt. Configure it as a string (`synthesizer: gemini`) or as an object with `name`, optional `model`, optional `web_search`, and optional `request_timeout_s`.

```yaml
request_timeout_s: 180
max_output_tokens: 16000
synthesizer_max_output_tokens: 24000
verifier_max_output_tokens: 8192
providers:
  - name: anthropic
    model: claude-opus-4-7
    max_output_tokens: 24000
    web_search: true
  - name: openai
    model: gpt-5.5
    web_search: false
  - name: grok
    max_output_tokens: 12000
  - name: gemini
    model: gemini-3-flash-preview
    request_timeout_s: 120
synthesizer:
  name: gemini
  model: gemini-3.1-pro-preview
  web_search: true
  request_timeout_s: 240
verifier_provider: gemini
verifier_model: gemini-3.1-pro-preview
```

The simple list form remains valid: `providers: ["anthropic", "openai"]`. A bare `synthesizer: openai` string is still accepted.

### Run timing (`run.json`)

After a **live** standard run, `run.json` includes a **`timing`** object: `parallel_provider_batch_wall_s`, `synthesis_wall_s`, `total_wall_s`, and `per_provider` latency snapshots. Iterative runs add a **`timing`** summary when `finalize` runs (per-iteration provider, synthesis, and verify wall times plus `total_sequential_wall_s`). INFO logs also emit a heartbeat every **30s** while waiting on a provider batch.

## Iterative mode

Iterative runs use a LangGraph `StateGraph` with nodes `fan_out` → `synthesize` → `verify` → `route` → (`fan_out` again or `finalize`). Each round:

1. **fan_out** – all configured providers answer the prompt plus any accumulated follow-up questions.
2. **synthesize** – the synthesizer must emit section confidences and a line `OVERALL_CONFIDENCE: <0.0-1.0>` for routing.
3. **verify** – default **Gemini** verifier returns JSON `verified` / `contradicted` / `unverifiable` using web search when enabled (same registry keys as fan-out: `verifier_provider` defaults to **`gemini`**; override `verifier_model` in YAML or `--verifier-model` on the CLI). The verify step does **not** use Gemini explicit context caching (`cacheable_prefix=None`) so it stays separate from equity fan-out persona caching. **Anthropic** remains supported: set `verifier_provider: anthropic` (and optional `verifier_model`) or pass `--verifier-provider anthropic`. Anthropic verification still uses `prompt_cache_enabled=False` and `force_tool_use=False` so tool-choice forcing from fan-out does not apply to the short verifier prompt.
4. **route** – finalize if `len(rounds) >= max_iterations`, or if overall confidence meets `--confidence-threshold` and there are no contradictions; otherwise append follow-ups and loop.
5. **finalize** – writes `synthesis.md`, per-round files under `iterations/`, and `checkpoint.sqlite` (removed after a successful finalize unless you pass **`--keep-checkpoint`** or set `DELETE_CHECKPOINT_AFTER_SUCCESS=false`).

```bash
python -m equity_analyst run --config configs/mndy_2026_05_08.yaml --iterative \
  --max-iterations 3 --confidence-threshold 0.85
```

Verifier overrides (optional):

```bash
python -m equity_analyst run --config configs/mndy_2026_05_08.yaml --iterative \
  --verifier-provider anthropic --verifier-model claude-opus-4-7
```

Iterative dry-run (compiles the graph and prints an excerpt of the rendered prompt):

```bash
python -m equity_analyst run --config configs/mndy_2026_05_08.yaml --iterative --dry-run
```

## Resume after a crash

Iterative runs store checkpoints at `outputs/<run_id>/checkpoint.sqlite` and metadata in `run.json`. On **success**, the default is to delete `checkpoint.sqlite` plus any `-wal` / `-shm` / `-journal` siblings after `finalize` (use **`--keep-checkpoint`** to retain them for debugging). Failed or aborted runs leave the checkpoint in place so **`--resume`** can continue. The CLI uses LangGraph's async SQLite checkpointer (`AsyncSqliteSaver` / `aiosqlite`): the DB connection is opened at the start of the run and closed when the run finishes (`async with`), which matches `app.ainvoke` and avoids leaking handles. Resume with the output folder name (e.g. `MNDY_20260509T120000Z`):

```bash
python -m equity_analyst run --iterative --resume MNDY_20260509T120000Z
```

If `run.json` is present, `--config` can be omitted; you may still pass `--config` to override.

## Graph (iterative)

```mermaid
flowchart TD
  START([start]) --> fan_out[fan_out]
  fan_out --> synthesize[synthesize]
  synthesize --> verify[verify]
  verify --> route[route]
  route -->|continue| fan_out
  route -->|done| finalize[finalize]
  finalize --> END([end])
```

### Cost optimization (iterative)

Several defaults-on behaviors reduce token and tool use on **iteration 2+** (see `RunConfig` / `run.json` `config`):

1. **Frozen facts packet (Strategy A)** — After round-1 synthesis, a compact **`facts_packet.md`** is written under the run directory (Gemini Flash extractor: `facts_packet_extractor_*`, `facts_packet_max_output_tokens`). Later fan-out rounds prepend a **FACTS (frozen from iteration 1 — do NOT re-fetch via web_search)** block so models lean on that snapshot instead of repeating the same market pulls. The verifier JSON may set **`"refresh_facts": true`** to re-extract from the **latest** synthesis before the next fan-out when figures are genuinely stale.

2. **Conditional fan-out (Strategy B)** — From iteration 2 onward, **only the synthesizer** re-runs on verifier feedback (plus the optional refinement block) unless the verifier requests more provider work via **`"refan_out_providers": ["anthropic", ...]`** or **`"refan_out_all": true`**. Empty list / false means “synthesizer only” for that transition.

3. **Refinement-mode provider prompt** — When iteration 2+ **does** call fan-out providers again (full re-fan, partial `refan_out_providers`, or because `conditional_fanout_enabled` is false), the user message includes a **REFINEMENT MODE** prefix: prior-round synthesis, follow-up targets, optional verifier **`sections_to_revise`**, and explicit instructions to **quote FACTS verbatim** instead of re-deriving 1σ/2σ/3σ tables, IV, PCR, and other frozen primitives. Toggle with `refinement_mode_prompt_enabled` (default `true`) or env `REFINEMENT_MODE_PROMPT_ENABLED`.

**YAML / `RunConfig`:** `facts_packet_enabled` (default `true`), `conditional_fanout_enabled` (default `true`), `refinement_mode_prompt_enabled` (default `true`), plus extractor fields above.

**Env:** `FACTS_PACKET_ENABLED`, `CONDITIONAL_FANOUT_ENABLED`, `REFINEMENT_MODE_PROMPT_ENABLED` (`1` / `true` / `yes` / `on` vs anything else treated as off when set), and `FACTS_PACKET_MAX_OUTPUT_TOKENS` (invalid values log a warning and keep the default). When a key is omitted from YAML, env applies; explicit YAML wins over env. Override via env: `FACTS_PACKET_MAX_OUTPUT_TOKENS=8192` in `.env` (default 4096, range 256–128000).

**CLI:** `--facts-packet` / `--no-facts-packet`, `--conditional-fanout` / `--no-conditional-fanout` (Boolean optional actions; omit to keep YAML/env defaults).

**Logs:** Each fan-out logs `Iteration N: fan_out=skipped|partial|full, facts_packet=frozen|refreshed|pending|off` and, when savings apply, `Iteration N saved approx X tokens vs full re-run` (rough `len(prompt)//4` estimate for skipped or partially skipped fan-out bodies).

**Legacy behavior:** pass **`--no-facts-packet --no-conditional-fanout`** (or set both false in YAML) to match the older “full fan-out every iteration” shape.

## Logging

Progress logs use the stdlib `logging` package on the `equity_analyst` logger. By default the CLI prints **INFO** lines to **stderr** with timestamp, level, logger name, and message.

- Set verbosity with `--log-level DEBUG|INFO|WARNING|ERROR` on `run` (default `INFO`).
- **DEBUG** adds request-shape hints from providers (model name, character counts, tool flags) without logging API keys, full bodies, or `.env` contents.

When the CLI writes a run under `outputs/<symbol>_<timestamp>/` (standard mode, iterative mode, or standard **dry-run**), it also appends the same log lines to **`outputs/<...>/agent.log`**.

**Iterative `--dry-run`** does not create an output directory, so no `agent.log` is produced for that path; use stderr only or run without `--dry-run` to capture a file.

### Caching

OpenAI and Grok cache hits are logged automatically when present (`cache_read=<N>`). Caching is automatic for both providers — no setup required.

Run with `--log-level DEBUG` to see the first 200 characters and a truncated SHA256 hash of each OpenAI/Grok request body. If consecutive runs have the same hash and `cache_read` is still 0, the prefix is stable but caching is not engaging — usually because the static prefix is below the 1024-token minimum.

## Customizing prompts

**Reference prices in YAML are hints, not facts.** The equity template requires providers to pull **live or recently sourced** quotes—especially the **last regular-session close** and last session range—and to **cite** source and timestamp. Optional config numbers are only for rough rescaling; models are instructed to cross-check them against fetched data. **`earnings_timing`** in YAML is optional: when absent, the template adds a **mandatory verification** block so models use **web_search** to establish **BMO / AMC / during-hours** timing for `earnings_date` and anchor timing-sensitive sections to that evidence; when present, it is printed as the brief’s stated schedule (you can still cross-check in prose).

These plain-text and template files control model instructions without editing Python:

- `prompts/equity_analyst_system.md` — persona / instructions (cached as the Anthropic system prompt and prepended for other providers).
- `prompts/equity_analyst.j2` — the 11 numbered sections, a Jinja template with `{{ symbol }}`, optional `reference_*` / legacy price context, dates, and the other template variables.
- `prompts/synthesizer_system.md` — how the synthesizer compares provider answers and formats the consensus.
- `prompts/prediction_extract_system.md` — JSON-only instructions for the optional prediction-extractor LLM (five fixed horizons).
- `prompts/facts_extract_system.md` — markdown-only instructions for the iterative **facts packet** extractor (round-1 synthesis → `facts_packet.md`).

Edits take effect on the next CLI run; you do not need to change code or restart a long-lived process.

## Development checks

```bash
ruff check .
mypy --strict equity_analyst
pytest -q
```

## Recording outcomes (for calibration / future training)

After earnings occur, you can label a prior run with realized market outcomes. This produces structured data for later calibration / RL / DPO workflows without changing the original model outputs.

Example:

```bash
python -m equity_analyst outcome-record \
  --run-dir /abs/path/to/outputs/SYM_20260511T123456Z \
  --earnings-day-close 12.34 \
  --next-trading-day-close 11.95 \
  --direction-vs-prior-close down \
  --notes "Beat on EPS, guide down; stock faded into close"
```

Files written:

- `outputs/<run-id>/outcome.json` (stored next to `run.json`)
- `outputs/outcomes_registry.jsonl` (one JSON line per recorded outcome, append-only)

`outputs/outcomes_registry.jsonl` is ignored by git by default (see `.gitignore`). If you want a shared registry (team workflow), you may choose to commit it.

### Auto-fetching outcomes from Yahoo Finance

Use **`--auto-fetch`** to pull earnings-day OHLC, next-trading-day OHLC, and the close ~5 trading days later from Yahoo Finance via **[yfinance](https://github.com/ranaroussi/yfinance)** instead of typing numbers in by hand:

```bash
python -m equity_analyst outcome-record \
  --run-dir /abs/path/to/outputs/CRCL_20260511T023700Z \
  --auto-fetch
```

How it works:

- Reads **`symbol`** and **`earnings_date`** from the run's `run.json` (`config` snapshot).
- Parses `earnings_date` leniently with **`python-dateutil`** (so values like `"Mon May 11 2026"` work).
- Calls `yfinance.Ticker(symbol).history(...)` for a 15-calendar-day window starting at the parsed earnings date, then picks bars in trading-day order: bar 0 → earnings day OHLC, bar 1 → next trading day OHLC, bar 5 → `one_week_later_close` (falls back to the last available bar with a WARNING if fewer than 6 trading days are present).
- If a `baseline_close` is available (from `RunConfig.current_price` in `run.json`, a "last verified close" / "last regular-session close" / "closing price" phrase in the first 8 KB of `synthesis.md`, or else the prior regular-session close from Yahoo via `yfinance` for the last trading day strictly before `earnings_date`), the tool sets `direction_vs_prior_close` to **`up`** / **`down`** / **`flat`** (flat = within ±0.1%).
- Per-field results are logged at **INFO**; missing fields log a **WARNING** but never crash the command. The same outcome flow writes `outcome.json`, appends to `outputs/outcomes_registry.jsonl`, and best-effort UPSERTs the `outcomes` row.

**Override individual fields:** explicit CLI flags (e.g. `--earnings-day-close 12.34`, `--direction-vs-prior-close down`) always win over fetched values. Combine `--auto-fetch` with `--interactive` to fill in only the fields yfinance could not return.

**yfinance reliability caveat:** yfinance scrapes the public Yahoo Finance endpoint, which is **unofficial and rate-limited**, and can return empty frames for **ADRs, recently-IPO'd tickers, or during Yahoo backend incidents**. The CLI treats yfinance as best-effort: any failed call (empty frame, timeout, parsing error) logs a WARNING, sets affected fields to `None`, and proceeds with whatever was returned — you can rerun `outcome-record --auto-fetch` later or fall back to the manual flags.

### Batch recording (`outcome-record-batch`)

After a multi-symbol earnings batch, you can record outcomes for **many runs in one command** (same per-run logic as `outcome-record`, including optional `--auto-fetch` and best-effort Postgres upsert).

**Shape A — batch directory:** reads `output_dir=...` paths from `batch_summary.txt` under the batch folder (same format the `run_all_symbols.sh` runner appends on `[OK]` lines).

```bash
python -m equity_analyst outcome-record-batch \
  --batch-dir outputs/batch_20260511T025203Z \
  --auto-fetch
```

**Shape B — symbol list:** resolves `outputs/<SYMBOL>_<TS>/` directories whose run timestamp is **on or after** `--since` (default: seven calendar days ago, UTC midnight). Use `--symbols SYM1,SYM2,...` or `--symbols-file path.txt` (one ticker per line or comma-separated; `#` starts a comment line). With `--newest-only` (the default), only the newest matching run per symbol is recorded; symbols with no matching directory are reported as `[FAIL]` and increment the **Skipped** counter in the printed summary (per-run exceptions increment **Failed** when that line is shown).

```bash
python -m equity_analyst outcome-record-batch \
  --symbols SE,ZBRA,ONON,QBTS,LIF,ETOR,JD,VOD,TME,RDY \
  --since 2026-05-12 \
  --auto-fetch
```

Common flags: `--dry-run` (no `outcome.json` / registry / DB writes), `--rate-limit-sleep-s 0.5` (delay between symbols after the first, to reduce yfinance throttling), `--continue-on-error` / `--no-continue-on-error`, and `--outputs-dir` for Shape B when runs live outside the default `outputs/` folder. Exit code **1** if any `[FAIL]` line is printed (missing run directory, missing `run.json`, I/O error, and so on); `[WARN]` lines (partial or empty auto-fetch) still exit **0**.

## Prediction extraction (calibration prep)

After a run, you can populate the Postgres **`predictions`** table with five fixed **horizons** (`earnings_day_open`, `earnings_day_close`, `next_trading_day_open`, `next_trading_day_close`, `one_week_later_close`) by calling a **fast LLM** (default **Gemini Flash**, `gemini-3-flash-preview`, no web search) against the same synthesis file the rest of the pipeline treats as final: `outputs/<run-id>/synthesis.md` when present (iterative runs write the packaged report there; otherwise the newest `iterations/iteration_*_synthesis.md` is used, matching outcome tooling).

**Explicit CLI (always runs extraction for the given run dir(s)):**

```bash
python -m equity_analyst predictions-extract --run-dir outputs/CRCL_20260511T023747Z
python -m equity_analyst predictions-extract-batch --batch-dir outputs/batch_20260511T025203Z
python -m equity_analyst predictions-extract-batch --symbols SE,ZBRA,ONON --since 2026-05-12
```

Batch mode mirrors **`outcome-record-batch`**: Shape A parses `output_dir=` lines from `batch_summary.txt`; Shape B resolves per-symbol run directories under `outputs/` on or after `--since` (default seven days ago UTC), with `--newest-only` (default) or all matches. Use **`--dry-run`** to list targets without calling the extractor.

**Postgres writes:** existing `predictions` rows for the run are **deleted** then re-inserted (idempotent reruns). Each row uses `source = 'llm_extracted'`.

**Fallback file:** if Postgres is unavailable, `DATABASE_URL` is unset/invalid, **`db_enabled`** is false in the run’s config snapshot, or the insert fails, the tool logs a **WARNING** and writes **`predictions_extract.json`** next to `run.json` when at least one structured row was parsed.

**Auto-run after each completion (default off):** set **`prediction_extract_enabled: true`** in YAML (it is recorded in `run.json`), or pass **`--extract-predictions`** on `python -m equity_analyst run` (Boolean optional: `--no-extract-predictions` forces off for that invocation). When enabled, standard runs invoke extraction after synthesis; iterative runs invoke it at the end of **`finalize`** (after `synthesis.md` and `run.json` are written).

**Tuning (RunConfig):** `prediction_extract_provider` (default `gemini`), `prediction_extract_model` (default `gemini-3-flash-preview`), `prediction_extract_max_output_tokens` (default `2048`), `prediction_extract_timeout_s` (default `120`).

## Backfilling existing runs into Postgres

The `db-backfill` CLI walks `outputs/<run_id>/` directories and inserts the corresponding rows into the `runs` and `provider_responses` tables. It is **idempotent**: `runs` is upserted on its primary key, and `provider_responses` rows for the run are deleted and re-inserted on each invocation, so rerunning the command converges to the same final state.

Preview what would be written (no DB writes; safe to run anywhere):

```bash
python -m equity_analyst db-backfill --dry-run
```

Backfill everything currently under `outputs/`:

```bash
python -m equity_analyst db-backfill
```

Useful filters:

- `--outputs-dir PATH` — root directory to scan (default `outputs/`).
- `--symbol SYM` — only run dirs whose name starts with `<SYM>_` (case-insensitive).
- `--since YYYY-MM-DD` — only run dirs whose timestamp segment is ≥ this date.
- `--limit N` — backfill at most N runs (oldest first; combine with `--newest-first` to flip).

Output ends with a small summary table:

```
Backfill summary
  Scanned:   55
  Inserted:  53
  Skipped:   2  (already up to date)
  Errors:    0
```

`batch_<ts>/` directories (no `run.json`) are skipped silently. Legacy `run.json` files missing newer fields (`run_environment`, `iterations_completed`, `started_at_utc`, ...) are backfilled with sensible defaults. **Unlike the additive best-effort DB writes in the main run path, this command requires a reachable `DATABASE_URL` and exits with a non-zero status if the DB is unavailable.**
