# Multi-LLM equity analyst

Python CLI that renders a Jinja2 equity/options prompt from YAML config, fans out to multiple LLM providers in parallel, and synthesizes a consensus report. Optional **iterative** mode runs a LangGraph loop: multi-provider fan-out, synthesis with per-round confidence parsing, web-grounded verification, routing, and final packaging with SQLite checkpointing for resume.

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

## Google Drive auto-upload

After each standard or iterative run finishes writing `outputs/<run-id>/` (including `run.json`), you can optionally mirror that folder to Google Drive using a **Google Cloud service account** JSON key. The CLI creates a subfolder named after the run id under a folder you choose, uploads every file (preserving paths such as `iterations/`), skips dotfiles, and appends `drive_folder_url` to `run.json` on success. Iterative runs also append the link to the footer of `synthesis.md`. Upload failures are logged and never fail the analysis run.

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

Per-run CLI overrides:

```bash
python -m equity_analyst run --config ... --upload-to-drive --drive-folder-id <folder-id>
python -m equity_analyst run --config ... --no-upload-to-drive
python -m equity_analyst run --config ... --drive-auth-mode oauth_user
```

### Caveats

- With a service account, file bytes are stored against **Shared Drive** (or delegated-user) quota—not “free” personal My Drive space for the SA.
- Files land in the folder you configure; anyone who can **view** that Drive folder can see uploaded runs. Treat the folder and sharing like sensitive storage.

## Configs

Stock-specific YAML lives under `configs/`. **Workflow:** pick the **MNDY** or **CRCL** pair below as a **template**, copy both YAMLs to new filenames (`<symbol>_YYYY_MM_DD.yaml` and `_fast.yaml`), then edit symbol, company name, session labels, dates, optional price hints, and any symbol-specific lookbacks. **Price fields** (`today_low` / `today_high` / `current_price`, or the aliases `reference_session_*` / `reference_last_price`) are **optional, unverified hints** for orientation only—the rendered prompt tells models to **fetch and cite** the **last regular-session official closing price** (and session high/low) via **web_search** and not to treat YAML numbers as ground truth.

| File | Use case |
|------|----------|
| `configs/mndy_2026_05_08.yaml` | Default **MNDY** run: all four fan-out providers (**Anthropic**, **OpenAI**, **Grok**, **Gemini Flash**) use long timeouts; web search follows each provider’s default (typically on). **Synthesis** runs on **Gemini 3.1 Pro Preview**. Best for highest-quality grounded research. |
| `configs/mndy_2026_05_08_fast.yaml` | **MNDY** hybrid speed: **OpenAI** alone runs deep `web_search`; **Anthropic**, **Grok**, and **Gemini Flash** reason without search; **Gemini 3.1 Pro Preview** synthesizer has no extra search. Shorter wall-clock for iteration. |
| `configs/crcl_2026_05_08.yaml` | Same layout as the MNDY standard config for **CRCL** (Circle Internet Group, NYSE: **CRCL**), aligned to the **May 11, 2026** earnings cycle. |
| `configs/crcl_2026_05_08_fast.yaml` | **CRCL** hybrid fast config (mirrors MNDY `_fast`: one grounded **OpenAI** search, fast fan-out otherwise). |

### Running multiple symbols

The 2026-05-10 batch ships ten standard configs that all mirror **CRCL**’s provider structure (Anthropic Opus → OpenAI → Grok → Gemini Flash fan-out, Gemini Pro synthesizer, Gemini verifier, 600 s timeouts, `historical_quarters: 6`). Price fields are `null` on purpose — every provider must source the last regular-session close via `web_search`.

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

`scripts/run_all_symbols.sh` wraps `python -m equity_analyst run` for the ten symbols above and is **Bash 3.2-compatible** (no `mapfile`, no `${var,,}`, no associative arrays) so it works with macOS `/bin/bash`:

```bash
# Sequential (default): one symbol at a time, --iterative --max-iterations 3 --log-level INFO.
scripts/run_all_symbols.sh

# Common overrides:
scripts/run_all_symbols.sh --max-iterations 2
scripts/run_all_symbols.sh --no-iterative
scripts/run_all_symbols.sh --log-level DEBUG

# Optional parallel mode (all 10 background jobs + `wait`).
# WARNING: every symbol shares one API key per provider. Anthropic/OpenAI/Gemini/Grok
# will rate-limit aggressive fan-out, so the wall-clock win over sequential is small
# unless you have elevated tier limits. Sequential is safer for production runs.
scripts/run_all_symbols.sh --parallel
```

Each run writes one combined log per symbol plus a `batch_summary.txt` under `outputs/batch_<UTC-timestamp>/`. Successful symbols append `[OK]   <SYM>  duration=<sec>s  output_dir=<abs path>` lines; failures append `[FAIL] <SYM>  duration=...  exit=...  log=...`. The script exits non-zero if any symbol failed.

**Wall-clock expectations.** Sequential iterative runs typically take several minutes per symbol — for ten symbols with `--max-iterations 3` and grounded web search on every provider, plan on **multiple hours** end-to-end (often **4–10+ hours** depending on provider latency, web-search retries, and Drive upload). Use `--max-iterations 2` or `--no-iterative` for faster passes, and let the batch run unattended overnight. `--parallel` reduces overall wall time only if your provider rate limits permit four concurrent grounded searches across 10 jobs; otherwise per-symbol time grows while the total still pays for ten runs.

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
5. **finalize** – writes `synthesis.md`, per-round files under `iterations/`, and `checkpoint.sqlite`.

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

Iterative runs store checkpoints at `outputs/<run_id>/checkpoint.sqlite` and metadata in `run.json`. The CLI uses LangGraph's async SQLite checkpointer (`AsyncSqliteSaver` / `aiosqlite`): the DB connection is opened at the start of the run and closed when the run finishes (`async with`), which matches `app.ainvoke` and avoids leaking handles. Resume with the output folder name (e.g. `MNDY_20260509T120000Z`):

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

**Reference prices in YAML are hints, not facts.** The equity template requires providers to pull **live or recently sourced** quotes—especially the **last regular-session close** and last session range—and to **cite** source and timestamp. Optional config numbers are only for rough rescaling; models are instructed to cross-check them against fetched data.

These plain-text and template files control model instructions without editing Python:

- `prompts/equity_analyst_system.md` — persona / instructions (cached as the Anthropic system prompt and prepended for other providers).
- `prompts/equity_analyst.j2` — the 11 numbered sections, a Jinja template with `{{ symbol }}`, optional `reference_*` / legacy price context, dates, and the other template variables.
- `prompts/synthesizer_system.md` — how the synthesizer compares provider answers and formats the consensus.

Edits take effect on the next CLI run; you do not need to change code or restart a long-lived process.

## Development checks

```bash
ruff check .
mypy --strict equity_analyst
pytest -q
```
