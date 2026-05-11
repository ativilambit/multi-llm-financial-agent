# Changelog

All notable changes to this project will be documented here.

The format is loosely based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### 2026-05-11

- **Iterative** Facts packet now includes 2σ and 3σ implied moves alongside 1σ.
- **Prompt template** Require all three standard deviation ranges (1σ / 2σ / 3σ) explicitly in section 1, 9, 11; synthesizer instructed to preserve all SD levels rather than collapsing to 1σ.
- **Iterative / Cost** Added optional frozen **facts packet** (`facts_packet.md`, extractor LLM) after round-1 synthesis and **conditional fan-out** (synthesizer-only on iteration 2+ unless the verifier requests `refan_out_providers` / `refan_out_all`); verifier JSON gains `refresh_facts`, `refan_out_providers`, `refan_out_all`. Defaults on; CLI `--no-facts-packet` / `--no-conditional-fanout` restores prior behavior. (`feat(iterative): facts packet + conditional fan-out for cost reduction`)
- **Outcomes** When `outcome-record --auto-fetch` (and batch) has no usable baseline in `run.json` or `synthesis.md` (`current_price` null, etc.), `direction_vs_prior_close` now falls back to the prior regular-session close from Yahoo Finance (`yfinance`) for the last trading day strictly before `earnings_date`. (`feat(outcome): yfinance prior-session close as baseline when config price null`)
- **Predictions** Added LLM-based extraction of five calibration horizons from final `synthesis.md` into Postgres (`predictions`, `source=llm_extracted`) with idempotent DELETE+INSERT; CLI `predictions-extract` / `predictions-extract-batch`; optional auto-run after standard or iterative completion via `prediction_extract_enabled` (default false) or `run --extract-predictions`; fallback artifact `predictions_extract.json` when DB writes are skipped or fail. (`feat(predictions): LLM extraction of synthesis horizons into Postgres`)
- **Iterative / Drive** Google Drive directory upload now skips `checkpoint.sqlite`, `checkpoint.sqlite-wal`, `checkpoint.sqlite-shm`, and `checkpoint.sqlite-journal` (exact basename match; INFO log per skipped path).
- **Iterative** After a successful `finalize`, checkpoint artifacts under the run directory are removed by default (`delete_checkpoint_after_success`, overridable with `DELETE_CHECKPOINT_AFTER_SUCCESS=false` or **`--keep-checkpoint`**); failed or aborted runs keep the DB for **`--resume`**.
- **Outcomes** Added `outcome-record-batch` to record outcomes for many runs at once: **Shape A** parses `output_dir=` lines from `outputs/batch_<ts>/batch_summary.txt`; **Shape B** resolves the newest (or all) `outputs/<SYM>_<TS>/` runs per ticker on or after `--since` (default seven days ago). Shares `record_outcome_for_run_dir` with `outcome-record`; supports `--auto-fetch`, `--dry-run`, `--rate-limit-sleep-s`, and `--continue-on-error`. (`feat(outcome): batch outcome-record for a whole batch or symbol list`)
- **DB** Added `python -m equity_analyst db-backfill` CLI to import existing `outputs/<run-id>/run.json` artifacts into `runs` + `provider_responses` (idempotent UPSERT for runs, DELETE+INSERT for provider_responses; supports `--outputs-dir`, `--limit`, `--newest-first`, `--dry-run`, `--symbol`, `--since`). (`feat(db,outcome): backfill existing runs + auto-fetch outcomes via yfinance`)
- **Outcomes** Added `--auto-fetch` flag to `outcome-record`: pulls earnings-day OHLC, next-trading-day OHLC, and the close ~5 trading days later from Yahoo Finance via `yfinance` (with `python-dateutil` fuzzy parsing of `earnings_date`); explicit `--earnings-day-*` / `--direction-vs-prior-close` flags still win when set. `direction_vs_prior_close` is computed against `RunConfig.current_price` from `run.json` or a regex-extracted "last close" figure from `synthesis.md`. (`feat(db,outcome): backfill existing runs + auto-fetch outcomes via yfinance`)
- **Env** Auto-load `.env` in Alembic migrations and `setup_db.sh` so `DATABASE_URL` works without manual export.
- **DB** Added additive Postgres metadata layer for run/outcome tracking (SQLAlchemy 2.0 async + psycopg 3, Alembic migrations). (`feat(db): add Postgres run/outcome tracking via SQLAlchemy + Alembic`)
- **Outcomes** Outcomes now also best-effort UPSERT to Postgres `outcomes` in addition to `outcome.json` + `outputs/outcomes_registry.jsonl`. (`feat(db): add Postgres run/outcome tracking via SQLAlchemy + Alembic`)
- **Outcomes** Added `outcome-record` CLI to record realized earnings outcomes per run and append to an outcomes registry JSONL for future calibration/training. (`feat(outcomes): record realized earnings outcomes per run`)
- **Prompt template** Added new section 8 "Bottom-up qualitative overlay" (mandatory before predictions). Renumbered subsequent sections 8→9, 9→10, 10→11, 11→12. Updated synthesizer and verifier cross-references. (`feat(prompt): add bottom-up qualitative overlay as new section 8`)
- **Drive upload** Added `run_environment` (`production` | `test`, default `production`) with CLI `--environment` / `--env` and `RUN_ENVIRONMENT` env override. Uploads resolve or create lowercase **`prod`** or **`test`** child folders under `drive_root_folder_id` before creating the per-run folder; `run.json` records `run_environment`, `drive_upload_parent_folder_id`, and `drive_upload_parent_folder_name`. (`feat(drive): route uploads to prod/test subfolders by run environment`)
- **Prompt template** Generalized price-action wording to relational phrasing (“day of the earnings call”, “next trading day”, “end of that earnings week”). Added Date anchors line at the top of the template. (`3943099`)
- **Prompt template** Made `earnings_timing` optional; LLM can confirm timing via web search when omitted. Stripped from legacy configs. (`39b2e4a`)
- **Iterative** Fixed mid-sentence truncation in iteration changelog preview (paragraph-boundary cut instead of a fixed character slice). Added WARNING when synthesizer hits real MAX_TOKENS across providers. (`baac3de`)
- **Retry / Backoff** Recognize Gemini `google.genai.errors.APIError` codes 429/5xx for backoff; wrapped Gemini Flash summarizer in `async_retry_call`; honor `retryDelay` from GenAI responses. (`8649ed5`)

### 2026-05-10

- **Configs** Added May 12 earnings batch (SE, ZBRA, ONON, QBTS, LIF, ETOR, JD, VOD, TME, RDY); refactored `run_all_symbols.sh` for `--date`, `--symbols`, and `--symbols-file`. (`e298c39`)
- **Scripts** Fixed bash 3.2 empty-array expansion crash in the parallel runner. (`4e01d0e`)
- **Scripts** Added `--parallel --jobs N` concurrency cap (default 2). (`dc78e7d`)
- **Scripts** Sequential batch runs stream per-symbol output to the terminal (tee). (`f50e90a`)
- **Configs** Added batch configs for ASTS, FIGR, HIMS, RGTI, GTM, PLUG, STE, ACHR, IX, QUBT plus batch runner wiring. (`bcfc346`)
- **Configs** Added CRCL configs for May 11, 2026 earnings (mirrors MNDY pattern). (`59be7e5`)
- **Configs** Added OpenAI-only cache validation config and companion test script. (`49f47ca`)
- **Caching** OpenAI: use Responses `instructions` for stable prompt cache behavior. (`235fe28`)
- **Caching** OpenAI: structured Responses input shape for prompt caching. (`8cdddfc`)
- **Caching** Gemini: move `system_instruction` and tools into `CachedContent` so explicit context caching stays valid. (`e81ff59`)
- **Quality / Tooling** Added OpenAI cache diagnostic back-to-back probe script. (`5074512`)
- **Providers** OpenAI / Grok: debug-log request prefix and hash for cache diagnosis. (`74cf3b4`)
- **Synthesizer** Summarize oversized provider outputs with Gemini Flash before synthesis; tune aggregate input size and 100k input budget. (`1ddedf6`, `844bac0`)
- **Verifier** Raised verifier output budget and salvage truncated JSON when possible. (`bbcd65e`)
- **Iterative** Configurable verifier provider with Gemini default; hardened verifier JSON parsing, raw response persistence, and verifier prompt. (`9888d0c`, `e8c7374`)
- **Prompt template** Require last closing price from sources; treat YAML prices as hints only. (`dbf58c1`)
- **Prompt template** Expanded persona to improve OpenAI / Gemini Flash prompt caching. (`0efc041`)
- **Drive upload** Load Drive upload settings from `.env`; clarify upload plan and credential validation at startup; suppress noisy tracebacks on malformed credentials; pre-flight Shared Drive check with clearer `storageQuotaExceeded` guidance. (`a44744f`, `0f8ffd9`, `bfcaed0`, `c0a98a8`)
- **OAuth** Widened OAuth scope to full Drive for access to manually created folders. (`da464c9`)
- **PDF output** Write PDF alongside Markdown for analysis artifacts. (`389449d`)
- **PDF output** Pin compatible WeasyPrint/pydyf and improve error routing when rendering fails. (`df6f586`)
- **Scripts** Replaced `mapfile` with a portable loop for macOS bash 3.2. (`5b4e755`)
- **Drive upload** Auto-upload run outputs to Google Drive via service account. (`ae3a82b`)

### 2026-05-09

- **Iterative** Use `AsyncSqliteSaver` for async LangGraph checkpointing. (`cb49ff4`)
- **Caching** Gemini: repair `count_tokens` contents handling and graceful cache-feasibility fallback. (`74552b1`)
- **Caching** Added Gemini to fan-out with caching enabled. (`e4b272b`)
- **Prompt template** Removed sections 5 and 10 (meta-prompts) from the equity analyst template. (`89b9976`)
- **Synthesizer** Upgraded synthesizer default to Gemini 3 Pro and expanded synthesizer prompt for caching. (`704a354`)
- **Configs** Hybrid “fast” preset: one deep-search provider plus two fast reasoners. (`7286b1d`)
- **Providers** Anthropic: raise per-provider timeout to 600s when web search is forced. (`82aed24`)
- **Providers** Anthropic: force tool use and require `web_search` in persona to reduce refusals. (`5bbb5ad`)
- **Quality / Tooling** Moved persona and synthesizer prompts into editable text files. (`85358a4`)
- **Providers** Honor per-provider `request_timeout_s` for OpenAI/Grok web-search runs. (`06c915f`)

### 2026-05-08

- **Prompt template** Request six quarters of historical post-earnings context instead of eleven. (`d7bbd9e`)
- **Caching** Gemini: explicit context caching for the static prompt prefix. (`a5893bd`)
- **Caching** Log OpenAI and Grok prompt cache statistics. (`6d05b7b`)
- **Providers** Anthropic: streaming Messages API for long web-search requests. (`3c5d336`)
- **Providers** OpenAI: streaming Responses API for long web-search runs. (`b1fd008`)
- **Caching** Anthropic: prompt caching with 1h TTL on system prompt and tools. (`15a2529`)
- **Synthesizer** Raised fan-out `max_output_tokens` defaults and allow per-provider overrides; separate larger synthesizer output budget to reduce truncation. (`fb547db`, `30eaa8b`)
- **Quality / Tooling** Configurable provider and synthesizer models (Opus default, optional Gemini synthesizer). (`9013ea0`)
- **Retry / Backoff** Hardened error handling, retries with exponential backoff, and synthesis input filtering. (`f70f624`)
- **Quality / Tooling** Parallelism, per-call timeouts, optional web search, and end-of-run timing summary. (`06a8a9b`)
- **Quality / Tooling** Structured logging for agent progress. (`12f202b`)
- **Iterative** LangGraph iterative refinement loop with verification, routing, and checkpointing. (`6baaaaf`)
- **Providers** Grok (xAI) provider with Live Search. (`d996490`)
- **Providers** Gemini provider with Google Search grounding. (`961f95e`)
- **Providers** Initial MVP: Anthropic and OpenAI fan-out plus synthesizer. (`0caed42`)
