# Multi-LLM equity analyst

Python CLI that renders a Jinja2 equity/options prompt from YAML config, fans out to multiple LLM providers in parallel, and synthesizes a consensus report. Optional **iterative** mode runs a LangGraph loop: multi-provider fan-out, synthesis with per-round confidence parsing, web-grounded verification, routing, and final packaging with SQLite checkpointing for resume.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## API keys

Copy `.env.example` to `.env` and set keys for the providers you enable in config:

- `ANTHROPIC_API_KEY`
- `OPENAI_API_KEY`
- `GEMINI_API_KEY`
- `XAI_API_KEY` (Grok)

## Configs

Stock-specific YAML lives under `configs/`. Copy either file as a starting point for a new symbol and edit prices, dates, and lookbacks.

| File | Use case |
|------|----------|
| `configs/mndy_2026_05_08.yaml` | Default MNDY run: all four fan-out providers (**Anthropic**, **OpenAI**, **Grok**, **Gemini Flash**) use long timeouts; web search follows each provider’s default (typically on). **Synthesis** runs on **Gemini 3.1 Pro Preview**. Best for highest-quality grounded research. |
| `configs/mndy_2026_05_08_fast.yaml` | Hybrid speed: **OpenAI** alone runs deep `web_search`; **Anthropic**, **Grok**, and **Gemini Flash** reason without search; **Gemini 3.1 Pro Preview** synthesizer has no extra search. Shorter wall-clock for iteration. |

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

YAML may set **per-provider** overrides: optional `model` (API model id for that backend), optional **`max_output_tokens`** (completion budget for that fan-out provider only; falls back to global `max_output_tokens`), `web_search: true|false`, and optional **per-provider timeouts** (`request_timeout_s`). Global defaults: `request_timeout_s` (default **180** seconds), `max_output_tokens` (default **16000** for fan-out — each parallel provider gets this unless overridden per provider), `synthesizer_max_output_tokens` (default **24000** for the final synthesis pass — larger because the synthesizer must weave every provider’s output across all 13 sections), and `verifier_max_output_tokens` (default **1536**, iterative verifier only). **OpenAI** long web-search runs use the **streaming** Responses API so the HTTP connection stays alive across multi-minute tool loops; if you still hit `asyncio` timeout errors, raise **`request_timeout_s`** globally or on specific providers (for example `900` or `1500` seconds on `openai`).

Default fan-out budget is **16,000** tokens per provider. For a 13-section deep-research prompt this is usually enough to avoid truncation in `claude.md` / `openai.md` / `grok.md` / `gemini.md`; if a run still cuts off, raise the global `max_output_tokens` or set a higher **`max_output_tokens`** on specific providers in YAML. Override from the CLI with **`--max-output-tokens`** (fan-out), **`--synthesizer-max-output-tokens`** (synthesis), and **`--verifier-max-output-tokens`** (iterative verify step).

**Anthropic** defaults to **Opus** (`claude-opus-4-7` unless you set `model`), which has substantially higher input-token rate limits on the standard tier (on the order of **~500k** tokens per minute) than Sonnet (often around **~30k** tokens per minute). Override with `model` under `providers` if you need a different snapshot. See [Anthropic model IDs](https://docs.anthropic.com/en/docs/about-claude/models/model-ids-and-versions). Long-running calls (including web search) use the **streaming** Messages API, as required by the Anthropic Python SDK for requests that may exceed ~10 minutes.

**OpenAI** uses the **streaming** Responses API (`stream=True`) for all fan-out calls so long web-search tool loops stay connected instead of timing out on a single blocking HTTP response.

**Synthesizer** defaults to **Gemini** (`gemini-3.1-pro-preview` per `GeminiProvider.DEFAULT_GEMINI_MODEL`) so synthesis runs on a different model than typical Anthropic fan-out, which avoids stacking the same provider’s rate limits on both parallel answers and the long synthesis prompt. Configure it as a string (`synthesizer: gemini`) or as an object with `name`, optional `model`, optional `web_search`, and optional `request_timeout_s`.

```yaml
request_timeout_s: 180
max_output_tokens: 16000
synthesizer_max_output_tokens: 24000
verifier_max_output_tokens: 1536
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
```

The simple list form remains valid: `providers: ["anthropic", "openai"]`. A bare `synthesizer: openai` string is still accepted.

### Run timing (`run.json`)

After a **live** standard run, `run.json` includes a **`timing`** object: `parallel_provider_batch_wall_s`, `synthesis_wall_s`, `total_wall_s`, and `per_provider` latency snapshots. Iterative runs add a **`timing`** summary when `finalize` runs (per-iteration provider, synthesis, and verify wall times plus `total_sequential_wall_s`). INFO logs also emit a heartbeat every **30s** while waiting on a provider batch.

## Iterative mode

Iterative runs use a LangGraph `StateGraph` with nodes `fan_out` → `synthesize` → `verify` → `route` → (`fan_out` again or `finalize`). Each round:

1. **fan_out** – all configured providers answer the prompt plus any accumulated follow-up questions.
2. **synthesize** – the synthesizer must emit section confidences and a line `OVERALL_CONFIDENCE: <0.0-1.0>` for routing.
3. **verify** – default Anthropic-backed verifier returns JSON `verified` / `contradicted` / `unverifiable` using web search when enabled.
4. **route** – finalize if `len(rounds) >= max_iterations`, or if overall confidence meets `--confidence-threshold` and there are no contradictions; otherwise append follow-ups and loop.
5. **finalize** – writes `synthesis.md`, per-round files under `iterations/`, and `checkpoint.sqlite`.

```bash
python -m equity_analyst run --config configs/mndy_2026_05_08.yaml --iterative \
  --max-iterations 3 --confidence-threshold 0.85
```

Iterative dry-run (compiles the graph and prints an excerpt of the rendered prompt):

```bash
python -m equity_analyst run --config configs/mndy_2026_05_08.yaml --iterative --dry-run
```

## Resume after a crash

Iterative runs store checkpoints at `outputs/<run_id>/checkpoint.sqlite` and metadata in `run.json`. Resume with the output folder name (e.g. `MNDY_20260509T120000Z`):

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

## Customizing prompts

These plain-text and template files control model instructions without editing Python:

- `prompts/equity_analyst_system.md` — persona / instructions (cached as the Anthropic system prompt and prepended for other providers).
- `prompts/equity_analyst.j2` — the 13 numbered sections, a Jinja template with `{{ symbol }}`, `{{ today_low }}`, and the other template variables.
- `prompts/synthesizer_system.md` — how the synthesizer compares provider answers and formats the consensus.

Edits take effect on the next CLI run; you do not need to change code or restart a long-lived process.

## Development checks

```bash
ruff check .
mypy --strict equity_analyst
pytest -q
```
