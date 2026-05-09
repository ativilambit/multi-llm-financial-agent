# Performance plan — multi-LLM equity analyst

## 1. Observed timings (reference run)

**Path:** `outputs/MNDY_20260509T040039Z/run.json`

This artifact is a **dry run** (`"dry_run": true`). There are **no measured provider latencies, token counts, or synthesis timings** in that file—only config, template path, and `web_search: true` per provider.

**Inference for real runs (from code structure):**

- **Standard mode:** Wall-clock for the provider phase is approximately **max(individual provider latencies)** because `asyncio.gather` runs tasks concurrently (`orchestrator.py` L117–L122).
- **Synthesis** runs **after** all providers finish; total run time ≥ provider batch wall time + synthesizer latency.
- **Iterative mode:** Each round does **fan_out (parallel providers) → synthesize (one LLM) → verify (one LLM with web search)** sequentially on the graph (`iterative.py` L273–L276). Up to `max_iterations` rounds multiplies cost.

## 2. Ranked bottlenecks (with evidence)

| Rank | Bottleneck | Evidence | Impact |
|------|------------|----------|--------|
| 1 | **Web search enabled by default on every provider and verifier** | `enable_web_search=True` default in `orchestrator.py` L53, `iterative.py` L299; Anthropic attaches `web_search` tool L30–L31 `anthropic_provider.py`; OpenAI/Grok `tools` L26–L27 / L31–L32; Gemini `GoogleSearch` L27–L30; verifier instruction says “Use web search” `iterative.py` L26–L27 | **High** — tool-grounded calls often dominate wall time (tens of seconds to minutes per call). |
| 2 | **No per-call timeout** — one hung provider blocks the whole `gather` | `orchestrator.py` L119 `await provider.generate(...)` with no `asyncio.wait_for`; same in `iterative.py` L135 | **High** — tail latency unbounded. |
| 3 | **Iterative mode re-runs full provider fan-out each round** | `fan_out` rebuilds prompt and calls all providers every iteration `iterative.py` L117–L138; graph loops via `route` L199–L227 | **High** (expected by design) — cost scales with iterations × providers. |
| 4 | **Verifier always sees full synthesis + web search** | `verify` passes entire `syn` into prompt `iterative.py` L187–L190; instruction encourages web search L26 | **Medium** — extra long generations and searches after each synthesis. |
| 5 | **Single slow model / large max output** | Anthropic `max_tokens` fixed 4096 `anthropic_provider.py` L27; no shared config knob in `config.py` | **Medium** — long completions increase time; no global cap from YAML/CLI. |
| 6 | **`asyncio.gather` without `return_exceptions=True`** | `orchestrator.py` L122; `iterative.py` L138 | **Medium** — one exception fails the entire batch after other work may have completed. |
| 7 | **Parallelism is real (async SDKs)** | `AsyncAnthropic`, `AsyncOpenAI`, `aio.models.generate_content` — not sync clients blocking the loop | **Low** as bottleneck — parallelism is already correct; wall clock is still **slowest-wins** among peers. |

## 3. Proposed optimizations

| Change | Expected impact | Risk |
|--------|-----------------|------|
| `asyncio.wait_for` around each provider `generate` with `request_timeout_s` (default 180), configurable per provider in YAML | High — caps tail latency | Medium — timeouts may truncate rare legitimate long runs; surfaced as error responses in artifacts |
| `max_output_tokens` (default 4096) on `RunConfig`, plumbed into all providers | Medium — shorter generations | Low — may trim very long reports if set too low |
| CLI `--web-search` / `--no-web-search` (keep `--enable-web-search` as alias) + per-provider `web_search` in YAML | High when disabled | Low — default remains search-on for backward compatibility |
| `asyncio.gather(..., return_exceptions=True)` + structured error `ProviderResponse` rows | Medium — resilience | Low |
| Narrow verifier prompt + lower verifier `max_output_tokens`; optional excerpt of synthesis focused on numbers / “Low” confidence lines | Medium | Low — may miss rare narrative contradictions |
| Heartbeat log every 30s while provider batch in flight | Low (UX) | Low |
| `run.json` timing section: batch wall, per-provider latency (existing), synthesis, totals | Low (observability) | Low |
| (Deferred) Cache provider outputs for unchanged sub-prompts across iterations | Medium | Higher — correctness/staleness risk |

## 4. Implementation checklist

- [x] `equity_analyst/config.py` — `ProviderConfig`, coercion for `providers`, `max_output_tokens`, `request_timeout_s`
- [x] Providers — `max_output_tokens` in API kwargs; `generate` supports timeout at call site (orchestrator wraps)
- [x] `orchestrator.py` — effective web search per provider, `wait_for`, `gather` + exceptions, heartbeat, timing in `run.json`
- [x] `iterative.py` — same patterns in `fan_out` / `verify`; state carries `provider_configs`
- [x] `cli.py` — web-search flags; pass timeouts from config where applicable
- [x] `README.md` — document flags and YAML shape
- [x] Tests — timeout, partial failure, YAML web_search override, timing keys in `run.json`
