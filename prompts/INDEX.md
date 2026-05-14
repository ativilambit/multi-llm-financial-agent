# Prompts index

Single map of everything under `prompts/` (markdown, Jinja2, and nested policy files). **Synthesizer policy text:** `prompts/policy/invariants.md` is **prepended at runtime** when loading `synthesizer_system.md` (`equity_analyst/prompt_parts.py::_load_prompt_file` replaces `{{ t0_blend_literal }}` with `__T0_BLEND_LITERAL__` before `inject_t0_blend_into_synthesizer_system_prompt` in `equity_analyst/synthesizer.py`). **Equity fan-out:** the same file is **`{% include "policy/invariants.md" %}`** at the top of `equity_analyst.j2` (Jinja `FileSystemLoader` search path is `prompts/`, see `equity_analyst/prompting.py::render_prompt`).

| File | Layer (policy / role / injection) | Loaded by | Used in artifact |
|------|-----------------------------------|-------------|-------------------|
| `INDEX.md` | Policy (documentation) | (human / tests) | This map only |
| `policy/invariants.md` | Policy (cross-cutting MUST / MUST NOT) | `prompt_parts:_load_prompt_file` (prepend for synthesizer); Jinja `include` from `equity_analyst.j2` | Rendered equity user message (`run_dir/prompts/*` fan-out exports); synthesizer system prompt body |
| `equity_analyst.j2` | Role + injection (Jinja2 user template); section 8B mandates ranked drivers then **`Suggested blend (advisory)`** per-horizon integer grid before blend prose | `equity_analyst/prompting.py::render_prompt` (`Orchestrator`, `cli` default `prompts/equity_analyst.j2`) | Provider fan-out markdown; `prompt_export` run artifacts |
| `equity_analyst_system.md` | Role (static system persona) | `prompt_parts:EQUITY_ANALYST_SYSTEM_PROMPT` (`prompting.split_static_dynamic`, providers) | Prefixed to rendered `.j2` body in `RenderedPrompt.text` |
| `synthesizer_system.md` | Role (synthesis instructions); merges provider **Suggested blend (advisory)** grids into one subsection B table | `prompt_parts:_load_prompt_file("synthesizer_system.md")` (`equity_analyst/synthesizer.py` `SYNTHESIS_SYSTEM_PROMPT`) | Synthesizer system prompt (after prepend + T0 literal injection) |
| `provider_summarize_system.md` | Role | `prompt_parts:_load_prompt_file` via `equity_analyst/provider_summarize.py` | Pre-synthesis summarizer system prompt |
| `facts_extract_system.md` | Role | `prompt_parts:_load_prompt_file` via `equity_analyst/facts_packet.py` | Facts packet extractor system prompt |
| `prediction_extract_system.md` | Role | `prompt_parts:_load_prompt_file` via `equity_analyst/prediction_extract.py` | Prediction extractor system prompt |
