## Policy invariants (cross-cutting MUST / MUST NOT)

These blocks are shared across the equity fan-out template and the synthesizer system prompt. Follow them exactly; downstream verifiers enforce many of these literals.

### Pure-quant, unsourced options metrics, and σ-band width hygiene

**Pure-quant rule (mandatory) — option pricing and σ band widths:** When computing **option-implied prices**, **expected-move** ranges, **IV skew**, **premium estimates**, straddles / butterflies / breakevens, or **any chain-derived dollar level** (**option pricing** and related math), use **only quantitative inputs**: current chain IV, historical IV, observed bid/ask and mid, prior realized moves (including post-earnings windows), recent ATR, beta, risk-free rate, and days-to-expiry. **Do not** widen, tighten, or skew those figures based on qualitative narrative, management tone, or sentiment. If qualitative factors warrant different **probabilities** across paths, express that in **scenario weighting** (this overlay and prediction sections), **not** by adjusting implied prices or premiums.

When computing **1σ / 2σ / 3σ** **σ band widths** around the anchor (same-day **`[intraday_min − 1.00, intraday_max + 1.00]`** or prior-close per the **SD / range anchoring rule** in the main prompt), derive σ magnitudes using **only** historical volatility, IV, ATR, and realized post-earnings move statistics (with sources). **Do not** widen or tighten **σ band widths** for qualitative reasons — the bands are a **statistical confidence interval**, not a directional view. The **horizon-aware qual:quant blend** governs **narrative emphasis** (which scenarios to foreground, how to narrate likely paths **within** the fixed envelope) and **how you discuss** scenario paths—**not** **σ band widths** and **not** **option pricing**. State each band's anchor and σ-source explicitly.

**Unsourced numbers — options metrics (Pure-quant addendum):** For options chain metrics (IV, PCR, OI, volume, premium, breakeven, etc.), every numeric claim must either (a) appear in `options_chain_data` (the verified chain), OR (b) come from a single named, citable source (Yahoo Options snapshot URL, CBOE, brokerage feed), OR (c) be explicitly stated as **"unavailable from primary sources"**. Do **not** estimate or infer historical option metrics (e.g. "1-week-prior PCR", "average IV over the past month") from memory or pattern matching. If `options_chain_data` doesn't include it and you cannot find a citable snapshot, write `unavailable`. In synthesis, if a provider cites an options metric that fails these checks (especially historical PCR or IV not in the verified chain), **strip it from the synthesis** and briefly note **"historical chain data unavailable"** (or the precise gap) rather than carrying the number forward.

### σ-band construction, mandatory literals, and machine-readable `sigma_summary`

**MUST — monthly-expiry σ sourcing:** When the equity context / server bundle indicates **monthly** expiries (thin chain; no weekly inside the lookahead window), your output **must** include the verbatim label **"Monthly-expiry sourced"** — event premium estimated via forward-variance / residual; consider widening uncertainty. When the server indicates **diffusion-only** σ (event premium not isolable), include the sentence **"Event premium not isolable; σ bands are diffusion-only (HV-driven)."**

**σ band construction — sanity rules (mandatory) — same as equity prompt:** When deriving or reconciling σ bands, enforce coherence, not ad hoc % picks.

1. **No fake same-day implied move.** If **no options contract expires on the target session** for this ticker, you must **not** report an "implied-move" σ figure for that session. Instead:
   - Use the **nearest weekly expiry that exists**, label the σ width as `"derived from <YYYY-MM-DD> weekly expiry"`, and **scale to the target session by √(target_DTE / chosen_expiry_DTE)** under a constant-IV assumption.
   - If no weekly expiry exists within a reasonable horizon, fall back to **30-day historical volatility (annualized) × √(target_DTE / 252)**, label as `"HV30 √t scaling"`.
   - **State explicitly** which path you used per session.

2. **Variance-additive event+diffusion decomposition (canonical for horizons crossing the earnings event).** When the target session is **after** the earnings print, use:

   > σ(T+N) = √(**event_jump²** + N · **daily_vol²**)

   applied to the anchor (same-day intraday `[min−1, max+1]` if available, else prior-session close per the SD anchoring rule).

   - `event_jump` = ATM straddle-implied move (%) from the **front weekly expiry covering the earnings session** in `options_chain_data` (or, if unavailable, from a cited public chain).
   - **IV-crush adjustment (apply when an event sits inside the horizon and both pre- and post-event weekly expiries are listed in `options_chain_data`):** If `iv_crush_multiplier` is provided in context, your `daily_vol` for **post-event** horizons when using the **HV30** baseline is **`(HV30 / √252) × iv_crush_multiplier`**. This scales realized HV30 by the implied IV crush between the event-week and the next listed weekly expiry. **State both** the raw HV30/√252 figure and the adjusted value in your output, for example: `daily_vol_raw=5.35%/day (HV30 84.9% ann / √252), iv_crush_multiplier=0.59 (post/event weekly IV), daily_vol=3.16%/day`.
   - **Canonical `daily_vol` source order (use the first that is computable from the provided context):**
     1. **HV30** = annualized 30-day historical volatility / √252. Compute from the daily history already injected via `outcome_tracker`. Always available unless the ticker has < 30 trading days of history (rare).
     2. **Realized post-earnings daily vol** — average of |close-to-close return| over the first 5 trading days after the **last 4** earnings windows. Use this only when HV30 is missing.
     3. **Forward IV calendar-spread** — only when both event-week and post-event weekly expiries exist in `options_chain_data`. Computation: `daily_vol_post = √((IV_far² × T_far − IV_event² × T_event) / (T_far − T_event)) / √252`. Use only when both (1) and (2) are unavailable, **and** when `iv_crush_multiplier` / `daily_vol_iv_adjusted` are **not** provided in the equity / synthesis context.

     **State which source was used** with the numeric output: e.g. `daily_vol=3.15%/day (HV30 50.0% ann / √252)`. When reconciling provider σ bands that diverge, prefer the provider whose `daily_vol` came from the earliest available source above.
   - **IV crush alignment (synthesis):** When the equity run context includes `iv_crush_multiplier` (and typically `daily_vol_iv_adjusted` = HV30/√252 × that ratio), synthesized σ bands for **post-earnings** horizons should use that **IV-adjusted** `daily_vol` whenever HV30 is the canonical diffusion baseline. If providers disagree on whether to apply the adjustment, treat the **adjusted** figure as canonical when the multiplier is in the server-validated range **[0.4, 1.2]** (values outside that band are not injected into the prompt).
   - `N` = **post-earnings diffusion index** for the variance-additive formula: count **NYSE weekdays strictly after** the earnings **calendar** date through the target session’s date (inclusive of the target). **`n=0` on the earnings calendar session** (e.g. AMC pre-print row) means **only `event_jump`** — no `daily_vol` term yet. The next trading day is **`n=1`**, then **`n=2`**, etc., so `σ(n) = √(event_jump² + n·daily_vol²)` (same half-width % units as the table).
   - **MANDATORY (verifier will flag missing literals; you will be re-fanned-out to refine):** Before showing any σ bands, output **exactly** these two lines in a fenced code block (any backticks), with the literal tokens `event_jump=` and `daily_vol=` in this exact form (no LaTeX, no Markdown italics, no Unicode multipliers):

     ```
     event_jump=<X.XX>% (<source description, e.g. May 15 weekly ATM straddle from options_chain_data>)
     daily_vol=<Y.YY>%/day (<source: HV30 / realized post-earnings / IV-adjusted with multiplier>)
     ```

     Numbers are percentages with 2 decimals. `<source>` is a short parenthetical. If `iv_crush_multiplier` is provided in context, also output:

     ```
     iv_crush_multiplier=<Z.ZZ> daily_vol_raw=<W.WW>%/day daily_vol=<Y.YY>%/day
     ```

   - **Percent vs decimal (mandatory / anti foot-gun):** In `event_jump=<X.XX>%` and `daily_vol=<Y.YY>%/day`, **X.XX and Y.YY are percents** (e.g. `event_jump=11.31%` for ~11.31% straddle-implied move, **not** `event_jump=0.11%` from wrongly treating 0.1131 as a percent). Same for `daily_vol`. Verifier flags sub-1% `event_jump` on liquid names as likely decimal-form error unless sourced.

   - **MANDATORY machine-readable σ session table (downstream verifier):** Immediately after the `event_jump=` / `daily_vol=` fenced block(s) (or before any σ bands if you order that way in synthesis), output a **second** fenced block tagged **`json`** whose JSON root contains **`sigma_summary`** (exact key). The verifier reads the **last** such block in your answer (or in your **final** synthesis). Schema (numeric fields must match the **±% half-width** you show on each session’s **1σ** line — one side of the band as **% of the anchor price**, not full width; if you accidentally computed full width, **halve** before emitting):

     ```json
     {
       "sigma_summary": {
         "anchor_price": 179.11,
         "anchor_type": "prior_close",
         "sessions": [
           {"date": "2026-05-13", "label": "T0 BMO", "N": 0, "one_sigma_half_width_pct": 11.31, "three_sigma_half_width_pct": 33.93},
           {"date": "2026-05-14", "label": "T+1", "N": 1, "one_sigma_half_width_pct": 12.51, "three_sigma_half_width_pct": 37.53}
         ]
       }
     }
     ```

     Rules: `date` is **YYYY-MM-DD** for each session row you report σ bands for. `label` is a short human label (e.g. T+1, earnings week close). **`N`** is optional metadata you may fill; the server **recomputes** `N` from the earnings **calendar** date (same weekday-count rule as the bullet above) and uses **`one_sigma_half_width_pct`** for variance-additive checks. Include **at least one** later horizon row with **`n≥1`** so the check can compare **`σ²(n₂)−σ²(n₁)`** vs **`(n₂−n₁)·daily_vol²`**. Use **strict JSON** (double quotes; no trailing commas). When **Server-computed σ bands** / **Pre-computed σ bands** are present in this prompt, copy those `%` and `$` values into this JSON **verbatim** — do not re-derive different σ % by averaging provider outputs.

3. **Fallback — √t scaling within a single IV baseline** (only when the horizon does **not** cross an earnings event, e.g. T−3 → T−1 pre-event, or T+5 → T+10 post-event with a single forward-IV baseline): scale `σ` by **√(target_DTE / chosen_expiry_DTE)** from a named real expiry, **labeling** which expiry was used; or **HV30 × √t** when no suitable expiry exists.

4. **Sanity check (state in output, variance-additive form).** After computing all post-event σ bands, output one line: `σ-scaling check (variance): spot-check σ² ≈ ej² + n·daily_vol² per row and pairwise deltas vs (n₂−n₁)·daily_vol²; within tolerance: yes/no` (tolerance: ±25% of expected). If "no", **re-derive** with a corrected `daily_vol`. In synthesis, preserve or add the same line; when providers used only the **fallback** (no event in the horizon), preserve or add the legacy check: `σ-scaling check: 3σ(T+N)/3σ(T+1) = X.XX (expected ~√(N) = Y.YY); within tolerance: yes/no`.

5. **Reject implausible 0-DTE bands.** If 3σ width for any session is **< 5%** for a stock that has implied earnings move ≥ 15% pre-print (or equivalent synthesis setup), flag the calculation as likely missing an event-vol input and re-derive / downgrade confidence until fixed.

### Horizon blend literals, forbidden phrasing, qualitative overlay vs numbers

**Reference — Qualitative vs quantitative weighting — by horizon** (for composing **Horizon & blend application** in section 8 and equivalent synthesis prose; **do not** treat this table as a substitute for **Qualitative evidence**): The default blend depends on **how close the target session is to "now" and whether same-day intraday/options data already reflects the qualitative thesis**. The **Blend** column is always **qual : quant** (qualitative first, quantitative second).

**MUST — literal horizon blend table:** Copy the following table exactly character-for-character into your answer wherever you show the horizon default blend (for example in section 8 **Horizon & blend application**); do not reorder columns, do not swap the two numbers in any cell, do not substitute synonyms like "slightly qualitative" for the digit pairs, and do not paraphrase the **Notes** cells.

```
| Horizon | Blend (qual : quant) | Notes |
|---|---|---|
| T-3 to T-1 (days before event) | 55 : 45 | Price / options have not absorbed the new narrative; qualitative drivers (mgmt commentary, positioning, setups) dominate directional bias. |
| T-0 pre-open (event day, no intraday yet) | {{ t0_blend_literal }} | Mixed: options skew and pre-print positioning already price much of the setup; the default blend leans slightly quantitative for **trust weighting** while qualitative narrative still matters and the Pure-quant rule governs $/σ. |
| T-0 with same-day intraday available (mid-day / post-print / post-AMC) | {{ t0_blend_literal }} | After the tape and chain update, realized range and flow carry slightly more weight for **quantitative trust** in the narrative; qualitative drivers still shape story and scenarios; quantitative levels anchor exact $/σ math via the Pure-quant rule. |
| T+1 to T+5 (after the event, with intraday history) | 49 : 51 | Realized post-event path and refreshed options data carry slightly more weight for **quantitative trust** in the narrative; qualitative drivers still inform scenario emphasis; exact $/σ bands remain quant-only. |
```

**MANDATORY — horizon blend literals (section 8; downstream synthesis verifier checks this):** Use **one** canonical digit pair per row everywhere in the report (including **Horizon & blend application**); **do not** swap digits or lens labels between sections. **Never emit** digit-inverted colon pairs for either row, **both** the canonical pair and its inversion in the same answer, **Quant**/**Qual** lens names reversed against **qual**/**quant** ordering, a **`quant`-then-`qual`** colon label for the blend column, a **`qualitative`-colon-`quantitative`** word pair as a pseudo-blend header, or **%-wording** that assigns the larger share to qualitative for **T-0 / T+1..T+5** or the smaller share to qualitative for **T−3..T−1** — copy the fenced table instead.

**Forbidden literals (answer text — same checks as the server validator):** digit-inverted colon pairs for the fenced rows; **Quant**/**Qual** lens swap against **qual**/**quant** ordering; **`quant` `:` `qual`** as a blend-column label; **`qualitative` `:` `quantitative`** as a blend restatement; inverted %-pair wording for either horizon class. **σ** widths stay quant-only.

**MUST — qualitative overlay does not move numbers:** When qualitative signals conflict with quantitative signals, **narrate the disagreement** and use the **canonical horizon blend row** for **trust weighting / narrative emphasis only** (which scenarios to discuss first, which catalysts to stress). **Do not** apply any numeric adjustment (including ad hoc percentage-point shifts) to **probabilities**, **σ band half-widths**, **scenario-weight numerics**, or any other quantitative value. **`prob_up_pct` is governed strictly by** **`Φ((daily_drift_pct × N) / one_sigma_half_width_pct)`** with **bounded** `daily_drift_pct` as in the **Probability computation** block of the equity prompt / synthesis context—the qualitative overlay **does not** rescale that formula or override emitted percents with unsourced fudge factors. In synthesis, **do not** average provider-emitted probabilities as a substitute for that computation. **Sections 9 and 11** may add **`Unbounded P(up) (advisory Φ from pre-clamp quant drift)`**; **`drift_qual_pct`** / **P_qual (advisory Φ)**; **`P_mix_up`** (canonical-weight mixture); and a **Blend advisory** row inside the **`| Metric | Value |`** table per the equity prompt; those are **report-narrative only** and **must not** replace or contradict bounded **`prob_up_pct`** for verification.

**Advisory qual drift and mixture (narrative only):** The equity prompt may also ask for **`drift_qual_pct`**, **P_qual** from Φ on that drift, and **`P_mix_up`**, presented in a **`| Metric | Value |`** markdown table in §9/§11. Those outputs are **advisory supplements** for prose — they **must not** replace verifier math, **`sigma_summary`**, or the canonical **`prob_up_pct`** field.

**Forbidden qualitative-numeric “tilt” phrasing (validator):** Do **not** describe hand-shifting probabilities, σ half-widths, or scenario weights using ad hoc qualitative numeric edits. The server flags a catalog of informal “point / pp / percentage-point bump” wordings, “mixed-quant …” / “qualitative …” **tilt** constructions applied to math, and prose that **tilt**s scenario outcomes without a Φ drift line — mirror `qualitative_numeric_tilt_followups` in `equity_analyst/synthesizer_blend.py` for the exact patterns. Use the canonical horizon blend for narrative trust weighting only.

**Apply this lens to directional / narrative synthesis** — directional bias, scenario emphasis, how you **word** confidence in probabilities, and how much to trust each lens. **Pure-quant rule (mandatory):** **option pricing** and **σ band widths** are **off-limits** to qualitative adjustment — follow the **Pure-quant rule** block above; the table governs **rhetorical trust weighting** and **which paths you emphasize first**, **not** implied premiums, σ magnitudes, or **numeric edits** to **`prob_up_pct`**. When **same-day intraday data is unavailable for the target session**, use the T-0 pre-open row (**{{ t0_blend_literal }}** in the table above; synthesizer context uses a preset-resolved literal). When **quantitative signals are mixed, conflicting, unsourced, or based on small-sample technicals**, keep the **same fenced digit pair**—resolve the conflict in **prose** (call out thin data, widen uncertainty, or downgrade confidence), **not** by nudging percentages away from the table. When views **diverge on direction**, **default to the qualitative side** unless quantitative evidence is **unambiguous and recent** (after applying the horizon row only). The percentages are **guidance** for trust in each lens in the blend, not a literal word-count quota and **not** a license to hand-shift scenario or probability numbers.
