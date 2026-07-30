# RMT "Propagation of Chaos" Herding Index — Research Note

**Status: prototype, NOT integrated with the overseer. Shelved pending a
decision on the position-concentration use (see below).**

Date: 2026-07-30. Code: `schwab/rmt_chaos.py`, `test_rmt_chaos.py`,
`schwab/download_sensor_universe.py`. Saved series: `data/rmt_chaos_index.csv`.

## Motivation

Deng–Hani–Ma's resolution of Hilbert's sixth problem (arXiv:2503.01800)
proves that a statistical/mean-field description of a many-body system is
valid **iff propagation of chaos holds** — micro-correlations stay negligible.
Inverting that for markets: our statistical price predictors (TimesFM, Kronos,
the Vault8 BiLSTM range model) should only be trusted while the market's
cross-sectional correlation structure is "chaotic" (noise-like). When a
herding/market mode dominates, chaos is broken and those predictors are
outside their domain of validity.

Random Matrix Theory gives a parameter-free test of that condition. For N
assets over T days, the eigenvalues of the sample correlation matrix of
pure-noise returns fall inside the Marchenko–Pastur bulk `[λ-, λ+]`,
`λ± = (1 ± √(N/T))²`. A genuine market/herding mode appears as the top
eigenvalue detaching above `λ+` and absorbing a large share of total variance.

**Metrics computed** (`decompose`):
- `market_mode_frac = λ_max / N` — headline herding measure, ranges 1/N (pure
  chaos) → 1 (total herding).
- `absorption_ratio` — share of variance in the top ~20% of modes
  (Kritzman–Li–Page–Rigobon 2010).
- `detachment = λ_max / λ+`, `n_deviating` (# eigenvalues above the bulk),
  `effective_rank` (exp-entropy of the spectrum).

## Sensor universe

The index is a **market-state sensor, not a stock picker** — the universe it
measures need not be the universe traded. `download_sensor_universe.py` pulls
~105 large-cap US names, ~10 per GICS sector, so no sector dominates the
correlation bulk. ETFs are deliberately excluded (an ETF is a linear combo of
members and would fabricate a dominant eigenvector).

## Findings

Validated on the 105-name universe back to 2006 (VIX from `data/vix_history.csv`,
forward tests vs SPY).

1. **It is a correct herding sensor.** COVID-2020 peaked at market-mode-frac
   0.612 vs a modern-era median 0.351 (+74%); it rises monotonically across the
   scanner's VIX regimes (Reclamation 0.30 → Wasteland 0.37 → Nuked 0.40).

2. **It does NOT lead VIX.** Lead-lag on daily changes is coincident (0-day,
   corr ≈ 0.20). It moves *with* the vol complex, not ahead of it.

3. **Unconditionally it looks predictive of forward vol** — top-20% herding
   days precede 21-day realized vol of ~20.6% vs ~14.7% for the rest. This gap
   was **stable across 47-, 65-, and 105-name universes** (evidence the effect
   is real, not universe cherry-picking).

4. **But the signal is subsumed by VIX — this is the kill test.** Conditioning
   on VIX, within calm days (VIX < 20) top-20% herding precedes forward vol of
   **11.9% vs 12.2%** — no separation. The unconditional gap in (3) was VIX in
   disguise: herding is high exactly when VIX is high, and VIX carries the
   forward-vol information. Once VIX is controlled for, the residual signal is
   ≈ 0.

**In hindsight this is the expected result:** average correlation and VIX are
both driven by the same latent fear factor, so for a scalar target like forward
vol they are collinear and VIX wins. RMT's unique output was never the *level*
(VIX already captures that) — it is the cross-sectional *structure*.

## Verdict

- **Do not wire this in as a volatility / risk gate.** It adds a 105-name data
  pipeline for ~zero incremental signal over the VIX regime already in use.

- **What survives:** the eigen*vectors*, not the eigenvalue. VIX is a scalar and
  physically cannot see *which* names are co-moving; the decomposition can. That
  maps onto an existing rule in the Overseer prompt — *"avoid stacking
  correlated same-sector names whose puts would all be assigned in the same
  drawdown."* Today that is an LLM judgment call; the correlation matrix over
  {open positions + candidate} makes it quantitative: before opening a new put,
  block it if it piles onto a mode the current book is already loaded on. This
  is a **position-concentration tool, not a timing tool**, and does not need the
  forward-vol machinery — just the current correlation matrix. Worth building
  only if positions are in fact ending up correlated; shelved otherwise.

- **Secondary:** today's herding sits at the ~1st percentile (extreme
  dispersion / low correlation), independently confirming the calm VIX regime.
  A cross-check, not new information.

## How to run

```bash
# refresh / download the sensor universe (~105 names)
python schwab/download_sensor_universe.py            # --force to re-fetch all

# full report (herding level, stress episodes, VIX lead-lag, forward tests)
python schwab/rmt_chaos.py --window 252 --save       # writes data/rmt_chaos_index.csv
python schwab/rmt_chaos.py --universe bluechips      # compare on the 47 blue chips

# tests
pytest test_rmt_chaos.py -q
```

## References

- Deng, Hani, Ma (2025), "Hilbert's sixth problem: derivation of fluid
  equations via Boltzmann's kinetic theory", arXiv:2503.01800.
- Laloux, Cizeau, Bouchaud, Potters (1999), "Noise dressing of financial
  correlation matrices".
- Kritzman, Li, Page, Rigobon (2010), "Principal Components as a Measure of
  Systemic Risk" (the Absorption Ratio).
