# Backtest Checkpoint — 2026-06-28b (Overseer Stock Routing + 10yr Top-3)

## What Changed Since Last Checkpoint

- **Overseer stock-aware routing** (`vault76/overseer.py`): `recommend_roles(regime, stock_ind)` now routes per stock based on its own ADX:
  - RECLAMATION + ADX ≥ 28 → Raider (ride the trend, don't wheel a runner)
  - RECLAMATION + ADX < 28 → Scavenger (sideways stock, collect premium)
  - WASTELAND → Scavenger (income focus regardless of ADX)
  - NUKED_ZONE → Chemist (unchanged)
- Gate applied at FLAT state entry only — open positions play out to completion
- **COMBO row** added to report: shows Combined / B&H / Edge only (per-role columns blank)
- Git: `ef46476` on `main`

---

## Full 18-Symbol Overseer Backtest (4yr, post-routing)

- **Period:** 2022-06-01 → 2026-06-26 (4.07 years)
- **Capital:** 100 shares / 1 contract per trade, no compounding
- **Watchlist (18):** NVDA, AMD, AAPL, AMZN, META, MSFT, GOOGL, IBM, INTC, IONQ, KO, MMM, XOM, PG, TSLA, UNH, HD, ABT

| Symbol | Trades | Scavenger | Raider | Chemist | Combined | B&H | Edge |
|--------|-------:|----------:|-------:|--------:|---------:|----:|-----:|
| TSLA | 45 | +$37,119 | -$2,443 | +$1,751 | +$36,427 | +$17,224 | **+$19,203** |
| UNH | 26 | +$7,044 | +$0 | -$4,839 | +$2,205 | -$8,559 | **+$10,764** |
| MMM | 27 | +$14,099 | +$0 | +$530 | +$14,629 | +$7,261 | **+$7,368** |
| IBM | 22 | +$17,380 | +$0 | +$358 | +$17,737 | +$15,189 | **+$2,548** |
| IONQ | 28 | +$5,209 | -$202 | +$111 | +$5,118 | +$4,415 | **+$703** |
| ABT | 21 | +$1,297 | +$0 | -$887 | +$410 | +$177 | **+$233** |
| XOM | 28 | +$3,859 | +$0 | -$639 | +$3,220 | +$3,739 | -$518 |
| MSFT | 33 | +$9,990 | +$4,573 | +$237 | +$14,800 | +$15,831 | -$1,031 |
| PG | 17 | +$1,260 | +$0 | +$98 | +$1,358 | +$2,619 | -$1,261 |
| HD | 38 | +$6,432 | +$0 | +$348 | +$6,780 | +$8,708 | -$1,928 |
| AMZN | 42 | +$11,178 | +$0 | +$499 | +$11,677 | +$14,171 | -$2,494 |
| KO | 8 | +$327 | +$0 | +$45 | +$371 | +$2,957 | -$2,586 |
| NVDA | 43 | +$13,409 | +$413 | +$484 | +$14,306 | +$17,927 | -$3,621 |
| AAPL | 27 | +$8,319 | +$2,331 | +$343 | +$10,992 | +$14,771 | -$3,779 |
| INTC | 33 | +$2,937 | +$2,398 | +$166 | +$5,500 | +$10,113 | -$4,613 |
| GOOGL | 26 | +$13,606 | +$0 | +$288 | +$13,894 | +$25,157 | -$11,263 |
| META | 39 | +$17,182 | +$8,113 | +$1,947 | +$27,241 | +$46,025 | -$18,784 |
| AMD | 43 | +$17,671 | +$0 | +$707 | +$18,378 | +$45,939 | -$27,561 |
| **TOTAL** | **546** | **+$188,315** | **+$15,183** | **+$1,545** | **+$205,043** | **+$243,663** | **-$38,620** |
| **COMBO** | — | — | — | — | **+$205,043** | **+$243,663** | **-$38,620** |

**Beats B&H: 6/18** — TSLA, UNH, MMM, IBM, IONQ, ABT

**vs previous checkpoint:** COMBO edge improved from -$45,323 → -$38,620 (+$6,703)
- Raider total: +$15,183 (was +$6,404) — routing sent runners to the right role
- TSLA: +$19,203 edge (was +$3,233)
- UNH: +$10,764 edge (was +$760)

---

## Top-3 Extended Backtest: 11.5 Years (2015-01-02 → 2026-06-26)

All three beat B&H over the full period.

| Symbol | Start Px | Capital | Strategy P&L | B&H P&L | Edge | Total Return | **Ann. Return** | B&H Ann. |
|--------|---------|---------|-------------|--------|------|------------|----------------|---------|
| UNH | $98.16 | $9,816 | +$47,230 | +$32,973 | +$14,257 | 481% | **16.6%/yr** | 13.7%/yr |
| TSLA | $12.58 | $1,258 | +$38,344 | +$36,713 | +$1,631 | 3,047% | **35.0%/yr** | 34.6%/yr |
| MMM | $96.30 | $9,630 | +$7,632 | +$6,771 | +$861 | 79% | **5.2%/yr** | 4.7%/yr |

### Key Takeaways

- **UNH is the real winner**: +16.6%/yr vs 13.7%/yr B&H, with $14K edge over 11.5 years. Consistent compounder; wheel extracts meaningful premium. Best risk-adjusted pick for live trading.
- **TSLA returns 35%/yr but barely beats B&H** (+$1.6K edge). Premium income is largely offset by getting called away on big runs. Impressive absolute return, negligible edge.
- **MMM only 5.2%/yr**: modest absolute return; beats its own weak B&H but not compelling in isolation.

---

## Strategy Parameters (unchanged from previous checkpoint)

See `data/backtest_checkpoint_2026-06-28.md` for full parameter tables.

### New parameter: Overseer ADX routing

| Parameter | Value | Location |
|-----------|-------|----------|
| `ADX_RUNNER` | 28 | `vault76/overseer.py` — threshold for runner vs sideways routing |

---

## Code State

- `vault76/overseer.py` — stock-aware `recommend_roles(regime, stock_ind=None)`
- `schwab/backtest_scavenger.py` — routing gate at FLAT entry
- `schwab/backtest_raider.py` — routing gate at FLAT entry
- `schwab/run_backtest.py` — COMBO row in report
- Git: `ef46476` on `main`
