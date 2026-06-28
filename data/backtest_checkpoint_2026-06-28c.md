# Backtest Checkpoint — 2026-06-28c (Full 18-Symbol 10-Year Run)

## What Changed Since Last Checkpoint

- Extended all 18 symbols from 4yr (2022–2026) to 10yr (2015–2026)
- Fixed `ZeroDivisionError` in Black-Scholes: sub-$1 split-adjusted prices (e.g. NVDA in 2015) caused `round(close × 0.95, 0) = 0.0` strike → K=0 → division by zero. Fixed by rounding to 2dp and adding `strike <= 0` guard in scavenger.py, plus `K <= 0 / S <= 0` guard in options_pricer.py.
- Git: `main` (post-fix commit)

---

## Full 18-Symbol Overseer Backtest (10yr / 11.5yr)

- **Period:** 2015-01-02 → 2026-06-26 (11.47 years)
- **Capital:** 100 shares / 1 contract per trade, no compounding
- **Watchlist (18):** NVDA, AMD, AAPL, AMZN, META, MSFT, GOOGL, IBM, INTC, IONQ, KO, MMM, XOM, PG, TSLA, UNH, HD, ABT

| Symbol | Trades | Scavenger | Raider | Chemist | Combined | B&H | Edge |
|--------|-------:|----------:|-------:|--------:|---------:|----:|-----:|
| AMZN | 134 | +$37,097 | +$0 | +$2,019 | +$39,116 | +$21,115 | **+$18,001** |
| UNH | 88 | +$48,345 | +$0 | -$1,236 | +$47,110 | +$32,951 | **+$14,158** |
| META | 123 | +$42,682 | +$8,113 | +$6,329 | +$57,124 | +$46,881 | **+$10,243** |
| IBM | 68 | +$25,292 | +$145 | +$1,364 | +$26,802 | +$16,920 | **+$9,882** |
| TSLA | 115 | +$46,504 | -$3,752 | -$1,117 | +$41,636 | +$36,300 | **+$5,336** |
| IONQ | 35 | +$6,667 | -$202 | +$151 | +$6,616 | +$3,876 | **+$2,740** |
| ABT | 70 | +$7,166 | +$0 | +$255 | +$7,421 | +$5,482 | **+$1,940** |
| MMM | 40 | +$6,597 | -$806 | +$1,551 | +$7,342 | +$7,057 | **+$285** |
| XOM | 83 | +$7,055 | +$0 | -$122 | +$6,933 | +$8,325 | -$1,392 |
| AAPL | 86 | +$20,675 | +$2,331 | +$1,041 | +$24,047 | +$25,503 | -$1,456 |
| INTC | 71 | +$5,837 | +$2,067 | +$971 | +$8,875 | +$10,355 | -$1,480 |
| NVDA | 116 | +$15,112 | +$217 | +$656 | +$15,985 | +$19,201 | -$3,216 |
| KO | 44 | +$905 | +$0 | +$310 | +$1,215 | +$5,427 | -$4,212 |
| PG | 52 | +$2,511 | +$0 | -$320 | +$2,191 | +$9,049 | -$6,858 |
| GOOGL | 99 | +$21,080 | -$226 | +$1,400 | +$22,254 | +$31,000 | -$8,746 |
| MSFT | 90 | +$19,087 | +$2,677 | +$2,754 | +$24,519 | +$33,288 | -$8,769 |
| HD | 90 | +$13,302 | -$780 | +$3,278 | +$15,800 | +$26,344 | -$10,544 |
| AMD | 122 | +$28,497 | -$1,406 | +$2,874 | +$29,965 | +$51,926 | -$21,961 |
| **TOTAL** | **1526** | **+$354,415** | **+$8,378** | **+$22,158** | **+$384,951** | **+$391,000** | **-$6,049** |
| **COMBO** | — | — | — | — | **+$384,951** | **+$391,000** | **-$6,049** |

**Beats B&H: 8/18** — AMZN, UNH, META, IBM, TSLA, IONQ, ABT, MMM

**vs 4yr checkpoint (2026-06-28b):** COMBO edge improved from -$38,620 → -$6,049 over the longer window. Strategy nearly matches B&H at the portfolio level over a full decade.

---

## Annual Returns — All 8 Positive-Edge Symbols

| Symbol | Capital | Strategy P&L | B&H P&L | Edge | **Ann. Return** | B&H Ann. |
|--------|---------|-------------|--------|------|----------------|---------|
| TSLA | $1,258 | +$41,636 | +$36,300 | +$5,336 | **36.0%/yr** | 34.4%/yr |
| AMZN | $1,860 | +$39,116 | +$21,115 | +$18,001 | **30.9%/yr** | 24.5%/yr |
| META | $8,150 | +$57,124 | +$46,881 | +$10,243 | **19.9%/yr** | 18.1%/yr |
| IONQ | $1,077 | +$6,616 | +$3,876 | +$2,740 | **18.7%/yr** | 14.2%/yr |
| UNH | $9,816 | +$47,110 | +$32,951 | +$14,158 | **16.5%/yr** | 13.7%/yr |
| IBM | $9,659 | +$26,802 | +$16,920 | +$9,882 | **12.3%/yr** | 9.2%/yr |
| ABT | $3,735 | +$7,421 | +$5,482 | +$1,939 | **10.0%/yr** | 8.2%/yr |
| MMM | $9,630 | +$7,342 | +$7,057 | +$285 | **5.1%/yr** | 4.9%/yr |

### Live Trading Candidates

| Tier | Symbol | Rationale |
|------|--------|-----------|
| **Primary** | AMZN | Largest edge ($18K), 30.9%/yr, +6.4pp over B&H |
| **Primary** | UNH | Consistent compounder, $14K edge, +2.8pp over B&H |
| **Primary** | IBM | Quiet performer, $9.9K edge, +3.1pp over B&H |
| **Secondary** | META | $10K edge but high capital; needs ~$8K per cycle |
| **Watch** | TSLA/IONQ | Great % returns but tiny capital base; edge modest vs B&H |

### Key Observations

- **AMZN** was invisible in the 4yr run (mostly recovery phase 2022–2026). Full decade reveals it as the best wheel candidate — wide natural range, consistent premium.
- **NVDA** goes negative (-$3.2K) over 10yr — the pre-2022 parabolic run belongs entirely to B&H; covered calls capped every leg.
- **COMBO nearly flat vs B&H** (-$6K on $391K) over 11.5yr — the wheel doesn't beat a diversified basket of these names, but it's far less volatile and generates cash flow.
- **Scavenger dominates**: $354K of $385K total (92%). Raider adds $8K, Chemist $22K.

---

## Bug Fix: Black-Scholes K=0 on Sub-$1 Prices

Split-adjusted historical prices can be < $1 (e.g. NVDA 2015: ~$0.50). Old code used `round(close × 0.95, 0)` which rounds to 0.0. Fix:

- `schwab/options_pricer.py`: `_safe_sigma()` helper; `K <= 0 or S <= 0` guard in both pricers
- `vault76/armory/scavenger.py`: round strikes to 2dp; `if strike <= 0: return` before pricing call

---

## Strategy Parameters

See `data/backtest_checkpoint_2026-06-28.md` for full parameter tables.
See `data/backtest_checkpoint_2026-06-28b.md` for Overseer routing parameters.
