# Backtest Checkpoint — 2026-06-28

## Configuration

- **Roles:** Overseer (Scavenger + Raider + Chemist)
- **Date range:** 2022-06-01 → 2026-06-26 (4.07 years)
- **Capital:** 100 shares / 1 contract per trade, no compounding
- **Watchlist (18):** NVDA, AMD, AAPL, AMZN, META, MSFT, GOOGL, IBM, INTC, IONQ, KO, MMM, XOM, PG, TSLA, UNH, HD, ABT

## Full Results (sorted by Edge)

| Symbol | Trades | Scavenger | Raider | Chemist | Combined | B&H | Edge |
|--------|-------:|----------:|-------:|--------:|---------:|----:|-----:|
| MMM | 28 | +$14,784 | -$2,714 | +$530 | +$12,599 | +$7,261 | **+$5,339** |
| IONQ | 32 | +$9,401 | -$403 | +$111 | +$9,109 | +$4,415 | **+$4,694** |
| TSLA | 43 | +$30,534 | -$11,827 | +$1,751 | +$20,457 | +$17,224 | **+$3,233** |
| IBM | 17 | +$12,973 | +$3,928 | +$358 | +$17,258 | +$15,189 | **+$2,069** |
| AAPL | 29 | +$9,050 | +$6,560 | +$343 | +$15,953 | +$14,771 | **+$1,182** |
| UNH | 29 | +$7,044 | -$10,004 | -$4,839 | -$7,799 | -$8,559 | **+$760** |
| ABT | 23 | +$1,297 | +$142 | -$887 | +$552 | +$177 | **+$375** |
| XOM | 29 | +$3,859 | +$800 | -$639 | +$4,021 | +$3,739 | **+$282** |
| PG | 18 | +$1,260 | +$750 | +$98 | +$2,107 | +$2,619 | -$512 |
| MSFT | 33 | +$9,990 | +$4,573 | +$237 | +$14,800 | +$15,831 | -$1,031 |
| KO | 12 | +$327 | +$1,223 | +$45 | +$1,594 | +$2,957 | -$1,363 |
| NVDA | 44 | +$13,409 | +$2,283 | +$484 | +$16,176 | +$17,927 | -$1,751 |
| AMZN | 43 | +$11,585 | +$121 | +$499 | +$12,205 | +$14,171 | -$1,966 |
| HD | 38 | +$6,349 | -$762 | +$348 | +$5,935 | +$8,708 | -$2,773 |
| GOOGL | 28 | +$13,229 | +$7,431 | +$288 | +$20,948 | +$25,157 | -$4,209 |
| INTC | 33 | +$2,152 | +$2,702 | +$166 | +$5,019 | +$10,113 | -$5,094 |
| AMD | 42 | +$25,970 | +$0 | +$707 | +$26,678 | +$45,939 | -$19,261 |
| META | 40 | +$17,182 | +$1,601 | +$1,947 | +$20,729 | +$46,025 | -$25,296 |
| **TOTAL** | **561** | **+$190,391** | **+$6,404** | **+$1,545** | **+$198,340** | **+$243,663** | **-$45,323** |

## Beat B&H: 8/18

MMM, IONQ, TSLA, IBM, AAPL, UNH, ABT, XOM

## Top 3 Annual Returns

| Symbol | Start Price | Capital | Combined P&L | Total Return | Annual Return |
|--------|------------|---------|-------------|-------------|--------------|
| MMM | $94.55 | $9,455 | +$12,599 | 133.2% | 23.1%/yr |
| IONQ | $6.19 | $619 | +$9,109 | 1,471.6% | 96.8%/yr |
| TSLA | $288.09 | $28,809 | +$20,457 | 71.0% | 14.1%/yr |

## Key Observations

- **Scavenger carries 96% of combined P&L** ($190K of $198K) — wheel is the workhorse
- **Raider adds $6.4K** but hurts volatile stocks (TSLA -$11.8K, UNH -$10K); strong on IBM/AAPL/GOOGL/MSFT
- **Chemist adds $1.5K** from 40 NUKED_ZONE credit spreads across 36 panic days (VIX≥30)
- **Wheel underperforms B&H on straight-line runners** (AMD, META, GOOGL) — covered calls cap upside
- **Wheel outperforms on mean-reverting/moderate-growth stocks** (MMM, IONQ, XOM, UNH)
- **Regime breakdown (4yr):** 73% RECLAMATION, 26% WASTELAND, 2% NUKED_ZONE (15 bars)

## Code State

- `schwab/run_backtest.py` — unified runner, `--role scavenger|raider|chemist|overseer`
- `schwab/backtest_scavenger.py` — wheel strategy with regime awareness
- `schwab/backtest_raider.py` — pullback-in-trend long
- `schwab/backtest_chemist.py` — credit put spread, NUKED_ZONE only
- `vault76/overseer.py` — regime classifier (RECLAMATION / WASTELAND / NUKED_ZONE)
- Git: `4c9ccc5` on `main`
