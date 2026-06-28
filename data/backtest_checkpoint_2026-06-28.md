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

## Model & Strategy Parameters

### Overseer — Regime Classifier (`vault76/overseer.py`)

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `VIX_NUKED` | 30.0 | VIX ≥ 30 → NUKED_ZONE (overrides SPY) |
| `MIN_BARS` | 60 | Minimum SPY bars needed to classify |
| RECLAMATION | SPY above rising EMA50, VIX < 30 | Bull market |
| WASTELAND | SPY below EMA50 or EMA50 falling, VIX < 30 | Bear/sideways |
| NUKED_ZONE | VIX ≥ 30 | Panic / extreme fear |
| Active roles | RECLAMATION: Raider+Scavenger · WASTELAND: Scavenger+Raider · NUKED_ZONE: Chemist | |

---

### Scavenger — Cash-Secured Put → Covered Call (`vault76/armory/scavenger.py`)

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `OTM_PUT_PCT` | 0.05 | Sell put 5% below current price |
| `OTM_CALL_PCT` | 0.08 | Sell call 8% above cost basis |
| `SELL_DTE` | 30 | Days to expiration (both legs) |
| `MIN_HV` | 0.20 | Min historical vol — thin premium below this |
| `MIN_PREMIUM_PCT` | 0.005 | Min premium as % of stock price (0.5%) |
| `ADX_TREND_MAX` | 20 | Block put-sell if stock is trending (Raider's turf) |
| `RSI_NEUTRAL_LO` | 35 | RSI floor for put-selling (falling knife guard) |
| `RSI_NEUTRAL_HI` | 65 | RSI ceiling for put-selling (overbought guard) |
| `UNDERWATER_MAX` | 0.10 | Block call-sell if price > 10% below cost basis |
| Exit target | 35–65% of premium collected | Buy back early if premium decays to this range |
| Backtest shares | 100 | 1 contract = 100 shares |
| `MIN_HISTORY` | 60 bars | Warm-up before scanning |
| `RISK_FREE` | 0.05 | Black-Scholes risk-free rate |

**Active in:** RECLAMATION, WASTELAND

---

### Raider — Pullback-in-Trend Long (`vault76/armory/raider.py`, `schwab/trend_scanner.py`)

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `RSI_PULLBACK_HI` | 47 | RSI must dip below this (real dip, not just noise) |
| `RSI_PULLBACK_LO` | 28 | RSI floor — don't enter a freefall |
| `ADX_MIN` | 20 | Confirmed trend strength required |
| `TARGET_PCT` | 0.20 | Take profit at +20% |
| `STOP_PCT` | 0.08 | Stop loss at -8% |
| Risk/reward | 2.5:1 | 20% target / 8% stop |
| `MAX_HOLD` | 60 bars | Safety valve (~3 months) |
| Primary exit | EMA20 < EMA50 | Trend-end exit (overrides time stop) |
| Backtest shares | 100 | |

**Active in:** RECLAMATION, WASTELAND

---

### Chemist — Credit Put Spread (`vault76/armory/chemist.py`, `schwab/backtest_chemist.py`)

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `SHORT_PUT_OTM` | 0.08 | Sell put 8% OTM (collect premium) |
| `LONG_PUT_OTM` | 0.18 | Buy put 18% OTM (cap risk) |
| Spread width | 10% | Distance between strikes |
| `SELL_DTE` | 30 | Trading days to expiration |
| `MIN_HV` | 0.20 | Min historical vol — usually 35%+ in NUKED_ZONE |
| `RSI_FREEFALL` | 20 | Block if RSI < 20 (skip freefalls) |
| `UNDERWATER_MAX` | 0.30 | Block if stock > 30% below EMA50 |
| `PROFIT_TARGET` | 0.70 | Buy back when 70% of max profit reached |
| `LOSS_LIMIT` | 2.00 | Cut when spread reaches 200% of credit received |
| `RISK_FREE` | 0.05 | Black-Scholes risk-free rate |
| Backtest contracts | 1 (100 shares) | |

**Active in:** NUKED_ZONE only

---

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
