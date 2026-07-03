"""
Vault 76 — strategy parameter config.

Edit these values to tune signal generation, regime gates, and risk filters.

NOTE: not every module imports from here yet. The FAST_RISKOFF_* and
SCAV_PROFIT_TARGET_* values are live (live_scanner, backtest_scavenger,
options_pricer). The SCAV_* and OVERSEER_* blocks below currently document
the same values that are hardcoded in vault76/armory/scavenger.py and
vault76/overseer.py — if you tune one, change both until those modules are
wired to import from here.

Units and rationale are documented inline. When changing a number, also update
the comment — future-you will thank present-you.
"""

# ---------------------------------------------------------------------------
# Scavenger — cash-secured put / covered call parameters
# ---------------------------------------------------------------------------

# How far OTM to sell the put, as a fraction of current price.
# 0.05 = 5% OTM → strike = close × 0.95
# Lower → more premium, but higher assignment risk.
# Higher → safer, but premium too thin to be worth the collateral.
SCAV_OTM_PUT_PCT = 0.05

# Covered call OTM fraction in a sideways/low-ADX stock.
# 0.08 = 8% OTM above cost basis. Maximize income when stock isn't running.
SCAV_OTM_CALL_PCT = 0.08

# Covered call OTM fraction in a trending stock (ADX ≥ SCAV_ADX_CALL_WIDE).
# 0.13 = 13% OTM — give the position more room before getting called away.
SCAV_OTM_CALL_PCT_RECLAMATION = 0.13

# Days to expiration for both put and call legs.
# 30 DTE is the tastyworks/TastyLive standard: captures the steepest theta decay
# curve while leaving enough premium to be worth the trade.
SCAV_SELL_DTE = 30

# Minimum historical volatility for premium to be worth selling.
# Below 20% annualized, the BS price of a 5% OTM put is negligible.
SCAV_MIN_HV = 0.20

# Minimum put/call premium as a fraction of stock price.
# 0.5% floor filters out edge cases where IV is fine but the specific
# strike/DTE combo produces a penny premium.
SCAV_MIN_PREMIUM_PCT = 0.005

# ADX threshold above which a stock is "trending" — route to Raider, not Scavenger.
# Below this, the stock is ranging/sideways → ideal for premium selling.
# ADX 20 is the standard Wilder threshold for a trend.
SCAV_ADX_TREND_MAX = 20

# ADX threshold at which to widen the covered call strike (see OTM_CALL_PCT_RECLAMATION).
SCAV_ADX_CALL_WIDE = 25

# ADX threshold above which we skip the covered call entirely (stock is running hard).
SCAV_ADX_CALL_BLOCK = 35

# RSI range for put-selling. Outside this range → price is not stable.
# Below 35: stock may be in a falling-knife move — put assignment risk is high.
# Above 65: overbought — mean reversion could hurt the position.
SCAV_RSI_NEUTRAL_LO = 35
SCAV_RSI_NEUTRAL_HI = 65

# Max % underwater before we stop selling covered calls.
# If the stock is >10% below cost basis, selling a call now locks in the loss.
# Better to wait for a recovery before capping upside.
SCAV_UNDERWATER_MAX = 0.10


# ---------------------------------------------------------------------------
# Overseer — market regime thresholds
# ---------------------------------------------------------------------------

# VIX level above which the entire market is in "Nuked Zone".
# At VIX ≥ 30, implied volatility is deceptively high and assignment risk is
# catastrophic — all premium-selling is suspended.
OVERSEER_VIX_NUKED = 30.0

# Per-stock ADX cutoff for routing: ≥ this → Raider (trend-following);
# < this → Scavenger (premium income).
OVERSEER_ADX_RUNNER = 28


# ---------------------------------------------------------------------------
# FinRL fast risk-off filter (#1)
#
# Of the three FinRL-Trading enhancements evaluated for the wheel strategy,
# this is the ONLY one that survived backtesting. #2 (adaptive position
# rotation) and #3 (residual momentum, documented below) were both tried and
# rejected — neither improved wheel P&L in the walk-forward backtest.
#
# If SPY drops more than FAST_RISKOFF_DROP over FAST_RISKOFF_LOOKBACK trading
# days, all new SELL_PUT entries are suppressed for FAST_RISKOFF_COOLDOWN days.
#
# Rationale (from FinRL-Trading adaptive rotation engine):
#   A sudden 3%+ 3-day SPY drop signals systemic stress — correlations spike,
#   IV rises sharply, and puts are far more likely to be assigned at a loss.
#   A 10-day cooling-off period lets volatility settle before re-entering.
#
# Tune these if you want:
#   - More sensitive: lower DROP threshold (e.g. -0.02) or longer LOOKBACK (e.g. 5)
#   - Less sensitive: higher DROP threshold (e.g. -0.05) or shorter COOLDOWN (e.g. 5)
# ---------------------------------------------------------------------------

FAST_RISKOFF_DROP     = -0.03   # SPY 3-day return that triggers fast risk-off (-3%)
FAST_RISKOFF_LOOKBACK = 3       # rolling window of trading days to measure the drop
FAST_RISKOFF_COOLDOWN = 10      # trading days to suppress new SELL_PUT after trigger


# ---------------------------------------------------------------------------
# FinRL residual momentum filter (#3) — TRIED AND REJECTED, kept for reference
#
# Idea: before selling a put, require the stock's beta-adjusted return vs SPY
# to be positive over the last RESID_MOM_WINDOW trading days.
#   Residual momentum = Σ(stock_daily_ret - beta × spy_daily_ret) over window
#
# STATUS (2026-07): implemented experimentally alongside fast risk-off (#1),
# but the walk-forward wheel backtest did NOT support it — the filter cut the
# number of put entries without improving P&L or reducing assignments, so it
# was removed from the pipeline. Only fast risk-off (#1) survived backtesting.
# These constants stay so the experiment can be re-run if conditions change;
# nothing imports them.
# ---------------------------------------------------------------------------

RESID_MOM_WINDOW  = 20      # lookback window in trading days
RESID_MOM_MIN_PCT = 0.0     # minimum residual return to allow a put entry (0 = any positive)


# ---------------------------------------------------------------------------
# Adaptive profit target for early exit
# ---------------------------------------------------------------------------

# Scavenger exits a short put/call early when its mark-to-market value falls
# to TARGET_MIN–TARGET_MAX of the entry premium (50% rule in tastyworks parlance).
# The exact threshold scales with entry IV and DTE — see
# options_pricer.adaptive_profit_target() (used by both backtest_scavenger.py
# and the live ledger processor in options_ledger.py).
SCAV_PROFIT_TARGET_MIN = 0.35   # exit at 35% of premium captured (thin/short trades)
SCAV_PROFIT_TARGET_MAX = 0.65   # exit at 65% of premium captured (fat/long trades)
