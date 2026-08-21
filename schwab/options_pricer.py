"""
Black-Scholes option pricing, Greeks, and volatility utilities.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq

RISK_FREE_RATE  = 0.05
PUT_DTE         = 21          # days to expiry for paper puts
PUT_MAX_HOLD    = 10          # max days to hold a put
PUT_PROFIT_MULT = 2.0         # exit put when value doubles
PUT_CONTRACTS   = 1           # 1 contract = 100 shares
PUT_BUDGET      = 300         # max USD to spend on put premium
NEW_HIGH_PCT    = 0.03        # exit put if stock rises 3% above entry


def _safe_sigma(sigma: float) -> float:
    """Coerce sigma to a safe positive float — guards against 0, nan, inf, subnormals."""
    s = float(sigma)
    return s if (s > 1e-8 and np.isfinite(s)) else 0.0


def black_scholes_put(S: float, K: float, T: float,
                      r: float, sigma: float) -> float:
    """European put price via Black-Scholes."""
    if T <= 0:
        return float(max(K - S, 0.0))
    sigma = _safe_sigma(sigma)
    if sigma == 0.0:
        return float(max(K * np.exp(-r * T) - S, 0.0))
    if K <= 0 or S <= 0:
        return float(max(K - S, 0.0))
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return float(max(price, 0.0))


def black_scholes_call(S: float, K: float, T: float,
                       r: float, sigma: float) -> float:
    """European call price via Black-Scholes."""
    if T <= 0:
        return float(max(S - K, 0.0))
    sigma = _safe_sigma(sigma)
    if sigma == 0.0:
        return float(max(S - K * np.exp(-r * T), 0.0))
    if K <= 0 or S <= 0:
        return float(max(S - K, 0.0))
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return float(max(price, 0.0))


def historical_vol(prices: pd.Series, window: int = 20) -> float:
    """Annualized historical volatility. Returns 0.0 if insufficient data."""
    if len(prices) < window + 1:
        return 0.0
    log_ret = np.log(prices / prices.shift(1)).dropna()
    vol = log_ret.rolling(window).std().iloc[-1]
    return float(vol * np.sqrt(252)) if not np.isnan(vol) else 0.0


def yang_zhang_vol(df: pd.DataFrame, window: int = 20,
                   trading_periods: int = 252) -> float:
    """
    Yang-Zhang volatility estimator (annualized).
    More accurate than close-only HV — accounts for overnight gaps and intraday range.
    Requires OHLC columns: open, high, low, close.
    Returns 0.0 if insufficient data.

    Formula combines three components with weight k:
      σ²_YZ = σ²_open + k·σ²_close + (1-k)·σ²_RS
    where σ²_RS is the Rogers-Satchell estimator (no overnight gap assumption).
    """
    if len(df) < window + 1:
        return 0.0
    required = {"open", "high", "low", "close"}
    if not required.issubset(df.columns):
        return 0.0

    log_ho = np.log(df["high"]  / df["open"])
    log_lo = np.log(df["low"]   / df["open"])
    log_co = np.log(df["close"] / df["open"])
    log_oc = np.log(df["open"]  / df["close"].shift(1))
    log_cc = np.log(df["close"] / df["close"].shift(1))

    # Rogers-Satchell: intraday component (no overnight gap)
    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)

    open_var  = (log_oc ** 2).rolling(window).sum() / (window - 1)
    close_var = (log_cc ** 2).rolling(window).sum() / (window - 1)
    rs_var    = rs.rolling(window).mean()

    k      = 0.34 / (1.34 + (window + 1) / (window - 1))
    yz_var = open_var + k * close_var + (1 - k) * rs_var

    last = yz_var.dropna()
    if last.empty:
        return 0.0
    return float(np.sqrt(last.iloc[-1] * trading_periods))


def implied_vol(market_price: float, S: float, K: float, T: float, r: float,
                option_type: str = "call") -> float | None:
    """
    Back-solve implied volatility from a market price using Brent's method.
    Returns None if the price implies no real solution (e.g. zero/negative price).
    option_type: 'call' or 'put'
    """
    if market_price <= 0 or T <= 0:
        return None

    pricer = black_scholes_call if option_type == "call" else black_scholes_put

    def objective(sigma):
        return pricer(S, K, T, r, sigma) - market_price

    try:
        # Brentq guarantees convergence when the function changes sign on [lo, hi]
        return float(brentq(objective, 1e-4, 10.0, xtol=1e-6, maxiter=200))
    except ValueError:
        return None


def atm_strike(price: float, step: float = 5.0) -> float:
    """Round price to nearest strike step."""
    return round(round(price / step) * step, 2)


def adaptive_profit_target(entry_iv: float, entry_dte: int,
                           target_min: float | None = None,
                           target_max: float | None = None) -> float:
    """
    Self-adjusting early-exit threshold for short options, in [35%, 65%]
    of entry premium (bounds from strategy_params):
    - Higher IV  → higher target (fat premium is worth waiting for more decay)
    - Longer DTE → higher target (more time to collect full theta decay)

    Calibration anchors:
      iv=0.20, dte=15 → ~35%  (thin premium, exit fast)
      iv=0.60, dte=45 → ~50%  (standard tastyworks rule)
      iv=1.00, dte=75 → ~65%  (high-IV, long-dated — hold for bigger capture)
    """
    if target_min is None or target_max is None:
        from strategy_params import SCAV_PROFIT_TARGET_MIN, SCAV_PROFIT_TARGET_MAX
        target_min = SCAV_PROFIT_TARGET_MIN if target_min is None else target_min
        target_max = SCAV_PROFIT_TARGET_MAX if target_max is None else target_max
    iv_factor  = min(max((entry_iv - 0.20) / 0.80, 0.0), 1.0)
    dte_factor = min(max((entry_dte - 15)  / 60,   0.0), 1.0)
    score      = 0.6 * iv_factor + 0.4 * dte_factor
    return target_min + (target_max - target_min) * score


def should_take_early_profit(entry_prem: float, mark: float | None,
                             days_held: int,
                             min_profit: float | None = None,
                             max_days: int | None = None) -> bool:
    """
    v1 dynamic early take-profit for a short option: bank a FAST winner instead
    of waiting out the last of the decay.

    Rationale: a quick mark-to-market gain on a short premium position is mostly
    IV crush — once the volatility edge is realized, the remaining theta is slow
    and low-yield while the same tail/assignment risk and collateral lock-up
    persist. So harvest and redeploy. "Reached profit fast" is a robust proxy
    for "IV dropped" and needs no extra data, which is why v1 keys off it.

    Trigger: profit captured so far >= min_profit AND days_held <= max_days.

    Thresholds are fixed constants for now (from strategy_params). The next
    iteration makes them market-aware (VIX regime) and stock-aware (per-name IV
    rank) — that logic belongs HERE so every caller upgrades at once. This never
    lowers the exit bar below the resting GTC floor; callers use it only to
    TIGHTEN (close sooner), never to hold longer.
    """
    if min_profit is None or max_days is None:
        from strategy_params import EARLY_TP_MIN_PROFIT, EARLY_TP_MAX_DAYS
        min_profit = EARLY_TP_MIN_PROFIT if min_profit is None else min_profit
        max_days   = EARLY_TP_MAX_DAYS if max_days is None else max_days
    if not entry_prem or entry_prem <= 0 or mark is None or mark < 0:
        return False
    profit = (entry_prem - mark) / entry_prem
    return profit >= min_profit and days_held <= max_days


def simulate_put_trade(future_df: pd.DataFrame, entry_S: float, K: float,
                       sigma: float, r: float = RISK_FREE_RATE,
                       dte: int = PUT_DTE, max_hold: int = PUT_MAX_HOLD,
                       profit_target_mult: float = PUT_PROFIT_MULT,
                       new_high_pct: float = NEW_HIGH_PCT) -> dict:
    """
    Simulate holding one put contract through future price bars.

    Entry: buy 1 ATM put at BS price on bar 0.
    Exit conditions (checked each bar):
      - new_high:  stock closes above entry_S * (1 + new_high_pct)
      - target:    put value >= entry_price * profit_target_mult
      - timeout:   held max_hold days

    Returns dict with pnl_dollar (for 1 contract = 100 shares).
    """
    entry_put = black_scholes_put(entry_S, K, T=dte/365, r=r, sigma=sigma)
    if entry_put <= 0:
        return {"exit_reason": "worthless", "hold_days": 0,
                "entry_put_price": 0.0, "exit_put_price": 0.0,
                "pnl_dollar": 0.0, "pnl_pct": 0.0}

    for i, (_, row) in enumerate(future_df.iterrows()):
        days_remaining = max(dte - i - 1, 0)
        T_remaining    = days_remaining / 365
        current_put    = black_scholes_put(row["close"], K, T_remaining, r, sigma)

        exit_reason = None
        exit_put    = current_put

        if row["high"] >= entry_S * (1 + new_high_pct):
            exit_reason = "new_high"
        elif current_put >= entry_put * profit_target_mult:
            exit_reason = "target"
        elif i >= max_hold - 1:
            exit_reason = "timeout"

        if exit_reason:
            pnl_dollar = (exit_put - entry_put) * 100
            pnl_pct    = (exit_put - entry_put) / entry_put * 100 if entry_put > 0 else 0.0
            return {
                "exit_reason":     exit_reason,
                "hold_days":       i + 1,
                "entry_put_price": round(entry_put, 4),
                "exit_put_price":  round(exit_put, 4),
                "pnl_dollar":      round(pnl_dollar, 2),
                "pnl_pct":         round(pnl_pct, 2),
            }

    # Fell off end of data
    exit_put   = black_scholes_put(future_df["close"].iloc[-1], K, 0, r, sigma)
    pnl_dollar = (exit_put - entry_put) * 100
    return {
        "exit_reason":     "timeout",
        "hold_days":       len(future_df),
        "entry_put_price": round(entry_put, 4),
        "exit_put_price":  round(exit_put, 4),
        "pnl_dollar":      round(pnl_dollar, 2),
        "pnl_pct":         round((exit_put - entry_put) / entry_put * 100, 2),
    }
