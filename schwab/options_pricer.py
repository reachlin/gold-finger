"""
Black-Scholes put option pricing utilities for backtesting.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

RISK_FREE_RATE  = 0.05
PUT_DTE         = 21          # days to expiry for paper puts
PUT_MAX_HOLD    = 10          # max days to hold a put
PUT_PROFIT_MULT = 2.0         # exit put when value doubles
PUT_CONTRACTS   = 1           # 1 contract = 100 shares
PUT_BUDGET      = 300         # max USD to spend on put premium
NEW_HIGH_PCT    = 0.03        # exit put if stock rises 3% above entry


def black_scholes_put(S: float, K: float, T: float,
                      r: float, sigma: float) -> float:
    """European put price via Black-Scholes."""
    if T <= 0:
        return float(max(K - S, 0.0))
    if sigma <= 0:
        return float(max(K - S * np.exp(-r * T), 0.0))
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return float(max(price, 0.0))


def black_scholes_call(S: float, K: float, T: float,
                       r: float, sigma: float) -> float:
    """European call price via Black-Scholes."""
    if T <= 0:
        return float(max(S - K, 0.0))
    if sigma <= 0:
        return float(max(S * np.exp(r * T) - K, 0.0))
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


def atm_strike(price: float, step: float = 5.0) -> float:
    """Round price to nearest strike step."""
    return round(round(price / step) * step, 2)


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
