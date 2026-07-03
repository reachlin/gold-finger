"""
Real option-chain quotes from the Schwab API for Scavenger signals.

The Scavenger prices premiums with Black-Scholes on historical volatility,
which drifts from reality whenever IV diverges from HV — exactly the moments
premium selling is most interesting. This module re-quotes a SELL_PUT /
SELL_CALL signal against the live chain: nearest listed strike, expiration
closest to the target DTE, mid price, real IV and delta.

If the chain fetch fails (API down, no liquid contracts), the signal keeps
its model premium and is tagged quote_source="model" so downstream consumers
(LLM prompt, ledger reason) know which price they are looking at.
"""
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(__file__))

from strategy_params import SCAV_MIN_PREMIUM_PCT

MIN_DTE       = 21     # earliest expiration considered
MAX_DTE       = 45     # latest expiration considered
MIN_OPEN_INT  = 1      # skip strikes nobody holds — unquotable in practice


def fetch_chain_quote(client, symbol: str, option_type: str,
                      target_strike: float, target_dte: int,
                      min_dte: int = MIN_DTE, max_dte: int = MAX_DTE) -> dict | None:
    """
    Fetch the chain for symbol and return the contract nearest to
    (target_strike, target_dte):

      {"strike", "premium" (mid), "bid", "ask", "dte", "expiry",
       "iv" (fraction), "delta", "open_interest"}

    option_type: "PUT" or "CALL". Returns None when nothing usable is found.
    """
    try:
        kwargs = {}
        try:
            # Narrow the server-side window when the enum is importable
            from schwab.client import Client as _C
            kwargs["contract_type"] = (_C.Options.ContractType.CALL
                                       if option_type == "CALL"
                                       else _C.Options.ContractType.PUT)
        except Exception:
            pass
        resp = client.get_option_chain(
            symbol,
            include_underlying_quote=True,
            from_date=datetime.now() + timedelta(days=min_dte),
            to_date=datetime.now() + timedelta(days=max_dte),
            **kwargs,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return None

    exp_map = data.get("callExpDateMap" if option_type == "CALL"
                       else "putExpDateMap", {})
    if not exp_map:
        return None

    best = None
    for exp_key, strikes in exp_map.items():
        try:
            exp_str, dte = exp_key.split(":")[0], int(exp_key.split(":")[1])
        except (IndexError, ValueError):
            continue
        if not (min_dte <= dte <= max_dte):
            continue
        for strike_str, opts in strikes.items():
            opt    = opts[0]
            strike = float(strike_str)
            bid    = float(opt.get("bid", 0) or 0)
            ask    = float(opt.get("ask", 0) or 0)
            oi     = int(opt.get("openInterest", 0) or 0)
            if bid <= 0 or oi < MIN_OPEN_INT:
                continue
            # Rank by distance to target strike first, then to target DTE
            rank = (abs(strike - target_strike), abs(dte - target_dte))
            if best is None or rank < best[0]:
                best = (rank, {
                    "strike":        strike,
                    "premium":       round((bid + ask) / 2, 4),
                    "bid":           bid,
                    "ask":           ask,
                    "dte":           dte,
                    "expiry":        exp_str,
                    "iv":            float(opt.get("volatility", 0) or 0) / 100,
                    "delta":         float(opt.get("delta", 0) or 0),
                    "open_interest": oi,
                })
    return best[1] if best else None


def requote_signal(client, s: dict, target_dte: int | None = None) -> dict | None:
    """
    Replace a Scavenger signal's model premium with a real chain quote.

    Returns the signal (mutated in place) tagged quote_source="schwab_chain"
    or "model" on fallback. Returns None when the *real* premium falls below
    the SCAV_MIN_PREMIUM_PCT floor — the trade the model priced does not
    actually exist at that yield, so the signal is dropped.
    Non-option signals pass through untouched.
    """
    if s.get("signal") not in ("SELL_PUT", "SELL_CALL"):
        return s

    option_type = "CALL" if s["signal"] == "SELL_CALL" else "PUT"
    target_dte  = target_dte or int(s.get("dte", 30) or 30)
    q = fetch_chain_quote(client, s["symbol"], option_type,
                          target_strike=float(s.get("strike", 0) or 0),
                          target_dte=target_dte)

    if q is None or q["premium"] <= 0:
        s["quote_source"] = "model"
        return s

    close = float(s.get("close", 0) or 0)
    premium_pct = q["premium"] / close * 100 if close else 0.0
    if close and q["premium"] / close < SCAV_MIN_PREMIUM_PCT:
        return None   # real premium too thin — the modeled trade doesn't exist

    s.update({
        "strike":       q["strike"],
        "premium":      q["premium"],
        "premium_pct":  round(premium_pct, 2),
        "dte":          q["dte"],
        "expiry":       q["expiry"],
        "iv":           round(q["iv"] * 100, 1),      # store as % like hv
        "delta":        q["delta"],
        "bid":          q["bid"],
        "ask":          q["ask"],
        "quote_source": "schwab_chain",
    })
    if s.get("signal") == "SELL_PUT":
        s["max_loss"] = round((q["strike"] - q["premium"]) * 100, 2)
    return s
