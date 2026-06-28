"""
Vault 20 — Option Candidate Finder

Screens option chains via Schwab API (real-time quotes + real Greeks).
Surfaces the best covered call / CSP candidates for a given symbol.

Usage:
  python vault20/option_finder.py INTC covered-call [--min-dte 21] [--max-dte 60] [--top 8]
  python vault20/option_finder.py INTC csp          [--min-dte 21] [--max-dte 60] [--top 8]
  python vault20/option_finder.py --all             [scans all vault20 equity positions]
  python vault20/option_finder.py INTC covered-call --slack
"""
import argparse
import json
import os
import sys
from datetime import date, datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

VAULT20_DATA = os.path.join(os.path.dirname(__file__), "..", "data", "vault20_positions.json")
TOKEN_PATH   = os.path.join(os.path.dirname(__file__), "..", "schwab", "schwab_token.json")
RISK_FREE    = 0.05


# ---------------------------------------------------------------------------
# Schwab client
# ---------------------------------------------------------------------------

def _import_schwab_lib():
    """Import the installed schwab-py library.
    The local schwab/ project folder shadows the installed package, so we
    temporarily strip the project root from sys.path before importing.
    """
    project_root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
    saved = sys.path[:]
    sys.path = [p for p in sys.path if os.path.normpath(p) != project_root]

    # evict any locally cached schwab module
    evicted = {k: v for k, v in sys.modules.items()
               if k == "schwab" or k.startswith("schwab.")}
    for k in evicted:
        del sys.modules[k]
    try:
        import schwab
        return schwab
    finally:
        sys.path = saved
        # restore local schwab/* modules (not the top-level package)
        for k, v in evicted.items():
            if k != "schwab" and k not in sys.modules:
                sys.modules[k] = v


def _make_schwab_client():
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
    client_id     = os.environ.get("SCHWAB_CLIENT_ID")
    client_secret = os.environ.get("SCHWAB_CLIENT_SECRET")
    if not (client_id and client_secret and os.path.exists(TOKEN_PATH)):
        raise RuntimeError("Schwab credentials not found — check .env and schwab_token.json")
    schwab_lib = _import_schwab_lib()
    return schwab_lib.auth.client_from_token_file(TOKEN_PATH, client_id, client_secret)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_covered_call(strike: float, bid: float, stock_price: float, dte: int) -> dict:
    if dte <= 0 or stock_price <= 0:
        return {"monthly_yield_pct": 0.0, "otm_pct": 0.0,
                "annual_yield_pct": 0.0, "breakeven": stock_price}
    monthly_yield = (bid / stock_price) / (dte / 30) * 100 if bid else 0.0
    annual_yield  = (bid / stock_price) * (365 / dte) * 100 if bid else 0.0
    otm_pct       = (strike - stock_price) / stock_price * 100
    return {
        "monthly_yield_pct": round(monthly_yield, 3),
        "annual_yield_pct":  round(annual_yield, 2),
        "otm_pct":           round(otm_pct, 5),
        "breakeven":         round(stock_price - bid, 2),
    }


def score_csp(strike: float, bid: float, stock_price: float, dte: int) -> dict:
    if dte <= 0 or strike <= 0:
        return {"monthly_yield_pct": 0.0, "otm_pct": 0.0,
                "annual_yield_pct": 0.0, "breakeven": strike}
    monthly_yield = (bid / strike) / (dte / 30) * 100 if bid else 0.0
    annual_yield  = (bid / strike) * (365 / dte) * 100 if bid else 0.0
    otm_pct       = (stock_price - strike) / stock_price * 100
    return {
        "monthly_yield_pct": round(monthly_yield, 3),
        "annual_yield_pct":  round(annual_yield, 2),
        "otm_pct":           round(otm_pct, 5),
        "breakeven":         round(strike - bid, 2),
    }


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def filter_calls(df, stock_price: float, min_oi: int = 100, min_otm_pct: float = 1.0):
    import pandas as pd
    if df.empty:
        return df
    df = df[~df["inTheMoney"]].copy()
    df = df[df["openInterest"] >= min_oi]
    df = df[df["strike"] >= stock_price * (1 + min_otm_pct / 100)]
    return df.reset_index(drop=True)


def filter_puts(df, stock_price: float, min_oi: int = 100, min_otm_pct: float = 1.0):
    if df.empty:
        return df
    df = df[~df["inTheMoney"]].copy()
    df = df[df["openInterest"] >= min_oi]
    df = df[df["strike"] <= stock_price * (1 - min_otm_pct / 100)]
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Format / rank
# ---------------------------------------------------------------------------

def format_candidates(rows: list[dict], top: int = 10) -> list[dict]:
    return sorted(rows, key=lambda r: r["monthly_yield_pct"], reverse=True)[:top]


# ---------------------------------------------------------------------------
# Schwab option chain fetch
# ---------------------------------------------------------------------------

def _iv_rank(symbol: str, current_iv: float) -> float | None:
    """
    IV rank: where current IV sits in its 52-week range (0–100).
    Uses Yang-Zhang HV over rolling 30-day windows as IV proxy.
    Returns None if insufficient data.
    """
    try:
        import contextlib, io, yfinance as yf
        with contextlib.redirect_stderr(io.StringIO()):
            df = yf.download(symbol, period="1y", interval="1d",
                             progress=False, auto_adjust=True,
                             multi_level_index=False)
        if len(df) < 60:
            return None

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "schwab"))
        from options_pricer import yang_zhang_vol

        # Rolling 30-day YZ vol series
        hv_series = [
            yang_zhang_vol(df.iloc[max(0, i-30):i], window=20)
            for i in range(30, len(df) + 1)
        ]
        hv_series = [v for v in hv_series if v > 0]
        if not hv_series:
            return None
        lo, hi = min(hv_series), max(hv_series)
        if hi <= lo:
            return None
        return round((current_iv - lo) / (hi - lo) * 100, 1)
    except Exception:
        return None


def _parse_schwab_exp_map(exp_map: dict, strategy: str,
                          stock_price: float, min_oi: int,
                          min_otm_pct: float,
                          target_delta: float | None = None,
                          delta_range: float = 0.10) -> list[dict]:
    """Parse Schwab callExpDateMap / putExpDateMap into candidate rows.
    If target_delta is set, only keeps options with |delta| in
    [target_delta - delta_range, target_delta + delta_range].
    """
    score_fn  = score_covered_call if strategy == "covered-call" else score_csp
    is_call   = strategy == "covered-call"
    candidates = []

    for exp_key, strikes in exp_map.items():
        exp_str = exp_key.split(":")[0]
        dte     = int(exp_key.split(":")[1])

        for strike_str, opts in strikes.items():
            opt    = opts[0]
            strike = float(strike_str)
            bid    = float(opt.get("bid", 0) or 0)
            ask    = float(opt.get("ask", 0) or 0)
            oi     = int(opt.get("openInterest", 0) or 0)
            vol    = int(opt.get("totalVolume", 0) or 0)
            # Schwab returns IV as a percentage (e.g. 94.1 means 94.1%)
            iv     = float(opt.get("volatility", 0) or 0) / 100
            delta  = float(opt.get("delta", 0) or 0)
            gamma  = float(opt.get("gamma", 0) or 0)
            theta  = float(opt.get("theta", 0) or 0)
            vega   = float(opt.get("vega", 0) or 0)
            itm    = bool(opt.get("inTheMoney", False))

            if itm or oi < min_oi or bid <= 0:
                continue
            if is_call and strike < stock_price * (1 + min_otm_pct / 100):
                continue
            if not is_call and strike > stock_price * (1 - min_otm_pct / 100):
                continue

            # Delta filter: use |delta| since puts have negative delta
            abs_delta = abs(delta)
            if target_delta is not None:
                if not (target_delta - delta_range <= abs_delta <= target_delta + delta_range):
                    continue

            scores = score_fn(strike, bid, stock_price, dte)
            candidates.append({
                "expiry":            exp_str,
                "strike":            strike,
                "bid":               round(bid, 2),
                "ask":               round(ask, 2),
                "dte":               dte,
                "openInterest":      oi,
                "volume":            vol,
                "iv":                round(iv, 4),
                "delta":             round(delta, 3),
                "gamma":             round(gamma, 4),
                "theta":             round(theta, 3),
                "vega":              round(vega, 3),
                "delta_diff":        round(abs(abs_delta - (target_delta or abs_delta)), 4),
                **scores,
            })

    return candidates


def _fetch_chain(client, symbol: str, strategy: str,
                 min_dte: int, max_dte: int,
                 min_oi: int, min_otm_pct: float,
                 target_delta: float | None = None,
                 delta_range: float = 0.10) -> tuple[float, list[dict], float | None]:
    schwab_lib = _import_schwab_lib()

    # Re-import enum from the same schwab_lib instance to avoid type-identity mismatch
    contract_type = (schwab_lib.client.Client.Options.ContractType.CALL
                     if strategy == "covered-call"
                     else schwab_lib.client.Client.Options.ContractType.PUT)

    # Build a fresh client using the correctly-imported library
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
    client = schwab_lib.auth.client_from_token_file(
        TOKEN_PATH,
        os.environ["SCHWAB_CLIENT_ID"],
        os.environ["SCHWAB_CLIENT_SECRET"],
    )

    from_date = datetime.now() + timedelta(days=min_dte)
    to_date   = datetime.now() + timedelta(days=max_dte)

    resp = client.get_option_chain(
        symbol,
        contract_type=contract_type,
        include_underlying_quote=True,
        from_date=from_date,
        to_date=to_date,
    )
    resp.raise_for_status()
    data = resp.json()

    stock_price = float(data["underlying"]["last"])
    exp_map_key = "callExpDateMap" if strategy == "covered-call" else "putExpDateMap"

    # Use the chain's average IV for IV rank (median across near-ATM options)
    all_ivs = []
    for strikes in data.get(exp_map_key, {}).values():
        for opts in strikes.values():
            v = float(opts[0].get("volatility", 0) or 0)
            if v > 0:
                all_ivs.append(v / 100)
    avg_iv   = float(np.median(all_ivs)) if all_ivs else 0.0
    iv_rank  = _iv_rank(symbol, avg_iv)

    candidates = _parse_schwab_exp_map(
        data.get(exp_map_key, {}), strategy, stock_price, min_oi, min_otm_pct,
        target_delta=target_delta, delta_range=delta_range,
    )
    return stock_price, candidates, iv_rank


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def _strategy_label(strategy: str) -> str:
    return "Covered Call" if strategy == "covered-call" else "Cash-Secured Put"


def _print_candidates(symbol: str, strategy: str, stock_price: float,
                      candidates: list[dict], shares: int = 0,
                      iv_rank: float | None = None):
    label = _strategy_label(strategy)
    now   = datetime.now().strftime("%Y-%m-%d %H:%M")
    sep   = "=" * 92

    print(f"\n{sep}")
    print(f"  {symbol} — {label} Candidates  (price: ${stock_price:.2f})  {now}")
    if shares:
        print(f"  You hold {shares} shares → up to {shares // 100} contracts")
    if iv_rank is not None:
        iv_signal = ("🔥 HIGH — good time to sell premium"   if iv_rank >= 50 else
                     "⚠ LOW  — thin premium, consider waiting" if iv_rank < 25 else
                     "  MID  — fair premium")
        print(f"  IV Rank (52w): {iv_rank:.0f}/100  {iv_signal}")
    print(sep)

    if not candidates:
        print("  No candidates found matching your criteria.")
        print(sep)
        return

    print(f"  {'Expiry':<12} {'Strike':>7} {'Bid':>6} {'Ask':>6} "
          f"{'DTE':>4}  {'Yield/mo':>9}  {'OTM%':>6}  "
          f"{'Delta':>6}  {'Theta':>6}  {'Vega':>6}  {'IV':>6}  {'OI':>7}")
    print(f"  {'-'*88}")
    for c in candidates:
        print(
            f"  {c['expiry']:<12} "
            f"{c['strike']:>7.2f} "
            f"{c['bid']:>6.2f} "
            f"{c['ask']:>6.2f} "
            f"{c['dte']:>4}  "
            f"{c['monthly_yield_pct']:>8.2f}%  "
            f"{c['otm_pct']:>5.1f}%  "
            f"{c['delta']:>6.3f}  "
            f"{c['theta']:>6.3f}  "
            f"{c['vega']:>6.3f}  "
            f"{c['iv']*100:>5.1f}%  "
            f"{c['openInterest']:>7,}"
        )
    print(sep)


def _slack_candidates(symbol: str, strategy: str, stock_price: float,
                      candidates: list[dict], top: int = 5) -> str:
    label = _strategy_label(strategy)
    lines = [f"*{symbol} — {label} Candidates* (${stock_price:.2f})"]
    if not candidates:
        lines.append("No candidates found.")
        return "\n".join(lines)

    lines.append("`Expiry       Strike   Bid  DTE  Yield/mo  OTM%  Delta  Theta`")
    for c in candidates[:top]:
        lines.append(
            f"`{c['expiry']:<12} {c['strike']:>6.2f} "
            f"{c['bid']:>5.2f} {c['dte']:>4} "
            f"{c['monthly_yield_pct']:>7.2f}% "
            f"{c['otm_pct']:>4.1f}% "
            f"{c['delta']:>6.3f} "
            f"{c['theta']:>6.3f}`"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Vault20 positions helper
# ---------------------------------------------------------------------------

def _load_vault20_positions() -> list[dict]:
    if not os.path.exists(VAULT20_DATA):
        return []
    return json.loads(open(VAULT20_DATA).read()).get("open", [])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

    parser = argparse.ArgumentParser(
        prog="option_finder",
        description="Find covered call / CSP candidates via Schwab API (real Greeks)")
    parser.add_argument("symbol",   nargs="?",
                        help="Ticker symbol (e.g. INTC). Omit with --all.")
    parser.add_argument("strategy", nargs="?", default="covered-call",
                        choices=["covered-call", "cc", "csp", "put"])
    parser.add_argument("--min-dte",      type=int,   default=21)
    parser.add_argument("--max-dte",      type=int,   default=60)
    parser.add_argument("--min-oi",       type=int,   default=100)
    parser.add_argument("--min-otm",      type=float, default=1.0,
                        help="Min OTM%% (default 1%%)")
    parser.add_argument("--target-delta", type=float, default=None,
                        help="Target |delta| e.g. 0.30 (industry standard). "
                             "When set, overrides --min-otm as primary filter.")
    parser.add_argument("--delta-range",  type=float, default=0.10,
                        help="±delta tolerance around --target-delta (default 0.10)")
    parser.add_argument("--top",          type=int,   default=8)
    parser.add_argument("--slack",        action="store_true")
    parser.add_argument("--all",          action="store_true",
                        help="Scan all vault20 equity positions")
    args     = parser.parse_args()
    strategy = "csp" if args.strategy in ("csp", "put") else "covered-call"

    try:
        client = _make_schwab_client()
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if args.all:
        positions = _load_vault20_positions()
        equities  = [(p["symbol"], int(p["shares"]))
                     for p in positions
                     if "_" not in p["symbol"] and p["shares"] > 0]
        if not equities:
            print("No equity positions found in vault20.")
            return
    elif args.symbol:
        positions = _load_vault20_positions()
        held      = next((p for p in positions if p["symbol"] == args.symbol.upper()), None)
        shares    = int(held["shares"]) if held and held["shares"] > 0 else 0
        equities  = [(args.symbol.upper(), shares)]
    else:
        parser.print_help()
        return

    all_slack = []
    for symbol, shares in equities:
        print(f"\nFetching {symbol} option chain from Schwab…", end=" ", flush=True)
        try:
            stock_price, candidates, iv_rank = _fetch_chain(
                client, symbol, strategy,
                min_dte=args.min_dte, max_dte=args.max_dte,
                min_oi=args.min_oi, min_otm_pct=args.min_otm,
                target_delta=args.target_delta, delta_range=args.delta_range,
            )
        except Exception as exc:
            print(f"error: {exc}")
            continue

        ranked = format_candidates(candidates, top=args.top)
        print(f"done — {len(ranked)} candidates  (IV rank: {iv_rank:.0f}/100)" if iv_rank is not None else f"done — {len(ranked)} candidates")
        _print_candidates(symbol, strategy, stock_price, ranked,
                          shares=shares, iv_rank=iv_rank)

        if args.slack:
            all_slack.append(_slack_candidates(symbol, strategy, stock_price, ranked))

    if args.slack and all_slack:
        from notify_slack import send
        send("\n\n".join(all_slack))
        print("  → Sent to Slack.")


if __name__ == "__main__":
    main()
