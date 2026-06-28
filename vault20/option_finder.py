"""
Vault 20 — Option Candidate Finder

Screens option chains and surfaces the best covered call / CSP candidates
for a given symbol. Uses yfinance for option chain data; actual orders
go through Schwab.

Usage:
  python vault20/option_finder.py INTC covered-call [--min-dte 21] [--max-dte 60] [--top 8]
  python vault20/option_finder.py INTC csp          [--min-dte 21] [--max-dte 60] [--top 8]
  python vault20/option_finder.py --all             [uses open vault20 positions]
"""
import argparse
import contextlib
import io
import json
import os
import sys
from datetime import date, datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

VAULT20_DATA = os.path.join(os.path.dirname(__file__), "..", "data", "vault20_positions.json")


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_covered_call(strike: float, bid: float, stock_price: float, dte: int) -> dict:
    """Score a covered call candidate."""
    if dte <= 0 or stock_price <= 0:
        return {"monthly_yield_pct": 0.0, "otm_pct": 0.0,
                "annual_yield_pct": 0.0, "breakeven": stock_price}
    monthly_yield = (bid / stock_price) / (dte / 30) * 100 if bid else 0.0
    annual_yield  = (bid / stock_price) * (365 / dte) * 100 if bid else 0.0
    otm_pct       = (strike - stock_price) / stock_price * 100
    breakeven     = stock_price - bid
    return {
        "monthly_yield_pct": round(monthly_yield, 3),
        "annual_yield_pct":  round(annual_yield, 2),
        "otm_pct":           round(otm_pct, 5),
        "breakeven":         round(breakeven, 2),
    }


def score_csp(strike: float, bid: float, stock_price: float, dte: int) -> dict:
    """Score a cash-secured put candidate. Yield is on collateral (strike × 100)."""
    if dte <= 0 or strike <= 0:
        return {"monthly_yield_pct": 0.0, "otm_pct": 0.0,
                "annual_yield_pct": 0.0, "breakeven": strike}
    monthly_yield = (bid / strike) / (dte / 30) * 100 if bid else 0.0
    annual_yield  = (bid / strike) * (365 / dte) * 100 if bid else 0.0
    otm_pct       = (stock_price - strike) / stock_price * 100
    breakeven     = strike - bid
    return {
        "monthly_yield_pct": round(monthly_yield, 3),
        "annual_yield_pct":  round(annual_yield, 2),
        "otm_pct":           round(otm_pct, 5),
        "breakeven":         round(breakeven, 2),
    }


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def filter_calls(df, stock_price: float, min_oi: int = 100,
                 min_otm_pct: float = 1.0):
    if df.empty:
        return df
    df = df[~df["inTheMoney"]].copy()
    df = df[df["openInterest"] >= min_oi]
    df = df[df["strike"] >= stock_price * (1 + min_otm_pct / 100)]
    return df.reset_index(drop=True)


def filter_puts(df, stock_price: float, min_oi: int = 100,
                min_otm_pct: float = 1.0):
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
    """Sort by monthly yield descending, return top N."""
    return sorted(rows, key=lambda r: r["monthly_yield_pct"], reverse=True)[:top]


# ---------------------------------------------------------------------------
# Chain fetch
# ---------------------------------------------------------------------------

def _fetch_chain(symbol: str, strategy: str,
                 min_dte: int, max_dte: int,
                 min_oi: int, min_otm_pct: float) -> tuple[float, list[dict]]:
    """
    Fetch option chain via yfinance and return (stock_price, candidate_rows).
    strategy: 'covered-call' | 'csp'
    """
    with contextlib.redirect_stderr(io.StringIO()):
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        info   = ticker.fast_info
        stock_price = float(info.last_price)
        expirations = ticker.options          # tuple of date strings

    today    = date.today()
    min_date = today + timedelta(days=min_dte)
    max_date = today + timedelta(days=max_dte)

    candidates = []
    for exp_str in expirations:
        exp_date = date.fromisoformat(exp_str)
        if not (min_date <= exp_date <= max_date):
            continue
        dte = (exp_date - today).days

        with contextlib.redirect_stderr(io.StringIO()):
            chain = ticker.option_chain(exp_str)

        if strategy == "covered-call":
            df = filter_calls(chain.calls, stock_price,
                              min_oi=min_oi, min_otm_pct=min_otm_pct)
            score_fn = score_covered_call
        else:
            df = filter_puts(chain.puts, stock_price,
                             min_oi=min_oi, min_otm_pct=min_otm_pct)
            score_fn = score_csp

        for _, row in df.iterrows():
            bid = float(row["bid"]) if row["bid"] > 0 else float(row["ask"]) * 0.9
            if bid <= 0:
                continue
            scores = score_fn(float(row["strike"]), bid, stock_price, dte)
            candidates.append({
                "expiry":            exp_str,
                "strike":            float(row["strike"]),
                "bid":               float(row["bid"]),
                "ask":               float(row["ask"]),
                "dte":               dte,
                "openInterest":      int(row["openInterest"]),
                "volume":            int(row.get("volume", 0) or 0),
                "iv":                round(float(row["impliedVolatility"]), 4),
                **scores,
            })

    return stock_price, candidates


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def _strategy_label(strategy: str) -> str:
    return "Covered Call" if strategy == "covered-call" else "Cash-Secured Put"


def _print_candidates(symbol: str, strategy: str, stock_price: float,
                      candidates: list[dict], shares: int = 0):
    label = _strategy_label(strategy)
    now   = datetime.now().strftime("%Y-%m-%d %H:%M")
    sep   = "=" * 74

    print(f"\n{sep}")
    print(f"  {symbol} — {label} Candidates  (price: ${stock_price:.2f})  {now}")
    if shares:
        contracts = shares // 100
        print(f"  You hold {shares} shares → up to {contracts} contracts")
    print(sep)

    if not candidates:
        print("  No candidates found matching your criteria.")
        print(sep)
        return

    hdr = f"  {'Expiry':<12} {'Strike':>7} {'Bid':>6} {'Ask':>6} {'DTE':>4}  {'Yield/mo':>9}  {'OTM%':>6}  {'Breakeven':>10}  {'OI':>7}  {'IV':>6}"
    print(hdr)
    print(f"  {'-'*70}")
    for c in candidates:
        print(
            f"  {c['expiry']:<12} "
            f"{c['strike']:>7.2f} "
            f"{c['bid']:>6.2f} "
            f"{c['ask']:>6.2f} "
            f"{c['dte']:>4}  "
            f"{c['monthly_yield_pct']:>8.2f}%  "
            f"{c['otm_pct']:>5.1f}%  "
            f"${c['breakeven']:>9.2f}  "
            f"{c['openInterest']:>7,}  "
            f"{c['iv']*100:>5.1f}%"
        )
    print(sep)


def _slack_candidates(symbol: str, strategy: str, stock_price: float,
                      candidates: list[dict], top: int = 5) -> str:
    label = _strategy_label(strategy)
    lines = [f"*{symbol} — {label} Candidates* (${stock_price:.2f})"]
    if not candidates:
        lines.append("No candidates found.")
        return "\n".join(lines)

    lines.append(f"{'Expiry':<12} {'Strike':>7} {'Bid':>5} {'DTE':>4}  {'Yield/mo':>8}  {'OTM%':>5}  {'Break':>7}")
    for c in candidates[:top]:
        lines.append(
            f"{c['expiry']:<12} {c['strike']:>7.2f} {c['bid']:>5.2f} "
            f"{c['dte']:>4}  {c['monthly_yield_pct']:>7.2f}%  "
            f"{c['otm_pct']:>4.1f}%  ${c['breakeven']:>6.2f}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_vault20_positions() -> list[dict]:
    if not os.path.exists(VAULT20_DATA):
        return []
    return json.loads(open(VAULT20_DATA).read()).get("open", [])


def main():
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

    parser = argparse.ArgumentParser(
        prog="option_finder",
        description="Find covered call / CSP candidates for a symbol")
    parser.add_argument("symbol",   nargs="?",
                        help="Ticker symbol (e.g. INTC). Omit with --all to scan vault20 positions.")
    parser.add_argument("strategy", nargs="?", default="covered-call",
                        choices=["covered-call", "cc", "csp", "put"],
                        help="covered-call (default) or csp")
    parser.add_argument("--min-dte",    type=int,   default=21)
    parser.add_argument("--max-dte",    type=int,   default=60)
    parser.add_argument("--min-oi",     type=int,   default=100)
    parser.add_argument("--min-otm",    type=float, default=1.0,
                        help="Min OTM%% (default 1%)")
    parser.add_argument("--top",        type=int,   default=8)
    parser.add_argument("--slack",      action="store_true",
                        help="Send results to Slack")
    parser.add_argument("--all",        action="store_true",
                        help="Scan all vault20 equity positions")
    args = parser.parse_args()

    # Normalise strategy alias
    strategy = "csp" if args.strategy in ("csp", "put") else "covered-call"

    # Determine symbols to scan
    if args.all:
        positions = _load_vault20_positions()
        equities  = [(p["symbol"], int(p["shares"]))
                     for p in positions
                     if "_" not in p["symbol"] and p["shares"] > 0]
        if not equities:
            print("No equity positions found in vault20.")
            return
    elif args.symbol:
        # check if we hold shares in vault20
        positions = _load_vault20_positions()
        held      = next((p for p in positions if p["symbol"] == args.symbol.upper()), None)
        shares    = int(held["shares"]) if held and held["shares"] > 0 else 0
        equities  = [(args.symbol.upper(), shares)]
    else:
        parser.print_help()
        return

    all_slack = []
    for symbol, shares in equities:
        print(f"\nFetching {symbol} option chain…", end=" ", flush=True)
        try:
            stock_price, candidates = _fetch_chain(
                symbol, strategy,
                min_dte=args.min_dte, max_dte=args.max_dte,
                min_oi=args.min_oi, min_otm_pct=args.min_otm,
            )
        except Exception as exc:
            print(f"error: {exc}")
            continue

        ranked = format_candidates(candidates, top=args.top)
        print(f"done — {len(ranked)} candidates")
        _print_candidates(symbol, strategy, stock_price, ranked, shares=shares)

        if args.slack:
            all_slack.append(_slack_candidates(symbol, strategy, stock_price, ranked))

    if args.slack and all_slack:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from notify_slack import send
        send("\n\n".join(all_slack))
        print("  → Sent to Slack.")


if __name__ == "__main__":
    main()
