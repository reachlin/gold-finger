"""
Live pullback-in-trend scanner for manual+Claude-assisted trading.

Runs in a tmux session (schwab-screen). Scans the watchlist every
SCAN_INTERVAL_MIN minutes during US market hours. Exits automatically
when the market closes (4pm ET).

Modes:
  normal:  prints signal, waits for y/n from Claude or user
  --paper: same prompt flow, but auto-records buys/sells and tracks P&L

Usage:
    python schwab/live_scanner.py            # normal mode
    python schwab/live_scanner.py --paper    # paper trading mode

Then tell Claude Code: "/schwab watch"
"""
import os
import sys
import time
import argparse
from datetime import datetime, timedelta, timezone

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from trend_scanner import compute_indicators
from vault76.overseer import Overseer
from vault76.armory.raider import Raider
from vault76.armory.scavenger import Scavenger

SCAN_INTERVAL_MIN = 5
MARKET_OPEN_ET    = 9
MARKET_CLOSE_ET   = 16
BUDGET_PER_TRADE  = 600     # USD per paper trade

SIGNAL_START = "===SIGNAL_START==="
SIGNAL_END   = "===SIGNAL_END==="
VERDICT_OK   = "===APPROVED==="
VERDICT_SKIP = "===SKIPPED==="

WATCHLIST = [
    "NVDA", "AMD", "AAPL", "AMZN",
    "META", "MSFT", "GOOGL",
    "IBM", "INTC", "IONQ", "KO",
    # Scavenger prime targets — high % time sideways
    "MMM", "PG", "XOM",
]

PAPER_TRADES_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "paper_trades.json"
)


# ---------------------------------------------------------------------------
# Market hours
# ---------------------------------------------------------------------------

def _et_hour() -> float:
    now_utc = datetime.now(timezone.utc)
    return (now_utc.hour - 4) % 24 + now_utc.minute / 60


def _is_weekday() -> bool:
    return datetime.now(timezone.utc).weekday() < 5


def _is_market_hours() -> bool:
    return _is_weekday() and MARKET_OPEN_ET <= _et_hour() < MARKET_CLOSE_ET


def _market_closed_for_today() -> bool:
    if not _is_weekday():
        return True
    return _et_hour() >= MARKET_CLOSE_ET


# ---------------------------------------------------------------------------
# Schwab data fetch
# ---------------------------------------------------------------------------

def _fetch_history(client, symbol: str, days: int = 200) -> pd.DataFrame | None:
    end   = datetime.now()
    start = end - timedelta(days=days)
    try:
        resp = client.get_price_history_every_day(
            symbol, start_datetime=start, end_datetime=end
        )
        resp.raise_for_status()
        candles = resp.json().get("candles", [])
        if not candles:
            return None
        df = pd.DataFrame(candles)
        df["datetime"] = pd.to_datetime(df["datetime"], unit="ms")
        df["volume"]   = df["volume"].astype(float)
        return df[["datetime", "open", "high", "low", "close", "volume"]]
    except Exception as exc:
        print(f"    fetch {symbol}: {exc}")
        return None


def _fetch_with_indicators(client, symbol: str) -> pd.DataFrame | None:
    df = _fetch_history(client, symbol)
    if df is None or len(df) < 60:
        return None
    return compute_indicators(df).dropna().reset_index(drop=True)


_overseer  = Overseer()
_raider    = Raider()
_scavenger = Scavenger()


def _fetch_regime(client) -> str:
    """Fetch SPY + VIX and return the current Overseer regime string."""
    df = _fetch_history(client, "SPY")
    if df is None or len(df) < 60:
        return Overseer.RECLAMATION   # can't determine — don't block

    try:
        from signal_verifier import fetch_vix
        vix = fetch_vix()
    except Exception:
        vix = 20.0

    return _overseer.classify(df, vix=vix)


def _scan_all(client, regime: str = Overseer.RECLAMATION) -> list[dict]:
    signals = []
    for symbol in WATCHLIST:
        df_ind = _fetch_with_indicators(client, symbol)
        if df_ind is None:
            continue
        # The Raider: buy pullbacks in uptrends
        r = _raider.scan(symbol, df_ind, regime=regime)
        if r["signal"] != "NONE":
            signals.append(r)
        # The Scavenger: sell options on sideways stocks
        s = _scavenger.scan(symbol, df_ind, regime=regime)
        if s["signal"] != "NONE":
            signals.append(s)
    return signals


def _get_current_prices(portfolio, price_fetcher) -> dict:
    """Fetch latest close price for each open position symbol."""
    prices = {}
    for pos in portfolio.open_positions:
        try:
            prices[pos["symbol"]] = price_fetcher(pos["symbol"])["close"]
        except Exception:
            pass
    return prices


def _make_price_fetcher(client):
    """Returns a callable price_fetcher(symbol) for PaperPortfolio.check_positions."""
    def fetcher(symbol: str) -> dict:
        df_ind = _fetch_with_indicators(client, symbol)
        if df_ind is None:
            raise RuntimeError(f"No data for {symbol}")
        last = df_ind.iloc[-1]
        return {
            "high":  float(last["high"]),
            "low":   float(last["low"]),
            "close": float(last["close"]),
            "ema20": float(last["ema20"]),
            "ema50": float(last["ema50"]),
        }
    return fetcher


# ---------------------------------------------------------------------------
# Signal display + verdict prompt
# ---------------------------------------------------------------------------

def _print_signal(s: dict, paper: bool = False):
    now      = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    mode_tag = " [PAPER]" if paper else ""
    sig      = s["signal"]
    card     = s.get("card", "?")

    print(f"\n{SIGNAL_START}")
    print(f"TIME:    {now}{mode_tag}")
    print(f"SYMBOL:  {s['symbol']}  [{card.upper()}]")
    print(f"ACTION:  {sig}")
    print(f"CLOSE:   ${s.get('close', '?')}")

    if sig == "BUY":
        print(f"ENTRY:   ${s['entry']}")
        print(f"TARGET:  ${s['target']}")
        print(f"STOP:    ${s['stop']}")
        print(f"RSI:     {s.get('rsi', '?')}  ADX: {s.get('adx', '?')}")
        print(f"EMA20:   ${s.get('ema20', '?')}  EMA50: ${s.get('ema50', '?')}")

    elif sig == "SELL_PUT":
        print(f"STRIKE:  ${s['strike']}  ({s.get('otm_pct', '5')}% OTM)")
        print(f"PREMIUM: ${s['premium']}/sh  (${s['premium']*100:.0f}/contract)  "
              f"+{s['premium_pct']:.2f}% yield")
        print(f"DTE:     {s['dte']} days")
        print(f"MAX LOSS:${s.get('max_loss', '?')}/contract if assigned")
        print(f"HV:      {s.get('hv', '?')}%  ADX: {s.get('adx', '?')}")

    elif sig == "SELL_CALL":
        print(f"STRIKE:  ${s['strike']}  ({s.get('otm_pct', '8')}% OTM)")
        print(f"PREMIUM: ${s['premium']}/sh  (${s['premium']*100:.0f}/contract)  "
              f"+{s['premium_pct']:.2f}% yield")
        print(f"COST BASIS: ${s.get('cost_basis', '?')}")
        print(f"DTE:     {s['dte']} days")
        print(f"MAX GAIN:${s.get('max_gain', '?')}/contract if called away")
        print(f"HV:      {s.get('hv', '?')}%  ADX: {s.get('adx', '?')}")

    print(f"REASON:  {s['reason']}")
    print(f"{SIGNAL_END}")
    print("\n>>> Waiting for Claude verification... (yes=approve / n=skip / q=quit)")
    print("Proceed? [yes/N]: ", end="", flush=True)


def _wait_for_verdict() -> str:
    try:
        ans = input().strip().lower()
        # Accept "yes" or "y", treat anything else as skip
        if ans == "yes":
            return "y"
        return ans
    except (EOFError, KeyboardInterrupt):
        return "q"


# ---------------------------------------------------------------------------
# Startup banner
# ---------------------------------------------------------------------------

def _send_slack(message: str):
    try:
        import sys as _sys
        _sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from notify_slack import send
        send(message)
    except Exception as exc:
        print(f"  [Slack] failed to send: {exc}")


def _position_lines(portfolio, price_fetcher=None) -> tuple[list[str], list[str]]:
    """
    Build terminal and Slack lines for open positions.
    Returns (terminal_lines, slack_lines).
    """
    term, slack = [], []
    positions = portfolio.open_positions if portfolio else []

    if not positions:
        term.append("  Positions:       none")
        slack.append("*Positions:* none")
        return term, slack

    term.append(f"  Positions ({len(positions)}):")
    slack.append(f"*Positions ({len(positions)}):*")

    for pos in positions:
        sym    = pos["symbol"]
        shares = pos["shares"]
        entry  = pos["entry"]
        tgt    = pos["target"]
        stp    = pos["stop"]
        since  = pos["entry_date"][:10]

        cur_str = ""
        slack_cur = ""
        if price_fetcher:
            try:
                cur = price_fetcher(sym)["close"]
                pct = (cur - entry) / entry * 100
                usd = shares * (cur - entry)
                cur_str   = f"  → ${cur:.2f} ({pct:+.1f}%  ${usd:+.0f})"
                slack_cur = f" → ${cur:.2f} ({pct:+.1f}%, ${usd:+.0f})"
            except Exception:
                pass

        term.append(f"    {sym:<6} {shares} sh @ ${entry:.2f}"
                    f"  tgt ${tgt:.2f}  stp ${stp:.2f}  [{since}]{cur_str}")
        slack.append(f"  • {sym} {shares}sh @${entry:.2f}"
                     f" | tgt ${tgt:.2f} stp ${stp:.2f} [{since}]{slack_cur}")

    return term, slack


def _print_startup(client, paper: bool, portfolio=None, price_fetcher=None):
    """Print the daily startup banner and post it to Slack."""
    now_str  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    mode     = "PAPER TRADING" if paper else "LIVE"
    mode_ico = "📄" if paper else "💰"

    try:
        regime = _fetch_regime(client)
    except Exception:
        regime = Overseer.WASTELAND

    regime_label = _overseer.describe(regime)
    cards        = _overseer.recommend_roles(regime)
    cards_str    = ", ".join(c.upper() for c in cards) if cards else "NONE — stand down"

    pos_term, pos_slack = _position_lines(portfolio, price_fetcher)

    # ── Terminal banner ──────────────────────────────────────────────────────
    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  VAULT 76 — DAILY BRIEFING  {now_str}")
    print(sep)
    print(f"  Mode:            {mode_ico}  {mode}")
    print(f"  Regime:          {regime_label}")
    print(f"  Active cards:    {cards_str}")
    print(f"  Watchlist ({len(WATCHLIST)}):   {', '.join(WATCHLIST)}")
    print(f"  Scan interval:   {SCAN_INTERVAL_MIN} min  |  Budget: ${BUDGET_PER_TRADE}/trade")
    if portfolio:
        print(f"  Cash:            ${portfolio.cash:,.2f}")
    for line in pos_term:
        print(line)
    if paper:
        print(f"  Paper trades:    {PAPER_TRADES_PATH}")
    print(f"  Monitor:         /schwab watch")
    print(sep)

    # ── Slack message ────────────────────────────────────────────────────────
    slack_lines = [
        f"*VAULT 76 — Daily Briefing* {now_str}",
        f"*Mode:* {mode_ico} {mode}",
        f"*Regime:* {regime_label}",
        f"*Active cards:* {cards_str}",
        f"*Watchlist ({len(WATCHLIST)}):* {', '.join(WATCHLIST)}",
        f"*Scan interval:* {SCAN_INTERVAL_MIN} min | *Budget/trade:* ${BUDGET_PER_TRADE}",
    ]
    if portfolio:
        slack_lines.append(f"*Cash:* ${portfolio.cash:,.2f}")
    slack_lines += pos_slack
    _send_slack("\n".join(slack_lines))


def _print_eod(portfolio, price_fetcher, scan_count: int):
    """Print end-of-day summary and post it to Slack."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cur_prices = _get_current_prices(portfolio, price_fetcher) if portfolio else {}

    print(f"\n{'='*62}")
    print(f"  VAULT 76 — END OF DAY  {now_str}  (scans: {scan_count})")
    print(f"{'='*62}")
    if portfolio:
        portfolio.print_status(cur_prices)

    pos_term, pos_slack = _position_lines(portfolio, price_fetcher)

    if portfolio:
        s = portfolio.summary(cur_prices)
        pnl_sign = "+" if s["total_pnl_dollar"] >= 0 else ""
        slack_lines = [
            f"*VAULT 76 — End of Day* {now_str}",
            f"*Scans today:* {scan_count}",
            f"*Cash:* ${s['cash']:,.2f}  |  "
            f"*Total:* ${s['total_value']:,.2f}  |  "
            f"*P&L:* {pnl_sign}${s['total_pnl_dollar']:,.2f} ({pnl_sign}{s['total_pnl_pct']:.2f}%)",
            f"*Realized:* ${s['realized_pnl_dollar']:+,.2f}  |  "
            f"*Unrealized:* ${s['unrealized_pnl_dollar']:+,.2f}",
        ] + pos_slack
    else:
        slack_lines = [f"*VAULT 76 — End of Day* {now_str}  (scans: {scan_count})"]

    _send_slack("\n".join(slack_lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", action="store_true",
                        help="Paper trading mode — track fake buys/sells, show P&L")
    args = parser.parse_args()

    import schwab as schwab_lib
    CLIENT_ID     = os.environ["SCHWAB_CLIENT_ID"]
    CLIENT_SECRET = os.environ["SCHWAB_CLIENT_SECRET"]
    TOKEN_PATH    = os.path.join(os.path.dirname(__file__), "schwab_token.json")
    client = schwab_lib.auth.client_from_token_file(TOKEN_PATH, CLIENT_ID, CLIENT_SECRET)

    portfolio = None
    if args.paper:
        from paper_portfolio import PaperPortfolio
        portfolio = PaperPortfolio(PAPER_TRADES_PATH)

    price_fetcher = _make_price_fetcher(client)
    _print_startup(client, paper=args.paper, portfolio=portfolio,
                   price_fetcher=price_fetcher)
    scan_count = 0

    while True:
        now = datetime.now().strftime("%H:%M:%S")

        if _market_closed_for_today():
            _print_eod(portfolio, price_fetcher, scan_count)
            if portfolio:
                portfolio.log_scan(scan_num=scan_count, symbols_scanned=0,
                                   signals_found=0)
            print("Scanner exiting. Restart tomorrow (9am–4pm ET).")
            sys.exit(0)

        if not _is_market_hours():
            print(f"[{now}] Pre-market — waiting for open (9am ET). Sleeping 5 min.")
            time.sleep(300)
            continue

        scan_count += 1
        regime     = _fetch_regime(client)
        regime_str = _overseer.describe(regime)
        print(f"\n[{now}] Scan #{scan_count} — {regime_str} — scanning {len(WATCHLIST)} symbols...")

        # Paper mode: check existing positions against current prices
        if portfolio:
            exits = portfolio.check_positions(price_fetcher)
            if exits:
                print(f"\n  [PAPER] Position exits this scan:")
                for ex in exits:
                    icon = "+" if ex["pnl_dollar"] > 0 else "-"
                    print(f"    {ex['symbol']:<6} CLOSED  {icon}${abs(ex['pnl_dollar']):.2f}"
                          f"  ({ex['pnl_pct']:+.1f}%)  [{ex['exit_reason']}]"
                          f"  exit ${ex['exit']:.2f}")

        # Scan for new signals
        signals = _scan_all(client, regime=regime)

        if portfolio:
            portfolio.log_scan(
                scan_num=scan_count,
                symbols_scanned=len(WATCHLIST),
                signals_found=len(signals),
            )

        if not signals:
            print(f"  No BUY signals.")
        else:
            print(f"  {len(signals)} signal(s) found.")
            for s in signals:
                _print_signal(s, paper=args.paper)
                verdict = _wait_for_verdict()
                shares  = max(1, int(BUDGET_PER_TRADE / s["entry"]))

                if verdict == "y":
                    print(f"\n{VERDICT_OK}")
                    print(f"APPROVED: {s['symbol']} @ ${s['entry']}")
                    if portfolio:
                        pos = portfolio.open_position(
                            s["symbol"], s["entry"], s["target"], s["stop"], shares
                        )
                        portfolio.log_signal(
                            s["symbol"], s["entry"], s["target"], s["stop"],
                            s["rsi"], s["adx"], verdict="APPROVED",
                        )
                        if pos:
                            print(f"  [PAPER] BUY {pos['shares']} sh @ ${pos['entry']:.2f}"
                                  f"  tgt ${pos['target']:.2f}  stp ${pos['stop']:.2f}"
                                  f"  cost ${pos['cost']:.2f}  cash left ${portfolio.cash:.2f}")
                        else:
                            print("  [PAPER] Insufficient cash — trade not recorded.")
                    else:
                        print(f"  Suggested: BUY {shares} shares @ market")
                        print(f"  Target ${s['target']}  |  Stop ${s['stop']}")
                    print(VERDICT_OK)

                elif verdict == "q":
                    if portfolio:
                        print("\nFinal portfolio status:")
                        cur_prices = _get_current_prices(portfolio, price_fetcher)
                        portfolio.print_status(cur_prices)
                    print("Exiting scanner.")
                    sys.exit(0)

                else:
                    if portfolio:
                        portfolio.log_signal(
                            s["symbol"], s["entry"], s["target"], s["stop"],
                            s["rsi"], s["adx"], verdict="SKIPPED",
                        )
                    print(f"\n{VERDICT_SKIP}")
                    print(f"SKIPPED: {s['symbol']}")
                    print(VERDICT_SKIP)

        # Show portfolio status at end of each scan
        if portfolio:
            cur_prices = _get_current_prices(portfolio, price_fetcher)
            portfolio.print_status(cur_prices)

        print(f"  Next scan in {SCAN_INTERVAL_MIN} min.")
        time.sleep(SCAN_INTERVAL_MIN * 60)


if __name__ == "__main__":
    main()
