"""
Signal context fetcher — data utilities for Claude Code's live analysis.

When Claude Code watches the live_scanner tmux session and sees a signal,
it calls these functions to gather data: VIX, earnings proximity, headlines.
Claude then makes the approve/reject decision using its full skill set.

NOT used during backtesting.
"""
import os
import sys
import logging
from datetime import date, timedelta

import feedparser
import finnhub
import yfinance as yf
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

logger = logging.getLogger(__name__)

VIX_WARN  = 20.0
VIX_BLOCK = 30.0
EARNINGS_DAYS = 5

RSS_FEEDS = [
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^GSPC&region=US&lang=en-US",
    "https://feeds.marketwatch.com/marketwatch/topstories/",
]

_finnhub_key  = os.environ.get("FINNHUB_API_KEY", "")
finnhub_client = finnhub.Client(api_key=_finnhub_key)


# ---------------------------------------------------------------------------
# VIX
# ---------------------------------------------------------------------------

def fetch_vix() -> float:
    ticker = yf.Ticker("^VIX")
    try:
        return float(ticker.fast_info["lastPrice"])
    except Exception:
        hist = ticker.history(period="1d")
        return float(hist["Close"].iloc[-1]) if not hist.empty else 20.0


def check_vix(vix_value: float) -> dict:
    if vix_value >= VIX_BLOCK:
        status = "BLOCK"
    elif vix_value >= VIX_WARN:
        status = "WARN"
    else:
        status = "PASS"
    return {"status": status, "vix": vix_value}


# ---------------------------------------------------------------------------
# Earnings proximity
# ---------------------------------------------------------------------------

def check_earnings_proximity(symbol: str, days_ahead: int = EARNINGS_DAYS) -> bool:
    try:
        today  = date.today()
        cutoff = today + timedelta(days=days_ahead)
        cal    = finnhub_client.earnings_calendar(
            _from=today.isoformat(), to=cutoff.isoformat(), symbol=symbol,
        )
        for entry in cal.get("earningsCalendar", []):
            try:
                if today <= date.fromisoformat(entry["date"]) <= cutoff:
                    return True
            except (KeyError, ValueError):
                continue
        return False
    except Exception as exc:
        logger.warning("Earnings proximity check failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Headlines
# ---------------------------------------------------------------------------

def fetch_headlines(symbol: str, max_headlines: int = 20) -> list[str]:
    headlines: list[str] = []

    try:
        today    = date.today()
        week_ago = today - timedelta(days=7)
        news     = finnhub_client.company_news(
            symbol, _from=week_ago.isoformat(), to=today.isoformat()
        )
        for item in news:
            h = item.get("headline", "").strip()
            if h:
                headlines.append(h)
    except Exception as exc:
        logger.warning("Finnhub news fetch failed: %s", exc)

    for feed_url in RSS_FEEDS:
        try:
            for entry in feedparser.parse(feed_url).entries:
                title = getattr(entry, "title", "").strip()
                if title:
                    headlines.append(title)
        except Exception as exc:
            logger.warning("RSS fetch failed (%s): %s", feed_url, exc)

    return headlines[:max_headlines]


# ---------------------------------------------------------------------------
# Convenience: gather all context for a symbol in one call
# ---------------------------------------------------------------------------

def gather_signal_context(symbol: str) -> dict:
    """
    Fetch all data Claude needs to decide on a signal.
    Returns a dict Claude can read to make its approve/reject decision.
    """
    vix_val      = fetch_vix()
    vix_check    = check_vix(vix_val)
    near_earnings = check_earnings_proximity(symbol)
    headlines    = fetch_headlines(symbol)

    hard_blocks = []
    if vix_check["status"] == "BLOCK":
        hard_blocks.append(f"VIX={vix_val:.1f} — extreme market fear")
    if near_earnings:
        hard_blocks.append(f"{symbol} earnings within {EARNINGS_DAYS} days")

    return {
        "symbol":        symbol,
        "vix":           vix_val,
        "vix_status":    vix_check["status"],
        "near_earnings": near_earnings,
        "hard_blocks":   hard_blocks,        # auto-reject conditions
        "headlines":     headlines,
    }


# ---------------------------------------------------------------------------
# CLI — for quick manual checks
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("symbol", nargs="?", default="NVDA")
    args = parser.parse_args()

    ctx = gather_signal_context(args.symbol)
    print(f"\nSignal context for {ctx['symbol']}")
    print("=" * 50)
    print(f"  VIX:           {ctx['vix']:.1f}  [{ctx['vix_status']}]")
    print(f"  Near earnings: {'YES' if ctx['near_earnings'] else 'No'}")
    if ctx["hard_blocks"]:
        print(f"\n  AUTO-BLOCK conditions:")
        for b in ctx["hard_blocks"]:
            print(f"    • {b}")
    print(f"\n  Headlines ({len(ctx['headlines'])} fetched):")
    for h in ctx["headlines"][:8]:
        print(f"    - {h}")
    print("=" * 50)


if __name__ == "__main__":
    main()
