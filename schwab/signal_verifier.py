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
import yfinance as yf
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

logger = logging.getLogger(__name__)

VIX_WARN      = 20.0
VIX_BLOCK     = 30.0
EARNINGS_DAYS = 5

# Earnings date field names to probe in Schwab fundamental response
_EARNINGS_DATE_FIELDS = ("nextEarningsDate", "reportDate", "nextReportDate")

RSS_FEEDS = [
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^GSPC&region=US&lang=en-US",
    "https://feeds.marketwatch.com/marketwatch/topstories/",
]


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

def _parse_earnings_date(raw: str | None) -> date | None:
    """Parse Schwab date strings like '2025-08-27' or '2025-08-27 00:00:00.0'."""
    if not raw:
        return None
    try:
        return date.fromisoformat(str(raw)[:10])
    except (ValueError, TypeError):
        return None


def _check_via_schwab(symbol: str, schwab_client, days_ahead: int) -> bool | None:
    """
    Try Schwab get_instruments(FUNDAMENTAL) for next earnings date.
    Returns True/False if a date was found, None if no date field present.
    """
    try:
        resp = schwab_client.get_instruments(
            symbols=[symbol],
            projection=schwab_client.Instrument.Projection.FUNDAMENTAL,
        )
        resp.raise_for_status()
        data = resp.json()
        instruments = data.get("instruments", []) if isinstance(data, dict) else []
        today  = date.today()
        cutoff = today + timedelta(days=days_ahead)
        for inst in instruments:
            fund = inst.get("fundamental", {})
            for field in _EARNINGS_DATE_FIELDS:
                ed = _parse_earnings_date(fund.get(field))
                if ed is not None:
                    return today <= ed <= cutoff
        return None   # no date field found in response
    except Exception as exc:
        logger.warning("Schwab earnings check failed for %s: %s", symbol, exc)
        return None


def _check_via_yfinance(symbol: str, days_ahead: int) -> bool:
    """yfinance calendar fallback for earnings dates."""
    try:
        cal = yf.Ticker(symbol).calendar
        if not cal:
            return False
        today  = date.today()
        cutoff = today + timedelta(days=days_ahead)
        for ed in cal.get("Earnings Date", []):
            try:
                ed_date = ed.date() if hasattr(ed, "date") else _parse_earnings_date(str(ed))
                if ed_date and today <= ed_date <= cutoff:
                    return True
            except (AttributeError, TypeError):
                continue
        return False
    except Exception as exc:
        logger.warning("yfinance earnings check failed for %s: %s", symbol, exc)
        return False


def check_earnings_proximity(symbol: str, days_ahead: int = EARNINGS_DAYS,
                             schwab_client=None) -> bool:
    """
    True if earnings are within days_ahead calendar days.

    Order of precedence:
      1. Schwab get_instruments(FUNDAMENTAL) — tried when schwab_client is provided
      2. yfinance ticker.calendar — fallback if Schwab fails or returns no date
    """
    if schwab_client is not None:
        result = _check_via_schwab(symbol, schwab_client, days_ahead)
        if result is not None:
            return result
    return _check_via_yfinance(symbol, days_ahead)


# ---------------------------------------------------------------------------
# Headlines (RSS only — Schwab has no news endpoint)
# ---------------------------------------------------------------------------

def fetch_headlines(symbol: str, max_headlines: int = 20) -> list[str]:
    headlines: list[str] = []
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

def gather_signal_context(symbol: str, schwab_client=None) -> dict:
    """
    Fetch all data Claude needs to decide on a signal.
    Pass schwab_client (the live_scanner's authenticated client) to use
    Schwab fundamentals for earnings proximity; falls back to yfinance otherwise.
    """
    vix_val       = fetch_vix()
    vix_check     = check_vix(vix_val)
    near_earnings = check_earnings_proximity(symbol, schwab_client=schwab_client)
    headlines     = fetch_headlines(symbol)

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
        "hard_blocks":   hard_blocks,
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

    # Try to load Schwab client if credentials are available
    schwab_client = None
    try:
        import schwab as schwab_lib
        token_path = os.path.join(os.path.dirname(__file__), "schwab_token.json")
        schwab_client = schwab_lib.auth.client_from_token_file(
            token_path,
            os.environ["SCHWAB_CLIENT_ID"],
            os.environ["SCHWAB_CLIENT_SECRET"],
        )
    except Exception:
        pass   # will fall back to yfinance

    ctx = gather_signal_context(args.symbol, schwab_client=schwab_client)
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
