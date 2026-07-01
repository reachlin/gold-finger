"""Fetch US stock options from multiple sources with automatic fallback.

Sources (in order): yfinance → Tradier → Polygon.io
API keys for Tradier and Polygon are read from .env (TRADIER_API_KEY, POLYGON_API_KEY).
"""

import argparse
import os
import sys
from datetime import date, timedelta

import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

STANDARD_COLUMNS = [
    "symbol", "expiration", "option_type", "strike",
    "bid", "ask", "last", "volume", "open_interest",
    "implied_volatility", "delta", "gamma", "theta", "vega",
]


def _default_end_date() -> str:
    return (date.today() + timedelta(days=60)).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Provider: yfinance
# ---------------------------------------------------------------------------

def _fetch_yfinance(symbol: str, end_date: str) -> pd.DataFrame:
    ticker = yf.Ticker(symbol)
    expirations = [d for d in ticker.options if d <= end_date]
    if not expirations:
        return pd.DataFrame(columns=STANDARD_COLUMNS)

    frames = []
    for exp in expirations:
        chain = ticker.option_chain(exp)
        for opt_type, raw in (("call", chain.calls), ("put", chain.puts)):
            if raw.empty:
                continue
            n = len(raw)
            frames.append(pd.DataFrame({
                "symbol": [symbol] * n,
                "expiration": [exp] * n,
                "option_type": [opt_type] * n,
                "strike": raw["strike"].values,
                "bid": raw["bid"].values,
                "ask": raw["ask"].values,
                "last": raw["lastPrice"].values,
                "volume": raw["volume"].values,
                "open_interest": raw["openInterest"].values,
                "implied_volatility": raw["impliedVolatility"].values,
                "delta": raw["delta"].values if "delta" in raw.columns else [None] * n,
                "gamma": raw["gamma"].values if "gamma" in raw.columns else [None] * n,
                "theta": raw["theta"].values if "theta" in raw.columns else [None] * n,
                "vega": raw["vega"].values if "vega" in raw.columns else [None] * n,
            }))

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=STANDARD_COLUMNS)


# ---------------------------------------------------------------------------
# Provider: Tradier
# ---------------------------------------------------------------------------

def _fetch_tradier(symbol: str, end_date: str) -> pd.DataFrame:
    api_key = os.getenv("TRADIER_API_KEY", "")
    base_url = os.getenv("TRADIER_BASE_URL", "https://sandbox.tradier.com")
    headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}

    # Step 1: fetch available expirations
    exp_resp = requests.get(
        f"{base_url}/v1/markets/options/expirations",
        params={"symbol": symbol},
        headers=headers,
        timeout=15,
    )
    exp_resp.raise_for_status()
    dates = exp_resp.json().get("expirations", {}).get("date", []) or []
    if isinstance(dates, str):
        dates = [dates]
    expirations = [d for d in dates if d <= end_date]

    # Step 2: fetch chains for each expiration
    frames = []
    for exp in expirations:
        resp = requests.get(
            f"{base_url}/v1/markets/options/chains",
            params={"symbol": symbol, "expiration": exp, "greeks": "true"},
            headers=headers,
            timeout=15,
        )
        resp.raise_for_status()
        options = resp.json().get("options", {}).get("option", []) or []
        for o in options:
            greeks = o.get("greeks") or {}
            frames.append({
                "symbol": symbol,
                "expiration": o.get("expiration_date", exp),
                "option_type": o.get("option_type"),
                "strike": o.get("strike"),
                "bid": o.get("bid"),
                "ask": o.get("ask"),
                "last": o.get("last"),
                "volume": o.get("volume"),
                "open_interest": o.get("open_interest"),
                "implied_volatility": greeks.get("mid_iv"),
                "delta": greeks.get("delta"),
                "gamma": greeks.get("gamma"),
                "theta": greeks.get("theta"),
                "vega": greeks.get("vega"),
            })

    if not frames:
        return pd.DataFrame(columns=STANDARD_COLUMNS)
    return pd.DataFrame(frames, columns=STANDARD_COLUMNS)


# ---------------------------------------------------------------------------
# Provider: Polygon.io
# ---------------------------------------------------------------------------

def _fetch_polygon(symbol: str, end_date: str) -> pd.DataFrame:
    api_key = os.getenv("POLYGON_API_KEY", "")
    url = f"https://api.polygon.io/v3/snapshot/options/{symbol}"
    params = {
        "expiration_date.lte": end_date,
        "limit": 250,
        "apiKey": api_key,
    }

    frames = []
    while url:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        for r in data.get("results", []):
            details = r.get("details", {})
            greeks = r.get("greeks", {})
            day = r.get("day", {})
            quote = r.get("last_quote", {})
            frames.append({
                "symbol": symbol,
                "expiration": details.get("expiration_date"),
                "option_type": details.get("contract_type"),
                "strike": details.get("strike_price"),
                "bid": quote.get("bid"),
                "ask": quote.get("ask"),
                "last": day.get("last_price"),
                "volume": day.get("volume"),
                "open_interest": r.get("open_interest"),
                "implied_volatility": r.get("implied_volatility"),
                "delta": greeks.get("delta"),
                "gamma": greeks.get("gamma"),
                "theta": greeks.get("theta"),
                "vega": greeks.get("vega"),
            })
        next_url = data.get("next_url")
        url = next_url if next_url else None
        params = {}  # next_url already contains query params

    if not frames:
        return pd.DataFrame(columns=STANDARD_COLUMNS)
    return pd.DataFrame(frames, columns=STANDARD_COLUMNS)


# ---------------------------------------------------------------------------
# Fallback orchestrator
# ---------------------------------------------------------------------------

def fetch_us_options(
    symbol: str,
    start_date: str | None = None,
    end_date: str | None = None,
    providers=None,
) -> pd.DataFrame:
    """Fetch US options for *symbol* within the given date range.

    Args:
        symbol:     US stock ticker, e.g. "AAPL".
        start_date: Earliest expiration date (YYYY-MM-DD). Defaults to today.
        end_date:   Latest expiration date (YYYY-MM-DD). Defaults to today + 60 days.
        providers:  List of provider callables to try in order.
                    Defaults to [_fetch_yfinance, _fetch_tradier, _fetch_polygon].

    Returns:
        DataFrame with STANDARD_COLUMNS, filtered to [start_date, end_date].

    Raises:
        RuntimeError: When all providers fail.
    """
    if end_date is None:
        end_date = _default_end_date()
    if start_date is None:
        start_date = date.today().strftime("%Y-%m-%d")

    # Resolve providers at call time so monkey-patching works in tests.
    _mod = sys.modules[__name__]
    if providers is None:
        providers = [_mod._fetch_yfinance, _mod._fetch_tradier, _mod._fetch_polygon]

    last_exc = None
    for provider in providers:
        try:
            df = provider(symbol, end_date)
            if not df.empty:
                df = df[df["expiration"] >= start_date]
                return df.reset_index(drop=True)
        except Exception as e:
            print(f"[WARN] {getattr(provider, '__name__', repr(provider))} failed: {e}, trying next...")
            last_exc = e

    raise RuntimeError(f"All providers failed. Last error: {last_exc}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Download US stock options data with multi-source fallback."
    )
    p.add_argument("symbol", help="US stock ticker, e.g. AAPL")
    p.add_argument(
        "--start-date",
        default=None,
        metavar="YYYY-MM-DD",
        help="Earliest expiration date (default: today)",
    )
    p.add_argument(
        "--end-date",
        default=None,
        metavar="YYYY-MM-DD",
        help="Latest expiration date (default: today + 60 days)",
    )
    p.add_argument(
        "--csv",
        default=None,
        metavar="PATH",
        help="Output CSV path (default: data/{SYMBOL}_options.csv)",
    )
    p.add_argument(
        "--provider",
        choices=["yfinance", "tradier", "polygon"],
        default=None,
        help="Force a specific provider instead of trying all",
    )
    return p


def main():
    args = _build_parser().parse_args()

    _mod = sys.modules[__name__]
    provider_map = {
        "yfinance": [_mod._fetch_yfinance],
        "tradier": [_mod._fetch_tradier],
        "polygon": [_mod._fetch_polygon],
    }
    providers = provider_map[args.provider] if args.provider else None

    df = fetch_us_options(
        args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        providers=providers,
    )

    out_path = args.csv or f"data/{args.symbol}_options.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")
    print(df.head())


if __name__ == "__main__":
    main()
