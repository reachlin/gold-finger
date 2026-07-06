"""
Relative-strength screener — feeds The Maggie (vault76/armory/maggie.py).

Qullamaggie screens for "the top 1-2% of stocks" by 1/3/6-month performance
to find the current leaders before looking for a breakout setup. This module
is universe-agnostic: hand it a dict of {symbol: daily OHLCV df} for whatever
watchlist/universe you can fetch, and it ranks them by relative strength.

CLI usage fetches history for a starting watchlist via the Schwab client —
expand WATCHLIST (or pass --symbols) to cover a broader universe.
"""
import os
import sys

import pandas as pd

LOOKBACKS = (21, 63, 126)   # ~1, 3, 6 trading months


def compute_return(df: pd.DataFrame, lookback: int) -> float | None:
    """% price return over the last `lookback` bars, or None if df is too short."""
    if len(df) <= lookback:
        return None
    recent = float(df["close"].iloc[-1])
    past   = float(df["close"].iloc[-1 - lookback])
    if past <= 0:
        return None
    return recent / past - 1


def rank_relative_strength(symbol_history: dict[str, pd.DataFrame],
                            lookbacks: tuple = LOOKBACKS) -> pd.DataFrame:
    """
    Rank symbols by relative strength across `lookbacks` (default 1/3/6mo).

    Composite rs_score is the mean of each lookback's percentile rank
    (0-100) across the universe — a stock strong on all three horizons
    ranks higher than one strong on only one.

    Symbols without enough history for every lookback are dropped.
    Returns a DataFrame sorted descending by rs_score.
    """
    rows = []
    for symbol, df in symbol_history.items():
        returns = [compute_return(df, lb) for lb in lookbacks]
        if any(r is None for r in returns):
            continue
        rows.append({"symbol": symbol, "ret_1m": returns[0],
                     "ret_3m": returns[1], "ret_6m": returns[2]})

    ranked = pd.DataFrame(rows)
    if ranked.empty:
        ranked["rs_score"] = []
        return ranked

    pct_cols = []
    for col in ("ret_1m", "ret_3m", "ret_6m"):
        pct_col = f"_pct_{col}"
        ranked[pct_col] = ranked[col].rank(pct=True) * 100
        pct_cols.append(pct_col)

    ranked["rs_score"] = ranked[pct_cols].mean(axis=1)
    ranked = ranked.drop(columns=pct_cols)
    ranked = ranked.sort_values("rs_score", ascending=False).reset_index(drop=True)
    return ranked


def top_percentile(ranked: pd.DataFrame, pct: float = 0.02) -> pd.DataFrame:
    """Top `pct` fraction of a ranked DataFrame (min 1 row) — Qullamaggie's
    "top 1-2% of stocks" leader screen."""
    n = max(1, round(len(ranked) * pct))
    return ranked.iloc[:n].reset_index(drop=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

WATCHLIST = [
    "NVDA", "AMD", "AAPL", "AMZN", "META", "MSFT", "GOOGL",
    "IBM",  "INTC", "IONQ", "KO",  "MMM",  "XOM",  "PG", "TSLA",
    "UNH",  "HD",   "ABT",
]


def _fetch_universe(client, symbols: list[str]) -> dict:
    from datetime import datetime, timedelta
    end   = datetime.now()
    start = end - timedelta(days=250)
    history = {}
    for symbol in symbols:
        try:
            resp = client.get_price_history_every_day(symbol, start_datetime=start, end_datetime=end)
            resp.raise_for_status()
            candles = resp.json().get("candles", [])
            if not candles:
                continue
            df = pd.DataFrame(candles)
            df["datetime"] = pd.to_datetime(df["datetime"], unit="ms")
            history[symbol] = df
        except Exception as exc:
            print(f"  {symbol}: fetch failed ({exc})")
    return history


def main():
    import argparse
    import schwab
    from dotenv import load_dotenv

    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="*", default=WATCHLIST,
                        help="Universe to screen (default: starting watchlist)")
    parser.add_argument("--pct", type=float, default=0.02,
                        help="Top fraction to report (default 0.02 = top 2%%)")
    args = parser.parse_args()

    client_id     = os.environ["SCHWAB_CLIENT_ID"]
    client_secret = os.environ["SCHWAB_CLIENT_SECRET"]
    token_path    = os.path.join(os.path.dirname(__file__), "schwab_token.json")
    client = schwab.auth.client_from_token_file(token_path, client_id, client_secret)

    history = _fetch_universe(client, args.symbols)
    ranked  = rank_relative_strength(history)
    top     = top_percentile(ranked, pct=args.pct)

    print(f"\n{'='*60}")
    print(f"  RELATIVE STRENGTH SCREEN — top {args.pct*100:.0f}% of {len(ranked)} symbols")
    print(f"{'='*60}")
    print(top.to_string(index=False))
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
