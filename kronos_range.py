"""
Kronos 30-day price range predictor for sideways stocks.

For each symbol in the watchlist, runs the Kronos foundation model on the
last LOOKBACK daily bars and predicts the next PRED_LEN trading days.
Outputs:
  - support  : min(predicted_low)  → put strike floor
  - resistance: max(predicted_high) → covered call ceiling
  - range_pct : (resistance - support) / current_price × 100
  - strike_ok : True if our 5% OTM put strike stays below predicted support

Usage:
  python kronos_range.py              # full watchlist
  python kronos_range.py KO IBM UNH   # specific symbols
"""
import os, sys, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, "/Users/lincai/dev/private/Kronos")

DATA_DIR   = os.path.join(os.path.dirname(__file__), "data")
LOOKBACK   = 400    # bars of history fed to Kronos (max_context=512, keep some buffer)
PRED_LEN   = 30     # trading days to forecast (~1 month)
OTM_PCT    = 0.05   # our standard put strike distance

WATCHLIST = [
    "NVDA", "AMD",  "AAPL", "AMZN", "META", "MSFT", "GOOGL",
    "IBM",  "INTC", "IONQ", "KO",   "MMM",  "XOM",  "PG",  "TSLA",
    "UNH",  "HD",   "ABT",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _future_biz_dates(last: pd.Timestamp, n: int) -> pd.Series:
    """Return n business days starting the day after `last` as a Series (Kronos needs .dt accessor)."""
    return pd.Series(pd.bdate_range(start=last + pd.Timedelta(days=1), periods=n))


def predict_range(df: pd.DataFrame, predictor, lookback: int = LOOKBACK,
                  pred_len: int = PRED_LEN) -> dict:
    """
    Run Kronos on the tail of `df` and return the predicted price range.

    Parameters
    ----------
    df        : DataFrame with columns [datetime, open, high, low, close, volume]
    predictor : KronosPredictor instance
    lookback  : number of historical bars to use as context
    pred_len  : number of future bars to predict

    Returns
    -------
    dict with keys:
      support        – min predicted low  (put strike floor)
      resistance     – max predicted high (covered call ceiling)
      range_pct      – (resistance - support) / current_price * 100
      current_price  – last close price
      pred_df        – full Kronos prediction DataFrame
    """
    if len(df) < lookback + 1:
        raise ValueError(
            f"Not enough history: need {lookback + 1} bars, got {len(df)}"
        )

    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    tail       = df.tail(lookback).reset_index(drop=True)
    x_df       = tail[["open", "high", "low", "close"]]
    x_ts       = tail["datetime"]
    last_date  = tail["datetime"].iloc[-1]
    y_ts       = _future_biz_dates(last_date, pred_len)

    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_ts,
        y_timestamp=y_ts,
        pred_len=pred_len,
        T=1.0,
        top_p=0.9,
        sample_count=1,
        verbose=False,
    )

    support    = float(pred_df["low"].min())
    resistance = float(pred_df["high"].max())
    cur_price  = float(tail["close"].iloc[-1])
    range_pct  = (resistance - support) / cur_price * 100

    return {
        "support":       support,
        "resistance":    resistance,
        "range_pct":     range_pct,
        "current_price": cur_price,
        "pred_df":       pred_df,
    }


def scan_watchlist(symbols: list[str], predictor,
                   data_dir: str = DATA_DIR) -> list[dict]:
    """Run predict_range for each symbol and return a list of result rows."""
    rows = []
    for sym in symbols:
        path = os.path.join(data_dir, f"{sym.lower()}_history.csv")
        if not os.path.exists(path):
            print(f"  [{sym}] no history file — skipping")
            continue
        df = pd.read_csv(path)
        try:
            result = predict_range(df, predictor)
        except Exception as exc:
            print(f"  [{sym}] error: {exc}")
            continue

        cur   = result["current_price"]
        sup   = result["support"]
        res   = result["resistance"]
        rng   = result["range_pct"]
        strike = cur * (1 - OTM_PCT)

        rows.append({
            "symbol":     sym,
            "current":    cur,
            "support":    sup,
            "resistance": res,
            "range_pct":  rng,
            "strike_5pct":strike,
            # True → our standard 5% OTM strike is below the predicted floor → safer entry
            "strike_ok":  strike > sup,
            # margin between our strike and the predicted support floor (positive = buffer)
            "strike_buf_pct": (strike - sup) / cur * 100,
        })
        verdict = "OK " if strike > sup else "WARN"
        print(f"  {sym:<6}  cur ${cur:>8.2f}  "
              f"support ${sup:>8.2f}  resist ${res:>8.2f}  "
              f"range {rng:>5.1f}%  strike ${strike:>8.2f}  [{verdict}]")
    return rows


def print_report(rows: list[dict]) -> None:
    if not rows:
        print("No results.")
        return

    sep = "=" * 82
    print(f"\n{sep}")
    print(f"  KRONOS 30-DAY RANGE FORECAST")
    print(f"  Lookback: {LOOKBACK} bars  |  Pred: {PRED_LEN} trading days  |  OTM: {OTM_PCT*100:.0f}%")
    print(sep)
    print(f"  {'Symbol':<6}  {'Current':>8}  {'Support':>8}  {'Resist':>8}  "
          f"{'Range':>6}  {'Strike':>8}  {'Buf%':>6}  {'OK?':>4}")
    print(f"  {'-'*78}")

    # Sort: strike_ok first, then by buffer (widest margin first)
    rows_s = sorted(rows, key=lambda r: (-r["strike_ok"], -r["strike_buf_pct"]))
    for r in rows_s:
        flag = "" if r["strike_ok"] else "WARN"
        print(f"  {r['symbol']:<6}  "
              f"${r['current']:>7.2f}  "
              f"${r['support']:>7.2f}  "
              f"${r['resistance']:>7.2f}  "
              f"{r['range_pct']:>5.1f}%  "
              f"${r['strike_5pct']:>7.2f}  "
              f"{r['strike_buf_pct']:>+5.1f}%  "
              f"{flag}")
    print(sep)

    ok   = [r["symbol"] for r in rows if r["strike_ok"]]
    warn = [r["symbol"] for r in rows if not r["strike_ok"]]
    print(f"\n  Strike safe  ({len(ok)}):  {', '.join(ok) or 'none'}")
    print(f"  Strike risky ({len(warn)}):  {', '.join(warn) or 'none'}")
    print(f"\n  (Risky = 5% OTM strike is above Kronos predicted support floor)")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Kronos 30-day price range predictor")
    parser.add_argument("symbols", nargs="*", help="Symbols to predict (default: full watchlist)")
    args = parser.parse_args()
    symbols = [s.upper() for s in args.symbols] if args.symbols else WATCHLIST

    print("Loading Kronos model (first run downloads ~500MB from HuggingFace)...")
    from model import Kronos, KronosTokenizer, KronosPredictor
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model     = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    predictor = KronosPredictor(model, tokenizer)
    print(f"Model ready on {predictor.device}\n")

    print(f"Scanning {len(symbols)} symbols...\n")
    rows = scan_watchlist(symbols, predictor)
    print_report(rows)


if __name__ == "__main__":
    main()
