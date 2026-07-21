"""
vault8/weekly_range_model.py

Vault 8 — Weekly Price Range Predictor (BiLSTM + Attention)

Architecture mirrors range_predictor.py (China daily model) but targets
the *weekly* (low, high) range instead of the next-day range.

Multi-stock design
------------------
One shared model is trained on all stocks together.  Targets are expressed
as relative moves from the week's open price so the loss scale is
stock-agnostic:
    target_low  = (week_low  - week_open) / week_open
    target_high = (week_high - week_open) / week_open

A single model sees patterns from AMD, NVDA, TSLA, AAPL etc. and learns
what "a 3% down-week low" looks like across different volatility regimes.
Per-stock evaluation then identifies which stocks the model predicts best.

Window: 8 weekly bars (≈ 2 months of market context).

Features (weekly OHLCV-derived, same family as range_predictor.py):
    rsi14        14-week RSI
    macd_hist    MACD histogram (12/26/9 on weekly closes)
    boll_pctb    Bollinger %B  (20w, 2σ)
    vol_ratio    volume / 20w avg volume
    roc5         5-week rate of change
    atr_ratio    14w ATR / close  (normalised volatility)
    hl_ratio     (high-low) / close  (intraweek range, key feature)
    ret1w        1-week return  (momentum)

Loss: pinball (asymmetric quantile), tau_low=0.8, tau_high=0.2
      (same reasoning as range_predictor.py: conservative tight bounds)

Usage:
    # Train on all stocks, evaluate per-stock, save model
    python vault8/weekly_range_model.py --train

    # Evaluate a saved model
    python vault8/weekly_range_model.py --eval

    # Predict next week's range for a single ticker
    python vault8/weekly_range_model.py --predict NVDA
"""

import argparse
import os
import sys
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE    = os.path.dirname(__file__)
DATA_DIR = os.path.join(_HERE, "..", "data")
MODEL_DIR = _HERE
MODEL_PATH = os.path.join(MODEL_DIR, "vault8_weekly_range.pt")

TICKERS = [
    "nvda", "amd", "tsla", "aapl", "msft", "googl", "meta",
    "amzn", "intc", "ionq", "ko", "pg", "xom", "ibm", "unh",
    "hd", "mmm", "abt", "spy", "qqq",
]

FEATURE_COLS = [
    "rsi14", "macd_hist", "boll_pctb", "vol_ratio",
    "roc5", "atr_ratio", "hl_ratio", "ret1w",
]

WINDOW = 8   # weeks of history as input
HIDDEN = 128
LAYERS = 2
DROPOUT = 0.3
LR = 1e-3
EPOCHS = 60
BATCH  = 64

# ---------------------------------------------------------------------------
# Feature engineering on weekly bars
# ---------------------------------------------------------------------------

def resample_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily OHLCV → weekly OHLCV (Monday-anchored ISO weeks)."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["datetime"] if "datetime" in df.columns else df["date"])
    df = df.set_index("date").sort_index()
    weekly = df.resample("W-FRI").agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }).dropna()
    # Drop partial weeks with fewer than 3 trading days
    daily_counts = df["close"].resample("W-FRI").count()
    weekly = weekly[daily_counts >= 3]
    return weekly.reset_index()


def compute_features(wdf: pd.DataFrame) -> pd.DataFrame:
    """Add technical features to weekly dataframe."""
    df = wdf.copy()
    close = df["close"]
    high  = df["high"]
    low   = df["low"]
    vol   = df["volume"]

    # RSI-14
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["rsi14"] = 100 - 100 / (1 + rs)

    # MACD histogram (12/26/9 EMA on weekly closes)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    sig   = macd.ewm(span=9, adjust=False).mean()
    df["macd_hist"] = (macd - sig) / close  # normalise by price

    # Bollinger %B  (20w, 2σ)
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df["boll_pctb"] = (close - (sma20 - 2 * std20)) / (4 * std20 + 1e-9)

    # Volume ratio
    vol_ma = vol.rolling(20).mean()
    df["vol_ratio"] = vol / (vol_ma + 1e-9)

    # 5-week ROC
    df["roc5"] = close.pct_change(5)

    # ATR ratio (14w)
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    df["atr_ratio"] = tr.rolling(14).mean() / close

    # Intra-week high-low range / close
    df["hl_ratio"] = (high - low) / close

    # 1-week return
    df["ret1w"] = close.pct_change(1)

    return df


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class WeeklyRangeDataset(Dataset):
    """
    Input:  WINDOW weekly bars of features  → shape (WINDOW, n_features)
    Target: next week's (low, high) relative to next week's open
            target_low  = (next_low  - next_open) / next_open
            target_high = (next_high - next_open) / next_open
    """

    def __init__(self, records: list[tuple]):
        # records: list of (X_window, y_2)
        self.X = torch.tensor(
            np.array([r[0] for r in records], dtype=np.float32)
        )
        self.y = torch.tensor(
            np.array([r[1] for r in records], dtype=np.float32)
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def build_records(df: pd.DataFrame) -> list[tuple]:
    """Build sliding-window records from a featured weekly dataframe."""
    df = df.dropna(subset=FEATURE_COLS + ["open", "low", "high"]).reset_index(drop=True)
    feats = df[FEATURE_COLS].values.astype(np.float32)
    opens  = df["open"].values.astype(np.float32)
    lows   = df["low"].values.astype(np.float32)
    highs  = df["high"].values.astype(np.float32)

    # Clip extreme feature values
    feats = np.clip(feats, -10, 10)

    records = []
    n = len(feats)
    for i in range(n - WINDOW - 1):
        X = feats[i : i + WINDOW]
        next_open = opens[i + WINDOW]
        next_low  = lows[i + WINDOW]
        next_high = highs[i + WINDOW]
        if next_open <= 0:
            continue
        y = np.array([
            (next_low  - next_open) / next_open,
            (next_high - next_open) / next_open,
        ], dtype=np.float32)
        records.append((X, y))
    return records


# ---------------------------------------------------------------------------
# Model: BiLSTM + attention (same architecture as range_predictor.py)
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.score = nn.Linear(hidden * 2, 1)

    def forward(self, h):  # h: (B, T, hidden*2)
        weights = torch.softmax(self.score(h), dim=1)  # (B, T, 1)
        return (weights * h).sum(dim=1)                # (B, hidden*2)


class WeeklyRangeModel(nn.Module):
    def __init__(self, n_features: int = len(FEATURE_COLS),
                 hidden: int = HIDDEN, layers: int = LAYERS,
                 dropout: float = DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.attn = Attention(hidden)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 2),   # [rel_low, rel_high]
        )

    def forward(self, x):   # x: (B, T, F)
        h, _ = self.lstm(x)  # (B, T, hidden*2)
        ctx  = self.attn(h)  # (B, hidden*2)
        return self.head(ctx)  # (B, 2)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def pinball_loss(pred: torch.Tensor, target: torch.Tensor,
                 tau_low: float = 0.8, tau_high: float = 0.2) -> torch.Tensor:
    err_low  = target[:, 0] - pred[:, 0]
    err_high = target[:, 1] - pred[:, 1]
    loss_low  = torch.where(err_low  >= 0, tau_low  * err_low,  (tau_low  - 1) * err_low)
    loss_high = torch.where(err_high >= 0, tau_high * err_high, (tau_high - 1) * err_high)
    return (loss_low + loss_high).mean()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_data(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Load, resample, and feature-engineer all tickers."""
    stock_dfs = {}
    for ticker in tickers:
        path = os.path.join(DATA_DIR, f"{ticker}_history.csv")
        if not os.path.exists(path):
            print(f"  [Data] {ticker}: file not found, skipping")
            continue
        daily = pd.read_csv(path, parse_dates=["datetime"])
        weekly = resample_to_weekly(daily)
        featured = compute_features(weekly)
        valid = featured.dropna(subset=FEATURE_COLS)
        if len(valid) < WINDOW + 10:
            print(f"  [Data] {ticker}: only {len(valid)} clean weeks, skipping")
            continue
        stock_dfs[ticker.upper()] = featured
        print(f"  [Data] {ticker.upper()}: {len(valid)} weekly bars")
    return stock_dfs


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train(tickers: list[str] = TICKERS, epochs: int = EPOCHS):
    print(f"\n=== Vault 8 — Training (multi-stock, {len(tickers)} tickers) ===\n")
    stock_dfs = load_all_data(tickers)

    # Build records per stock, use 80/20 train/val split by time
    train_records, val_records = [], []
    val_by_ticker: dict[str, list] = {}

    for ticker, df in stock_dfs.items():
        recs = build_records(df)
        if not recs:
            continue
        split = int(len(recs) * 0.8)
        train_records.extend(recs[:split])
        val_records.extend(recs[split:])
        val_by_ticker[ticker] = recs[split:]

    print(f"\n  Train samples: {len(train_records)}  |  Val samples: {len(val_records)}")

    loader = DataLoader(WeeklyRangeDataset(train_records),
                        batch_size=BATCH, shuffle=True)
    val_ds = WeeklyRangeDataset(val_records)

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}\n")

    model = WeeklyRangeModel().to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            pred = model(X)
            loss = pinball_loss(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
        sched.step()

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                Xv = val_ds.X.to(device)
                yv = val_ds.y.to(device)
                vl = pinball_loss(model(Xv), yv).item()
            print(f"  Epoch {epoch:>3}/{epochs}  train={total_loss/len(loader):.4f}  val={vl:.4f}")
            if vl < best_val:
                best_val = vl
                torch.save(model.state_dict(), MODEL_PATH)
                print(f"           ↑ saved (best val={best_val:.4f})")

    print(f"\n  Training complete. Best val loss: {best_val:.4f}")
    print(f"  Model saved → {MODEL_PATH}")

    # Per-stock evaluation
    print(f"\n=== Per-stock validation performance ===\n")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    evaluate_by_ticker(model, device, val_by_ticker, stock_dfs)


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

def evaluate_by_ticker(model, device, val_by_ticker: dict, stock_dfs: dict):
    rows = []
    for ticker, recs in val_by_ticker.items():
        if not recs:
            continue
        ds = WeeklyRangeDataset(recs)
        with torch.no_grad():
            pred = model(ds.X.to(device)).cpu().numpy()
        actual = ds.y.numpy()

        # Capture ratio: how much of the oracle range does the model capture?
        # If you buy at pred_low and sell at pred_high (but constrained to actual range):
        pred_low  = pred[:, 0]
        pred_high = pred[:, 1]
        act_low   = actual[:, 0]
        act_high  = actual[:, 1]

        oracle_spread = act_high - act_low
        # Clamp predicted range to actual range (can't buy below actual low)
        captured_low  = np.maximum(pred_low,  act_low)
        captured_high = np.minimum(pred_high, act_high)
        captured = np.maximum(captured_high - captured_low, 0)
        capture_ratio = (captured / (oracle_spread + 1e-9)).mean() * 100

        # Direction accuracy: did pred_low < pred_high? (sane range)
        sane = (pred_high > pred_low).mean() * 100

        # MAE on low/high sides (in % terms)
        mae_low  = np.abs(pred_low  - act_low).mean()  * 100
        mae_high = np.abs(pred_high - act_high).mean() * 100

        rows.append({
            "Ticker":         ticker,
            "Val weeks":      len(recs),
            "Capture %":      round(capture_ratio, 1),
            "Sane range %":   round(sane, 1),
            "MAE low (%)":    round(mae_low, 2),
            "MAE high (%)":   round(mae_high, 2),
        })

    df = pd.DataFrame(rows).sort_values("Capture %", ascending=False)
    print(df.to_string(index=False))

    out = os.path.join(_HERE, "vault8_per_stock_eval.csv")
    df.to_csv(out, index=False)
    print(f"\n  Per-stock eval saved → {out}")
    return df


# ---------------------------------------------------------------------------
# Predict next week
# ---------------------------------------------------------------------------

def predict_next_week(ticker: str):
    path = os.path.join(DATA_DIR, f"{ticker.lower()}_history.csv")
    if not os.path.exists(path):
        print(f"No data for {ticker}")
        return

    daily = pd.read_csv(path, parse_dates=["datetime"])
    weekly = resample_to_weekly(daily)
    featured = compute_features(weekly).dropna(subset=FEATURE_COLS)

    if len(featured) < WINDOW:
        print(f"Not enough weekly data for {ticker}")
        return

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    model = WeeklyRangeModel().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    last_window = featured[FEATURE_COLS].values[-WINDOW:].astype(np.float32)
    last_window = np.clip(last_window, -10, 10)
    X = torch.tensor(last_window).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(X).cpu().numpy()[0]

    last_close = featured["close"].iloc[-1]
    last_date  = featured["date"].iloc[-1]
    pred_low   = last_close * (1 + pred[0])
    pred_high  = last_close * (1 + pred[1])
    pred_range = (pred_high - pred_low) / pred_low * 100

    print(f"\n  Vault 8 — Next-week range prediction for {ticker.upper()}")
    print(f"  Last close ({last_date.date()}): ${last_close:.2f}")
    print(f"  Predicted low:  ${pred_low:.2f}  ({pred[0]*100:+.1f}%)")
    print(f"  Predicted high: ${pred_high:.2f}  ({pred[1]*100:+.1f}%)")
    print(f"  Predicted range: {pred_range:.1f}%")
    print(f"  Strategy: BUY @ ${pred_low:.2f}  →  SELL @ ${pred_high:.2f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",   action="store_true", help="Train the model")
    parser.add_argument("--eval",    action="store_true", help="Evaluate saved model")
    parser.add_argument("--predict", metavar="TICKER",    help="Predict next week for ticker")
    parser.add_argument("--epochs",  type=int, default=EPOCHS)
    parser.add_argument("--tickers", nargs="+", default=TICKERS)
    args = parser.parse_args()

    if args.train:
        train(tickers=args.tickers, epochs=args.epochs)
    elif args.eval:
        if not os.path.exists(MODEL_PATH):
            print("No model found. Run --train first.")
            return
        stock_dfs = load_all_data(args.tickers)
        val_by_ticker = {}
        for ticker, df in stock_dfs.items():
            recs = build_records(df)
            split = int(len(recs) * 0.8)
            val_by_ticker[ticker] = recs[split:]
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model = WeeklyRangeModel().to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        evaluate_by_ticker(model, device, val_by_ticker, stock_dfs)
    elif args.predict:
        if not os.path.exists(MODEL_PATH):
            print("No model found. Run --train first.")
            return
        predict_next_week(args.predict)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
