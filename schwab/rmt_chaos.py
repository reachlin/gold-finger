"""
rmt_chaos.py — Random Matrix Theory "propagation of chaos" / herding index.

Motivation
----------
Deng-Hani-Ma's resolution of Hilbert's sixth problem (arXiv:2503.01800) shows
that a mean-field/statistical description of a many-body system is valid *iff*
propagation of chaos holds — micro-correlations stay negligible. Invert that
for markets: our statistical price predictors (TimesFM, Kronos, the Vault8
BiLSTM range model) are only in their domain of validity while the cross-
sectional correlation structure of the market is "chaotic" (noise-like).

Random Matrix Theory gives a clean, parameter-free test of that condition.
For N assets over T days, the eigenvalues of the sample correlation matrix of
pure-noise returns fall inside the Marchenko-Pastur bulk [λ-, λ+]. A genuine
market/herding mode shows up as the top eigenvalue *detaching* above λ+ and
absorbing a large share of total variance. High herding = chaos broken =
predictors outside their valid regime = widen uncertainty / stand down.

This module is a market-state SENSOR, not a stock picker: the universe it
measures need not be the universe traded. Run it over a broad, sector-balanced
set of names and use the resulting regime to gate the watchlist wheel trades.

References
----------
- Laloux, Cizeau, Bouchaud, Potters (1999) "Noise dressing of financial
  correlation matrices" — MP bulk / market mode.
- Kritzman, Li, Page, Rigobon (2010) "Principal Components as a Measure of
  Systemic Risk" — the Absorption Ratio and its standardized shift.

Run:
    python schwab/rmt_chaos.py                 # report on the blue-chip CSVs
    python schwab/rmt_chaos.py --window 150 --save
"""
import os
import glob
import argparse
import numpy as np
import pandas as pd

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# Blue-chip universe already on disk (download_bluechips.py). Sector-
# concentrated (mega-cap tech + Dow) — kept as a fallback / comparison.
BLUE_CHIPS = [
    "AAPL", "MSFT", "UNH", "GS", "HD", "MCD", "CAT", "CRM", "V", "AMGN",
    "HON", "AXP", "TRV", "JPM", "IBM", "JNJ", "WMT", "PG", "CVX", "MRK",
    "DIS", "NKE", "MMM", "KO", "BA", "CSCO", "VZ", "INTC", "DOW",
    "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AMD", "AVGO", "QCOM", "TXN",
    "LLY", "ABT", "TMO", "ABBV", "BRKB", "BAC", "WFC", "COST", "XOM",
]


def sensor_universe() -> list[str]:
    """Broad, GICS-sector-balanced set (~105 names) for a clean MP bulk.
    Falls back to BLUE_CHIPS if the sensor downloader module is unavailable."""
    try:
        from download_sensor_universe import all_tickers
        return all_tickers()
    except Exception:
        return BLUE_CHIPS


# ---------------------------------------------------------------------------
# Core RMT math
# ---------------------------------------------------------------------------

def marchenko_pastur_edges(q: float, sigma2: float = 1.0) -> tuple[float, float]:
    """
    Support edges [λ-, λ+] of the Marchenko-Pastur law for aspect ratio
    q = N/T and per-variable variance sigma2. For a correlation matrix of
    standardized returns sigma2 = 1 and the average eigenvalue is 1.
    """
    root = np.sqrt(q)
    lo = sigma2 * (1.0 - root) ** 2
    hi = sigma2 * (1.0 + root) ** 2
    return float(lo), float(hi)


def correlation_eigenvalues(window_returns: np.ndarray) -> np.ndarray:
    """
    Eigenvalues (descending) of the Pearson correlation matrix of a
    T x N block of returns. Columns with zero variance are dropped.
    """
    X = np.asarray(window_returns, dtype=float)
    std = X.std(axis=0)
    X = X[:, std > 0]
    if X.shape[1] == 0:
        return np.array([])
    C = np.corrcoef(X, rowvar=False)
    C = np.atleast_2d(C)
    eigs = np.linalg.eigvalsh(C)          # ascending, real (C symmetric)
    return np.clip(eigs[::-1], 0.0, None)  # descending, no tiny negatives


def effective_rank(eigs: np.ndarray) -> float:
    """exp(Shannon entropy of the normalized eigenvalue spectrum) — the
    effective number of independent modes. N for pure noise, →1 for one
    dominant mode."""
    p = eigs / eigs.sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def decompose(eigs: np.ndarray, N: int, T: int, top_k: int) -> dict:
    """Per-window RMT metrics. `top_k` is the number of leading eigenvalues
    counted in the Absorption Ratio."""
    q = N / T
    _, hi = marchenko_pastur_edges(q)
    lam1 = float(eigs[0])
    n_dev = int((eigs > hi).sum())

    # Bulk-adjusted upper edge: refit sigma2 to the noise part after removing
    # the deviating (signal) eigenvalues, so the market mode itself doesn't
    # inflate the bulk it is being compared against. sigma2 is the share of
    # total variance still carried by the bulk (Laloux et al.).
    sigma2_bulk = float(eigs[n_dev:].sum() / N) if 0 < n_dev < N else 1.0
    _, hi_adj = marchenko_pastur_edges(q, sigma2_bulk)

    return {
        "lambda_max":       lam1,
        "mp_upper":         hi,
        "mp_upper_adj":     hi_adj,
        "detachment":       lam1 / hi,                 # >1 = above noise band
        "market_mode_frac": lam1 / N,                  # herding: 1/N..1
        "absorption_ratio": float(eigs[:top_k].sum() / N),
        "n_deviating":      n_dev,                     # # genuine factors
        "effective_rank":   effective_rank(eigs),
    }


# ---------------------------------------------------------------------------
# Returns + rolling index
# ---------------------------------------------------------------------------

def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns; first row dropped."""
    return np.log(prices).diff().iloc[1:]


def rolling_chaos_index(rets: pd.DataFrame, window: int = 252,
                        top_k_frac: float = 0.2, step: int = 1,
                        min_assets: int = 10, max_gap: int = 5) -> pd.DataFrame:
    """
    Roll a `window`-day correlation matrix across `rets` and record the RMT
    metrics at each step. At each window a name is kept if it has data on all
    but at most `max_gap` days (isolated calendar mismatches / halts, filled
    with 0 return); names still short — i.e. not yet listed — are dropped, so
    N grows over time. N and T are reported so q = N/T is auditable.
    """
    dates = rets.index
    rows = []
    for end in range(window, len(dates) + 1, step):
        block = rets.iloc[end - window:end]
        cov = block.notna().sum()
        keep = cov[cov >= window - max_gap].index
        sub = block[keep].fillna(0.0)
        N, T = sub.shape[1], sub.shape[0]
        if N < min_assets:
            continue
        eigs = correlation_eigenvalues(sub.values)
        if eigs.size == 0:
            continue
        N = eigs.size                        # after zero-variance drop
        d = decompose(eigs, N, T, top_k=max(1, int(round(N * top_k_frac))))
        d["date"], d["N"], d["T"] = dates[end - 1], N, T
        rows.append(d)
    return pd.DataFrame(rows).set_index("date")


def absorption_shift(rets: pd.DataFrame, fast_window: int = 63,
                     slow_window: int = 252, top_k_frac: float = 0.2) -> pd.DataFrame:
    """
    Kritzman standardized shift of the Absorption Ratio:
        shift = (AR_fast - mean(AR_slow)) / std(AR_slow)
    A shift rising above ~ +1 has historically preceded drawdowns. Returns a
    frame with ar_fast, ar_slow, and shift aligned on the fast series' dates.
    """
    ar_fast = rolling_chaos_index(rets, window=fast_window,
                                  top_k_frac=top_k_frac)["absorption_ratio"]
    ar_slow = rolling_chaos_index(rets, window=slow_window,
                                  top_k_frac=top_k_frac)["absorption_ratio"]
    slow_mean = ar_slow.rolling(slow_window, min_periods=slow_window // 4).mean()
    slow_std  = ar_slow.rolling(slow_window, min_periods=slow_window // 4).std()
    df = pd.DataFrame({"ar_fast": ar_fast})
    df["ar_slow_mean"] = slow_mean.reindex(df.index).ffill()
    df["ar_slow_std"]  = slow_std.reindex(df.index).ffill()
    df["shift"] = (df["ar_fast"] - df["ar_slow_mean"]) / df["ar_slow_std"]
    return df.dropna()


# ---------------------------------------------------------------------------
# Lead-lag validation vs a reference series (e.g. VIX)
# ---------------------------------------------------------------------------

def lead_lag(sig: pd.Series, ref: pd.Series, max_lag: int = 20) -> dict:
    """
    Cross-correlate a signal against a reference over integer day lags.
    Convention: at lag k>0 we correlate sig(t) with ref(t+k), i.e. the signal
    LEADS the reference by k days. Returns per-lag correlations plus the lag
    of maximum absolute correlation.
    """
    s = sig.dropna()
    r = ref.reindex(s.index.union(ref.index)).astype(float)
    common = s.index.intersection(r.dropna().index)
    s = s.reindex(common)
    r = r.reindex(common)
    corrs = {}
    for k in range(-max_lag, max_lag + 1):
        rr = r.shift(-k)
        pair = pd.concat([s, rr], axis=1).dropna()
        if len(pair) > 30:
            corrs[k] = float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))
    if not corrs:
        return {"best_lag": None, "best_corr": float("nan"), "corrs": {}}
    best_lag = max(corrs, key=lambda k: abs(corrs[k]))
    return {"best_lag": best_lag, "best_corr": corrs[best_lag], "corrs": corrs}


# ---------------------------------------------------------------------------
# Forward-predictive test — the decision-relevant validation for a risk gate:
# do high-herding days precede worse forward drawdowns / higher forward vol?
# ---------------------------------------------------------------------------

def forward_stats(signal: pd.Series, spy_close: pd.Series,
                  horizon: int = 21, hi_q: float = 0.8) -> dict:
    """
    Split days by whether `signal` is in its top (1-hi_q) tail vs the rest,
    and compare the SPY forward `horizon`-day max drawdown and realized vol
    that FOLLOW each group. A useful risk gate makes the high-signal group's
    forward drawdown materially worse than the low group's.
    """
    spy = spy_close.reindex(signal.index.union(spy_close.index)).sort_index().ffill()
    logret = np.log(spy).diff()

    fwd_dd, fwd_vol = {}, {}
    for t in signal.index:
        loc = spy.index.get_indexer([t], method="ffill")[0]
        if loc < 0 or loc + horizon >= len(spy):
            continue
        path = spy.iloc[loc:loc + horizon + 1]
        fwd_dd[t]  = float(path.min() / path.iloc[0] - 1.0)          # ≤ 0
        fwd_vol[t] = float(logret.iloc[loc + 1:loc + horizon + 1].std()
                           * np.sqrt(252))
    dd  = pd.Series(fwd_dd)
    vol = pd.Series(fwd_vol)
    sig = signal.reindex(dd.index).dropna()
    dd, vol = dd.reindex(sig.index), vol.reindex(sig.index)

    thr  = sig.quantile(hi_q)
    hi   = sig >= thr
    return {
        "horizon": horizon,
        "n_hi": int(hi.sum()), "n_lo": int((~hi).sum()),
        "dd_hi":  float(dd[hi].mean()),  "dd_lo":  float(dd[~hi].mean()),
        "vol_hi": float(vol[hi].mean()), "vol_lo": float(vol[~hi].mean()),
        "dd_worst_hi": float(dd[hi].min()), "dd_worst_lo": float(dd[~hi].min()),
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _read_close(path: str) -> pd.Series:
    df = pd.read_csv(path)
    dcol = "datetime" if "datetime" in df.columns else df.columns[0]
    ccol = "close" if "close" in df.columns else ("Close" if "Close" in df.columns else None)
    if ccol is None:
        raise ValueError(f"no close column in {path}")
    s = pd.Series(df[ccol].values,
                  index=pd.to_datetime(df[dcol]).dt.tz_localize(None))
    return s[~s.index.duplicated(keep="last")].sort_index()


def load_universe(tickers: list[str], data_dir: str = _DATA_DIR) -> pd.DataFrame:
    """Wide close-price frame for `tickers` that have a *_history.csv.
    Missing names are skipped (reported by the caller via column set)."""
    cols = {}
    for t in tickers:
        path = os.path.join(data_dir, f"{t.lower()}_history.csv")
        if os.path.exists(path):
            try:
                cols[t] = _read_close(path)
            except Exception as exc:
                print(f"  [rmt] skip {t}: {exc}")
    if not cols:
        raise FileNotFoundError(f"no history CSVs found in {data_dir}")
    return pd.DataFrame(cols).sort_index()


def load_vix(data_dir: str = _DATA_DIR) -> pd.Series | None:
    path = os.path.join(data_dir, "vix_history.csv")
    if not os.path.exists(path):
        return None
    try:
        return _read_close(path)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

# Known US-equity stress episodes for eyeballing the index around them.
STRESS_WINDOWS = {
    "2018 Volmageddon":  ("2018-01-15", "2018-02-28"),
    "2018 Q4 selloff":   ("2018-10-01", "2018-12-31"),
    "2020 COVID crash":  ("2020-02-15", "2020-04-15"),
    "2022 bear onset":   ("2022-01-01", "2022-06-30"),
    "2025 tariff shock": ("2025-03-15", "2025-05-15"),
}


def _fmt(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def report(window: int = 252, save: bool = False, start: str = "2006-01-01",
           universe: str = "sensor"):
    tickers = sensor_universe() if universe == "sensor" else BLUE_CHIPS
    px = load_universe(tickers)
    rets = log_returns(px)
    print(f"Universe: {rets.shape[1]} names, "
          f"{rets.index.min().date()} → {rets.index.max().date()} "
          f"({rets.shape[0]} days)\n")

    idx_full = rolling_chaos_index(rets, window=window)
    shift_full = absorption_shift(rets)
    # Baseline stats use the N-stable modern era only: market_mode_frac is
    # mechanically larger at small N, and pre-2006 windows have far fewer
    # listed names, which would bias every percentile/median.
    idx   = idx_full[idx_full.index >= start]
    shift = shift_full[shift_full.index >= start]
    print(f"(baseline stats restricted to ≥ {start}: "
          f"N ranges {int(idx['N'].min())}–{int(idx['N'].max())})\n")

    latest = idx.iloc[-1]
    print(f"=== Latest ({idx.index[-1].date()}) — window {window}d, "
          f"N={int(latest['N'])}, q={latest['N']/latest['T']:.2f} ===")
    print(f"  market-mode frac (herding):  {latest['market_mode_frac']:.3f}   "
          f"(1/N={1/latest['N']:.3f} = pure chaos, 1.0 = total herding)")
    print(f"  absorption ratio (top {int(round(latest['N']*0.2))}):     "
          f"{latest['absorption_ratio']:.3f}")
    print(f"  top eigenvalue / MP edge:    {latest['detachment']:.2f}x   "
          f"(λmax={latest['lambda_max']:.1f}, MP+={latest['mp_upper']:.2f})")
    print(f"  # deviating modes:           {int(latest['n_deviating'])}")
    print(f"  effective rank:              {latest['effective_rank']:.1f} of "
          f"{int(latest['N'])}")
    if not shift.empty:
        print(f"  absorption-ratio shift:      {shift['shift'].iloc[-1]:+.2f}σ  "
              f"(> +1σ = herding building)")

    # Percentile of today's herding within its own history
    mmf = idx["market_mode_frac"]
    pct = (mmf < mmf.iloc[-1]).mean() * 100
    print(f"  herding percentile vs history: {pct:.0f}th\n")

    # Behavior around known stress windows
    print("=== Herding (market-mode frac) around known stress episodes ===")
    base_med = mmf.median()
    print(f"  full-history median: {base_med:.3f}")
    for label, (a, b) in STRESS_WINDOWS.items():
        seg = mmf.loc[(mmf.index >= a) & (mmf.index <= b)]
        if len(seg):
            print(f"  {label:20s} peak {seg.max():.3f}  "
                  f"(+{(seg.max()/base_med-1)*100:.0f}% vs median)")
        else:
            print(f"  {label:20s} — no data in window")

    # Lead-lag vs VIX
    vix = load_vix()
    if vix is not None:
        print("\n=== Lead-lag vs VIX (does herding lead the vol regime?) ===")
        vix_chg = np.log(vix).diff()
        for name, s in (("herding level", mmf),
                        ("herding shift", shift["shift"] if not shift.empty else None)):
            if s is None:
                continue
            ll = lead_lag(s.diff().dropna(), vix_chg.dropna(), max_lag=20)
            lead = ll["best_lag"]
            direction = ("leads VIX" if lead and lead > 0 else
                         "lags VIX" if lead and lead < 0 else "coincident")
            print(f"  {name:14s}: best lag {lead:+d}d ({direction}), "
                  f"corr {_fmt(ll['best_corr'])}")

        # Calibrate against the scanner's VIX thresholds (20, 30)
        print("\n=== Herding at the scanner's VIX regime thresholds ===")
        j = pd.concat([mmf.rename("mmf"), vix.rename("vix")], axis=1).dropna()
        for lo, hi, lbl in ((0, 20, "VIX<20  Reclamation"),
                            (20, 30, "VIX 20-30 Wasteland"),
                            (30, 999, "VIX>=30 Nuked")):
            seg = j[(j["vix"] >= lo) & (j["vix"] < hi)]
            if len(seg):
                print(f"  {lbl:22s}: herding median {seg['mmf'].median():.3f}  "
                      f"(n={len(seg)})")
    else:
        print("\n(no data/vix_history.csv — skipping VIX lead-lag)")
        j = None

    # The real test for a risk gate: does high herding precede worse forward
    # drawdowns? Compare SPY forward 21d moves after high- vs low-signal days.
    # SPY is loaded separately — it must NOT be in the eigen-universe (an ETF
    # is a linear combo of members and would fabricate a dominant mode).
    spy_df = load_universe(["SPY"])
    if not spy_df.empty:
        print("\n=== Forward predictiveness vs SPY (does herding precede pain?) ===")
        spy = spy_df["SPY"]
        for label, s in (("herding level", mmf),
                         ("herding shift", shift["shift"] if not shift.empty else None)):
            if s is None:
                continue
            fs = forward_stats(s, spy, horizon=21, hi_q=0.8)
            print(f"  {label:14s} (top 20% vs rest, next 21d):")
            print(f"      avg drawdown : {fs['dd_hi']*100:+.2f}%  vs "
                  f"{fs['dd_lo']*100:+.2f}%   (worst {fs['dd_worst_hi']*100:.1f}% "
                  f"vs {fs['dd_worst_lo']*100:.1f}%)")
            print(f"      avg fwd vol  : {fs['vol_hi']*100:5.1f}%  vs "
                  f"{fs['vol_lo']*100:5.1f}%   (n_hi={fs['n_hi']})")

        # THE decision test: incremental value over VIX. Within calm-VIX days
        # (VIX<20 — when the scanner is happily selling premium), does herding
        # still separate forward vol? If yes, it flags structural stress that
        # VIX is missing — real, orthogonal information worth gating on.
        if j is not None:
            calm = j[j["vix"] < 20].index
            hi_vix = j[j["vix"] >= 20].index
            print("\n=== Incremental value over VIX (does herding beat VIX?) ===")
            for lbl, dts in (("within VIX<20 (calm)", calm),
                             ("within VIX>=20 (already stressed)", hi_vix)):
                sub = mmf.reindex(dts).dropna()
                if len(sub) > 200:
                    fs = forward_stats(sub, spy, horizon=21, hi_q=0.8)
                    print(f"  {lbl:34s}: top-20% herding fwd vol "
                          f"{fs['vol_hi']*100:.1f}% vs {fs['vol_lo']*100:.1f}%  "
                          f"(n_hi={fs['n_hi']})")

    if save:
        out = os.path.join(_DATA_DIR, "rmt_chaos_index.csv")
        merged = idx.join(shift[["shift"]], how="left")
        merged.to_csv(out)
        print(f"\nSaved → {out}")
    return idx, shift


def main():
    ap = argparse.ArgumentParser(description="RMT herding/chaos index report")
    ap.add_argument("--window", type=int, default=252,
                    help="rolling correlation window in trading days")
    ap.add_argument("--save", action="store_true",
                    help="write data/rmt_chaos_index.csv")
    ap.add_argument("--universe", choices=("sensor", "bluechips"),
                    default="sensor", help="which name set to measure")
    args = ap.parse_args()
    report(window=args.window, save=args.save, universe=args.universe)


if __name__ == "__main__":
    main()
