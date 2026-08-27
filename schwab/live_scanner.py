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

# LightGBM + torch OpenMP deadlock guard — see real_overseer.py header.
# Also needed here for direct `python live_scanner.py` runs.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from trend_scanner import compute_indicators
from options_pricer import historical_vol
import options_ledger as ol
import assignment_risk
import timesfm_advisor
import wheel_router
import allocator
import cash_ledger as cl
from chain_quotes import requote_signal
import signal_confidence
from vault76.overseer import Overseer
from vault76.armory.raider import Raider
from vault76.armory.scavenger import Scavenger
from vault76.armory.medic import Medic, MEDIC_ETFS
from vault76.armory.hunter import Hunter, SECTOR_MAP

SCAN_INTERVAL_MIN = 5
MARKET_OPEN_ET    = 9
MARKET_CLOSE_ET   = 16
BUDGET_PCT_MIN    = 0.10    # 10% of capital at lowest confidence
BUDGET_PCT_MAX    = 0.20    # 20% of capital at highest confidence


def _equity_budget(confidence, capital: float) -> float:
    """Return dollar budget for an equity BUY: 10–20% of capital scaled by confidence."""
    pct = BUDGET_PCT_MIN + (min(max(confidence or 0, 0), 100) / 100) * (BUDGET_PCT_MAX - BUDGET_PCT_MIN)
    return round(capital * pct, 2)

SIGNAL_START = "===SIGNAL_START==="
SIGNAL_END   = "===SIGNAL_END==="
VERDICT_OK   = "===APPROVED==="
VERDICT_SKIP = "===SKIPPED==="

WATCHLIST = [
    "NVDA", "AMD", "AAPL", "AMZN",
    "META", "MSFT", "GOOGL",
    "IBM", "INTC", "KO",
    # Scavenger prime targets — high % time sideways
    "MMM", "PG", "XOM",
    # Top backtest candidates — wheel strategy
    "UNH",
]

PAPER_TRADES_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "paper_trades.json"
)
OPTION_LEDGER_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "paper_options_ledger.csv"
)
WHEEL_HOLDINGS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "paper_wheel_holdings.json"
)
FAST_RISKOFF_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "fast_riskoff.json"
)

# FinRL fast risk-off thresholds — imported from strategy_params so they stay
# in sync with the backtest. Edit strategy_params.py to tune them.
from strategy_params import (
    FAST_RISKOFF_DROP     as FAST_RISKOFF_THRESHOLD,
    FAST_RISKOFF_COOLDOWN as FAST_RISKOFF_DAYS,
    FAST_RISKOFF_LOOKBACK,
)


# ---------------------------------------------------------------------------
# Kronos range cache
# ---------------------------------------------------------------------------

def _load_kronos_cache(symbols: list[str]) -> dict:
    """
    Load the Kronos foundation model and run 30-day range predictions for all
    symbols. Returns a dict keyed by symbol:
      { "support": float, "resistance": float, "range_pct": float, "buf_pct": float }

    Cached for the trading session — predictions are 30-day so no need to refresh
    every scan cycle. Returns {} if Kronos is unavailable (scanner still works).
    """
    try:
        import sys as _sys
        _sys.path.insert(0, os.environ.get("KRONOS_PATH",
                                           "/Users/lincai/dev/private/Kronos"))
        from model import Kronos, KronosTokenizer, KronosPredictor
        import kronos_range as kr
        print("Loading Kronos model (this takes ~30s on first run)...", flush=True)
        tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        model     = Kronos.from_pretrained("NeoQuasar/Kronos-small")
        predictor = KronosPredictor(model, tokenizer)
        print(f"  Kronos ready on {predictor.device}", flush=True)
        cache = {}
        for sym in symbols:
            path = os.path.join(os.path.dirname(__file__), "..", "data",
                                f"{sym.lower()}_history.csv")
            if not os.path.exists(path):
                continue
            try:
                import pandas as _pd
                df = _pd.read_csv(path)
                result = kr.predict_range(df, predictor)
                strike = result["current_price"] * (1 - 0.05)
                buf    = (strike - result["support"]) / result["current_price"] * 100
                cache[sym] = {
                    "support":    result["support"],
                    "resistance": result["resistance"],
                    "range_pct":  result["range_pct"],
                    "buf_pct":    buf,
                }
            except Exception as e:
                print(f"  [Kronos] {sym}: {e}")
        print(f"  Kronos cache ready: {len(cache)}/{len(symbols)} symbols", flush=True)
        return cache
    except Exception as exc:
        print(f"  [Kronos] unavailable — scanner will run without range filter ({exc})")
        return {}


def _kronos_advisory(s: dict, kronos_cache: dict) -> tuple[bool, str]:
    """
    Return (warn, message) for a SELL_PUT signal vs Kronos predicted support.

    warn=True  → our 5% OTM strike is above the predicted floor (assignment risk)
    warn=False → strike stays below predicted support (safer entry)
    Returns ("", False) for non-SELL_PUT signals or missing cache entry.
    """
    if s.get("signal") != "SELL_PUT":
        return False, ""
    sym   = s["symbol"]
    entry = kronos_cache.get(sym)
    if not entry:
        return False, ""
    support    = entry["support"]
    resistance = entry["resistance"]
    range_pct  = entry["range_pct"]
    buf        = entry["buf_pct"]
    strike     = float(s.get("strike", 0))
    warn       = strike > support     # strike above predicted floor = assignment risk

    flag = "WARN" if warn else "OK"
    msg  = (f"Kronos 30d range: support ${support:.2f}  resist ${resistance:.2f}"
            f"  ({range_pct:.1f}%)  strike buf {buf:+.1f}%  [{flag}]")
    return warn, msg


# ---------------------------------------------------------------------------
# Sector ETF cache — for Hunter's three-pillar sector-strength check
# ---------------------------------------------------------------------------

_sector_cache: dict = {}   # etf → pd.DataFrame with indicators


def _load_sector_cache() -> None:
    """Load pre-downloaded sector ETF CSVs into memory once at startup."""
    global _sector_cache
    etfs = set(SECTOR_MAP.values())
    for etf in etfs:
        path = os.path.join(os.path.dirname(__file__), "..", "data",
                            f"{etf.lower()}_history.csv")
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path, parse_dates=["datetime"])
            df = compute_indicators(df).dropna().reset_index(drop=True)
            _sector_cache[etf] = df
        except Exception as exc:
            print(f"  [sector] {etf}: {exc}")


def _sector_above_ema50(etf: str) -> bool:
    """True if the sector ETF's latest close ≥ EMA50 (from cached CSV)."""
    df = _sector_cache.get(etf)
    if df is None or df.empty:
        return True   # no data — don't block
    row = df.iloc[-1]
    return bool(row["close"] >= row["ema50"])


# ---------------------------------------------------------------------------
# Market hours
# ---------------------------------------------------------------------------

def _now_et():
    """Current datetime in US/Eastern — DST-aware, timezone-independent."""
    from zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo("America/New_York"))


def _display_now() -> str:
    """
    Human-readable timestamp for banners/signals/Slack: market time (ET)
    plus the machine's current local time. The local zone is read from the
    system at call time, so it follows the machine wherever it travels —
    nothing is hardcoded. Ledger/state files always store pure ET.
    """
    et    = _now_et()
    local = datetime.now().astimezone()
    text  = et.strftime("%Y-%m-%d %H:%M:%S ET")
    if local.utcoffset() != et.utcoffset():
        text += local.strftime(" (local %H:%M %Z)")
    return text


def _et_hour() -> float:
    now = _now_et()
    return now.hour + now.minute / 60


def _is_weekday() -> bool:
    return _now_et().weekday() < 5   # Mon=0 … Fri=4 in ET, not UTC


def _is_market_hours() -> bool:
    return _is_weekday() and MARKET_OPEN_ET <= _et_hour() < MARKET_CLOSE_ET


def _market_closed_for_today() -> bool:
    if not _is_weekday():
        return True
    return _et_hour() >= MARKET_CLOSE_ET


def _sleep_until_market_open(now_et):
    """Sleep until 09:00 ET the next trading day, then return."""
    next_open = now_et.replace(hour=MARKET_OPEN_ET, minute=0, second=0, microsecond=0)
    if next_open <= now_et:
        next_open += timedelta(days=1)
    while next_open.weekday() >= 5:
        next_open += timedelta(days=1)
    secs = (next_open - now_et).total_seconds()
    print(f"  Market closed. Sleeping {secs/3600:.1f}h until {next_open.strftime('%Y-%m-%d %H:%M ET')}.")
    time.sleep(secs)


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
        exc_str = str(exc).lower()
        if "invalid_grant" in exc_str or "401" in exc_str or "unauthorized" in exc_str:
            msg = (f":rotating_light: *Schwab auth expired* (`{symbol}` fetch: "
                   f"`{exc}`). Re-auth required — run `schwab.auth.client_from_manual_flow`.")
            print(f"    [AUTH ERROR] {exc}")
            _send_slack(msg)
        else:
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
_hunter    = Hunter()
_medic     = Medic()

# The Medic: crisis accumulation — budget per ETF per NUKED_ZONE episode,
# positions persisted like router holds
MEDIC_BUDGET_PER_ETF = 600
MEDIC_HOLDS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "paper_medic_holdings.json")

# LGBM move-risk models — trained at startup in main(), advisory only.
# _assign_models: P(>5% drop in 30d) for SELL_PUT/BUY; _upside_models:
# P(>8% rally in 30d) for SELL_CALL (called-away risk).
_assign_models: dict = {}
_upside_models: dict = {}

# TimesFM zero-shot 30-day SMA5 forecasts — cached at startup, advisory only
_timesfm_cache: dict = {}

# Wheel-vs-hold router: hold mode persists across restarts; proposals are
# deduped per (date, symbol, action) so a rejected one doesn't re-fire all day
ROUTER_HOLDS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "paper_router_holds.json")
CASH_LEDGER_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "cash_ledger.csv")
_router_proposed: set = set()
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
_SCAN_COUNTER_PATH = os.path.join(DATA_DIR, "scan_counter.json")

_cash_ledger = None   # initialised in main() once starting_capital is known


def _load_global_scan() -> int:
    try:
        import json as _j
        return int(_j.load(open(_SCAN_COUNTER_PATH)).get("global_scan", 0))
    except Exception:
        return 0


def _save_global_scan(n: int) -> None:
    try:
        import json as _j
        with open(_SCAN_COUNTER_PATH, "w") as f:
            _j.dump({"global_scan": n}, f)
    except Exception:
        pass

# ---------------------------------------------------------------------------
# AutoOverseer hook — set by real_overseer.py to replace interactive input
# ---------------------------------------------------------------------------

_decision_fn       = None   # callable(signal: dict) -> "y" | "n" | "q"
_scan_hook_fn      = None   # callable() — called at start of each scan cycle
_current_scan_signals: list = []  # this scan's signals — AutoOverseer reads
                                  # them to show each signal its competitors
_current_portfolio = None   # populated in main() after PaperPortfolio init
_current_kronos_cache: dict = {}  # populated in main() after _load_kronos_cache()
_current_client    = None   # populated in main() after Schwab auth
_slack_prefix      = ""     # e.g. "(deepseek) " — set by real_overseer
_schwab_cash       = None   # Schwab cashBalance (total, incl. locked collateral)
_schwab_available  = None   # Schwab availableFunds (free cash after collateral deduction)
_schwab_pending_deposits = None  # unsettled ACH deposits — tradeable per Schwab but
                                 # fenced from the overseer until settled (set by reconcile)


def set_decision_fn(fn):
    """Replace the interactive y/n prompt with an automated decision function."""
    global _decision_fn
    _decision_fn = fn


def set_scan_hook_fn(fn):
    """Register a callback fired at the start of each scan (e.g. pending-order check)."""
    global _scan_hook_fn
    _scan_hook_fn = fn


def _fetch_regime_and_spy(client) -> tuple[str, "pd.DataFrame | None"]:
    """Fetch SPY + VIX, return (regime, spy_df). spy_df is reused for fast risk-off + ranking."""
    spy_df = _fetch_history(client, "SPY")
    if spy_df is None or len(spy_df) < 60:
        return Overseer.RECLAMATION, spy_df
    try:
        from signal_verifier import fetch_vix
        vix = fetch_vix()
    except Exception:
        vix = 20.0
    return _overseer.classify(spy_df, vix=vix), spy_df


def _fetch_regime(client) -> str:
    """Thin wrapper kept for _print_startup compatibility."""
    regime, _ = _fetch_regime_and_spy(client)
    return regime


def _check_fast_riskoff(spy_df: "pd.DataFrame | None") -> tuple[bool, str]:
    """
    FinRL-style fast risk-off gate (Enhancement #1).

    Checks two conditions in order:
      1. Are we still within FAST_RISKOFF_DAYS cooldown from a past trigger?
         State is persisted in FAST_RISKOFF_PATH (JSON) so it survives restarts.
      2. Did SPY just drop more than FAST_RISKOFF_THRESHOLD over FAST_RISKOFF_LOOKBACK days?
         If yes, write a new trigger file and start the cooldown.

    Returns (is_active, human-readable message).
    When is_active=True, all new SELL_PUT signals are suppressed.

    Tune thresholds in strategy_params.py:
      FAST_RISKOFF_DROP     — e.g. -0.02 for more sensitive, -0.05 for less
      FAST_RISKOFF_LOOKBACK — days to measure the drop over (default 3)
      FAST_RISKOFF_COOLDOWN — days to block new puts after trigger (default 10)
    """
    import json

    # Check if still in cooldown from a previous trigger
    if os.path.exists(FAST_RISKOFF_PATH):
        with open(FAST_RISKOFF_PATH) as f:
            state = json.load(f)
        triggered = datetime.fromisoformat(state["triggered_at"]).date()
        elapsed   = (_now_et().date() - triggered).days
        if elapsed < FAST_RISKOFF_DAYS:
            days_left = FAST_RISKOFF_DAYS - elapsed
            return True, (f"fast risk-off active — {days_left}d left "
                          f"(triggered {triggered}, SPY {state['spy_return_3d']:.1%} "
                          f"in {FAST_RISKOFF_LOOKBACK}d)")

    # Measure SPY return over the last FAST_RISKOFF_LOOKBACK trading days
    lookback = FAST_RISKOFF_LOOKBACK + 1   # +1 because we compare close[0] vs close[-1]
    if spy_df is None or len(spy_df) < lookback:
        return False, ""
    closes = spy_df["close"].values
    ret = (closes[-1] - closes[-lookback]) / closes[-lookback]

    if ret <= FAST_RISKOFF_THRESHOLD:
        # Trigger: write state file so cooldown persists across scanner restarts
        state = {"triggered_at": _now_et().isoformat(), "spy_return_3d": float(ret)}
        with open(FAST_RISKOFF_PATH, "w") as f:
            json.dump(state, f, indent=2)
        return True, (f"fast risk-off TRIGGERED — SPY {ret:.1%} in {FAST_RISKOFF_LOOKBACK}d "
                      f"(threshold {FAST_RISKOFF_THRESHOLD:.0%})")

    return False, ""




def _scan_all(client, regime: str = Overseer.RECLAMATION,
              spy_df: "pd.DataFrame | None" = None) -> list[dict]:
    signals  = []
    holdings = ol.load_holdings(WHEEL_HOLDINGS_PATH)
    for symbol in WATCHLIST:
        df_ind = _fetch_with_indicators(client, symbol)
        if df_ind is None:
            continue
        # TimesFM advisory: zero-shot 30-day SMA5 forecast, attached to every
        # signal on this symbol as a second foundation-model view next to Kronos
        tfm_pct = timesfm_advisor.advise(_timesfm_cache, symbol)

        # Wheel-vs-hold router: mechanical proposal at ROUTER_TAU, the LLM
        # disposes. While a hold is active the Scavenger is suppressed for
        # this symbol (that's the point — don't cap the runner).
        router_holds = wheel_router.load_holds(ROUTER_HOLDS_PATH)
        in_hold      = symbol in router_holds
        action       = wheel_router.live_route(tfm_pct, in_hold)
        close        = float(df_ind.iloc[-1]["close"])
        dedup_key    = (str(pd.Timestamp.now().date()), symbol, action)
        if action is not None and dedup_key not in _router_proposed:
            _router_proposed.add(dedup_key)
            sig = {"symbol": symbol, "signal": action, "card": "router",
                   "close": round(close, 2), "timesfm_30d_pct": tfm_pct,
                   "router_tau": wheel_router.ROUTER_TAU}
            if action == "HOLD_SHARES":
                _cap = _current_portfolio.cash if _current_portfolio else 30_000.0
                sig["shares"] = max(1, int(_equity_budget(sig.get("confidence", 50), _cap) / close))
                sig["reason"] = (f"Router: TimesFM 30d forecast {tfm_pct:+.1f}% >= "
                                 f"{wheel_router.ROUTER_TAU}% — hold shares "
                                 f"uncapped instead of wheeling")
            else:
                pos = router_holds[symbol]
                sig["shares"]     = pos["shares"]
                sig["hold_entry"] = pos["entry"]
                sig["hold_date"]  = pos["date"]
                sig["unrealized"] = round((close - pos["entry"]) * pos["shares"], 2)
                sig["reason"] = (f"Router: TimesFM 30d forecast {tfm_pct:+.1f}% < "
                                 f"{wheel_router.ROUTER_TAU}% — resume wheeling")
            signals.append(sig)

        # The Raider: buy pullbacks in uptrends
        r = _raider.scan(symbol, df_ind, regime=regime)
        if r["signal"] != "NONE":
            if tfm_pct is not None:
                r["timesfm_30d_pct"] = tfm_pct
            # LGBM advisory: high drop risk → the "pullback" may be a breakdown
            risk = assignment_risk.advise(_assign_models, symbol, df_ind)
            if risk is not None:
                r["drop_risk_pct"] = risk
                auc = assignment_risk.model_auc(_assign_models, symbol)
                if auc is not None:
                    r["model_auc"] = auc
            r["confidence"] = signal_confidence.score(r)
            signals.append(r)
        # The Hunter: buy calls on VCP momentum breakouts.
        # Only fires in RECLAMATION; sector-strength is advisory (not a gate).
        if regime == Overseer.RECLAMATION:
            h = _hunter.scan(symbol, df_ind, regime=regime)
            if h["signal"] != "NONE":
                if tfm_pct is not None:
                    h["timesfm_30d_pct"] = tfm_pct
                sector_etf = SECTOR_MAP.get(symbol)
                h["sector_etf"] = sector_etf
                h["sector_ok"]  = _sector_above_ema50(sector_etf) if sector_etf else True
                h["confidence"] = signal_confidence.score(h)
                signals.append(h)

        # The Scavenger: sell options on sideways stocks.
        # Model signals are re-quoted against the real Schwab chain — real
        # strike/premium/IV/delta, or dropped if the real premium is too thin.
        # Suppressed while a router hold is active on this symbol.
        s = (_scavenger.scan(symbol, df_ind, regime=regime)
             if not in_hold else {"signal": "NONE"})
        if s["signal"] != "NONE":
            s = requote_signal(client, s)
            if s is not None:
                if tfm_pct is not None:
                    s["timesfm_30d_pct"] = tfm_pct
                # LGBM advisory: P(>5% drop within 30 trading days).
                # Never gates — shown to Claude/LLM next to Kronos.
                risk = assignment_risk.advise(_assign_models, symbol, df_ind)
                if risk is not None:
                    s["assign_risk_pct"] = risk
                    auc = assignment_risk.model_auc(_assign_models, symbol)
                    if auc is not None:
                        s["model_auc"] = auc
                kronos_buf = _current_kronos_cache.get(symbol, {}).get("buf_pct")
                s["confidence"] = signal_confidence.score(s, kronos_buf=kronos_buf)
                signals.append(s)
        # Wheel phase 2: assigned shares → sell covered calls
        # (also suppressed while a router hold is active — don't cap the run)
        if symbol in holdings and not in_hold:
            c = _scavenger.scan(symbol, df_ind, regime=regime,
                                cost_basis=holdings[symbol]["cost_basis"])
            if c["signal"] != "NONE":
                c = requote_signal(client, c)
                if c is not None:
                    if tfm_pct is not None:
                        c["timesfm_30d_pct"] = tfm_pct
                    # LGBM advisory: P(>8% rally within 30 trading days) —
                    # how likely the shares get called away at the strike.
                    up = assignment_risk.advise(_upside_models, symbol, df_ind)
                    if up is not None:
                        c["called_away_pct"] = up
                        auc = assignment_risk.model_auc(_upside_models, symbol)
                        if auc is not None:
                            c["model_auc"] = auc
                    c["confidence"] = signal_confidence.score(c)
                    signals.append(c)

    # The Medic: crisis accumulation of dividend ETFs. Buys only in
    # NUKED_ZONE; sells only on RECLAMATION recovery — so the ETFs are
    # only fetched when the blast radius is active or a position is open.
    medic_holds = wheel_router.load_holds(MEDIC_HOLDS_PATH)
    if regime == Overseer.NUKED_ZONE or medic_holds:
        today = str(pd.Timestamp.now().date())
        for sym in MEDIC_ETFS:
            df_etf = _fetch_with_indicators(client, sym)
            if df_etf is None:
                continue
            res = _medic.scan(sym, df_etf, regime=regime,
                              holding=sym in medic_holds)
            if res["signal"] == "NONE":
                continue
            if (today, sym, res["signal"]) in _router_proposed:
                continue
            _router_proposed.add((today, sym, res["signal"]))
            close = float(df_etf.iloc[-1]["close"])
            if res["signal"] == "BUY_ETF":
                res["shares"] = max(1, int(MEDIC_BUDGET_PER_ETF / close))
            else:
                pos = medic_holds[sym]
                res["shares"]     = pos["shares"]
                res["hold_entry"] = pos["entry"]
                res["hold_date"]  = pos["date"]
                res["unrealized"] = round((close - pos["entry"])
                                          * pos["shares"], 2)
            signals.append(res)
    return signals


def _make_quote_fetcher(client):
    """Returns fetch_quote(symbol) → {"close", "hv"} for the ledger processor."""
    def fetch_quote(symbol: str) -> dict:
        df_ind = _fetch_with_indicators(client, symbol)
        if df_ind is None:
            raise RuntimeError(f"No data for {symbol}")
        return {
            "close": float(df_ind.iloc[-1]["close"]),
            "hv":    historical_vol(df_ind["close"]),
        }
    return fetch_quote


def _process_option_ledger(client, paper: bool = True) -> list[dict]:
    """
    Paper mode: settle expiries/assignments/early-exits in the simulated ledger.
    Semi-auto/live (paper=False): skip — AutoOverseer.check_pending_orders handles
    profit-target detection and auto-places BUY_TO_CLOSE orders via Schwab API.
    """
    if not paper:
        return []

    rows   = ol.read_rows(OPTION_LEDGER_PATH)
    opens  = {ol.row_key(r): r for r in ol.open_options(rows)}
    events = ol.process_expirations(
        OPTION_LEDGER_PATH, WHEEL_HOLDINGS_PATH,
        _make_quote_fetcher(client),
    )
    if not events:
        return []

    print(f"\n  [WHEEL] {len(events)} option settlement(s):")
    slack_lines = ["*VAULT 76 — Wheel settlements*"]
    for ev in events:
        icon = "+" if ev["pnl"] >= 0 else "-"
        line = (f"{ev['symbol']:<6} {ev['action']:<13} "
                f"{icon}${abs(ev['pnl']):.2f}  {ev['detail']}")
        print(f"    {line}")
        slack_lines.append(f"  • {line}")
        if _cash_ledger:
            opening = opens.get(ev.get("symbol", ""))
            cl.from_options_event(_cash_ledger, ev, opening)
    _send_slack("\n".join(slack_lines))
    return events


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

def _print_signal(s: dict, paper: bool = False, kronos_cache: dict | None = None):
    now      = _display_now()
    mode_tag = " [PAPER]" if paper else ""
    sig      = s["signal"]
    card     = s.get("card", "?")

    print(f"\n{SIGNAL_START}")
    print(f"TIME:    {now}{mode_tag}")
    print(f"SYMBOL:  {s['symbol']}  [{card.upper()}]")
    print(f"ACTION:  {sig}")
    if s.get("confidence") is not None:
        conf = s["confidence"]
        t    = signal_confidence.tier(conf)
        bar  = "█" * (conf // 10) + "░" * (10 - conf // 10)
        print(f"CONF:    {conf:3d}/100  [{t:<4}]  {bar}")
    print(f"CLOSE:   ${s.get('close', '?')}")

    if sig == "BUY":
        print(f"ENTRY:   ${s['entry']}")
        print(f"TARGET:  ${s['target']}")
        print(f"STOP:    ${s['stop']}")
        print(f"RSI:     {s.get('rsi', '?')}  ADX: {s.get('adx', '?')}")
        print(f"EMA20:   ${s.get('ema20', '?')}  EMA50: ${s.get('ema50', '?')}")
        if s.get("drop_risk_pct") is not None:
            print(f"LGBM:    {s['drop_risk_pct']}% chance of >5% drop "
                  f"within 30 trading days (advisory"
                  + (f", AUC {s['model_auc']}" if s.get("model_auc") else "")
                  + ")")

    elif sig == "SELL_PUT":
        collateral = float(s.get("strike", 0)) * 100
        print(f"STRIKE:  ${s['strike']}  ({s.get('otm_pct', '5')}% OTM)")
        print(f"PREMIUM: ${s['premium']}/sh  (${s['premium']*100:.0f}/contract)  "
              f"+{s['premium_pct']:.2f}% yield")
        print(f"DTE:     {s['dte']} days")
        print(f"COLLAT:  ${collateral:,.0f}/contract (cash-secured, no margin)")
        print(f"MAX LOSS:${s.get('max_loss', '?')}/contract if assigned")
        print(f"HV:      {s.get('hv', '?')}%  ADX: {s.get('adx', '?')}")
        if s.get("quote_source") == "schwab_chain":
            print(f"QUOTE:   real chain — exp {s.get('expiry', '?')}  "
                  f"IV {s.get('iv', '?')}%  delta {s.get('delta', '?')}  "
                  f"bid/ask ${s.get('bid', '?')}/${s.get('ask', '?')}")
        else:
            print(f"QUOTE:   model (Black-Scholes on HV) — chain unavailable")
        if s.get("assign_risk_pct") is not None:
            print(f"LGBM:    {s['assign_risk_pct']}% chance of >5% drop "
                  f"within 30 trading days (advisory"
                  + (f", AUC {s['model_auc']}" if s.get("model_auc") else "")
                  + ")")
        if kronos_cache is not None:
            _, k_msg = _kronos_advisory(s, kronos_cache)
            if k_msg:
                print(f"KRONOS:  {k_msg}")

    elif sig == "SELL_CALL":
        print(f"STRIKE:  ${s['strike']}  ({s.get('otm_pct', '8')}% OTM)")
        print(f"PREMIUM: ${s['premium']}/sh  (${s['premium']*100:.0f}/contract)  "
              f"+{s['premium_pct']:.2f}% yield")
        print(f"COST BASIS: ${s.get('cost_basis', '?')}")
        print(f"DTE:     {s['dte']} days")
        print(f"MAX GAIN:${s.get('max_gain', '?')}/contract if called away")
        print(f"HV:      {s.get('hv', '?')}%  ADX: {s.get('adx', '?')}")
        if s.get("quote_source") == "schwab_chain":
            print(f"QUOTE:   real chain — exp {s.get('expiry', '?')}  "
                  f"IV {s.get('iv', '?')}%  delta {s.get('delta', '?')}  "
                  f"bid/ask ${s.get('bid', '?')}/${s.get('ask', '?')}")
        else:
            print(f"QUOTE:   model (Black-Scholes on HV) — chain unavailable")
        if s.get("called_away_pct") is not None:
            print(f"LGBM:    {s['called_away_pct']}% chance of >8% rally "
                  f"within 30 trading days — called away (advisory"
                  + (f", AUC {s['model_auc']}" if s.get("model_auc") else "")
                  + ")")

    elif sig == "BUY_CALL":
        sector_flag = ""
        if s.get("sector_etf"):
            ok = s.get("sector_ok", True)
            sector_flag = f"  sector {s['sector_etf']} {'✓' if ok else '⚠ BELOW EMA50'}"
        print(f"STRIKE:  ${s['strike']}  ATM call  {s.get('dte', 35)} DTE{sector_flag}")
        print(f"PREMIUM: ${s['premium']}/sh  (${s.get('cost_per_ct', s['premium']*100):.0f}/contract)  "
              f"{s['premium_pct']:.1f}% of close")
        print(f"EXIT T1: ${s['exit_25pct']}/sh (+50%) — sell 25% of position")
        print(f"EXIT T2: ${s['exit_50pct']}/sh (+100%) — sell 50% of remaining")
        print(f"STOP:    ${s['stop_premium']}/sh (-50% on premium)")
        print(f"VCP:     tightest base {s.get('vcp_tight_pct', '?')}%  "
              f"breakout vol ×{s.get('breakout_vol', '?')}")
        print(f"HV:      {s.get('hv', '?')}%  ADX: {s.get('adx', '?')}  RSI: {s.get('rsi', '?')}")
        print(f"QUOTE:   model (Black-Scholes on HV) — verify real chain before entry")

    elif sig == "HOLD_SHARES":
        print(f"SHARES:  {s['shares']}  (≈${s['shares'] * s.get('close', 0):,.0f})")
        print(f"ROUTER:  TimesFM 30d {s['timesfm_30d_pct']:+.1f}%  "
              f"(τ = {s.get('router_tau', '?')}%)")
        print(f"POLICY:  backtested wheel-vs-hold router — default APPROVE; "
              f"veto only on context (earnings/halt/macro)")

    elif sig == "RESUME_WHEEL":
        print(f"SHARES:  {s['shares']}  held since {s.get('hold_date', '?')} "
              f"@ ${s.get('hold_entry', '?')}")
        print(f"UNREAL:  ${s.get('unrealized', 0):+,.2f}")
        print(f"ROUTER:  TimesFM 30d {s['timesfm_30d_pct']:+.1f}%  "
              f"(τ = {s.get('router_tau', '?')}%)")

    elif sig == "BUY_ETF":
        print(f"SHARES:  {s['shares']}  (≈${s['shares'] * s.get('close', 0):,.0f})")
        print(f"POLICY:  Medic crisis buy — backtested 15-16/19 winning "
              f"episodes; default APPROVE, veto only on context")

    elif sig == "SELL_ETF":
        print(f"SHARES:  {s['shares']}  held since {s.get('hold_date', '?')} "
              f"@ ${s.get('hold_entry', '?')}")
        print(f"UNREAL:  ${s.get('unrealized', 0):+,.2f}")

    if sig not in ("HOLD_SHARES", "RESUME_WHEEL", "BUY_ETF", "SELL_ETF") \
            and s.get("timesfm_30d_pct") is not None:
        print(f"TIMESFM: 30d SMA5 forecast {s['timesfm_30d_pct']:+.1f}% "
              f"(zero-shot advisory)")
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
# Options ledger (paper trading — CSV, one row per trade)
# ---------------------------------------------------------------------------

def _log_option_trade(s: dict, verdict: str):
    """Append one row to the CSV options ledger."""
    now = _now_et().strftime("%Y-%m-%d %H:%M:%S")
    ol.append_row(OPTION_LEDGER_PATH, {
        "date":        now,
        "symbol":      s["symbol"],
        "signal":      s["signal"],
        "close":       s.get("close", ""),
        "strike":      s.get("strike", ""),
        "premium_sh":  s.get("premium", ""),
        "premium_ct":  round(s.get("premium", 0) * 100, 2),
        "premium_pct": s.get("premium_pct", ""),
        "dte":         s.get("dte", ""),
        "hv":          s.get("hv", ""),
        "adx":         s.get("adx", ""),
        "confidence":  s.get("confidence", ""),
        "regime":      s.get("regime", ""),
        "verdict":     verdict,
        "reason":      s.get("reason", ""),
    })


# ---------------------------------------------------------------------------
# Collateral / budget check (no naked contracts, no margin)
# ---------------------------------------------------------------------------

def _committed_collateral() -> float:
    """
    Cash locked by short puts in the options ledger: open puts plus assigned
    puts whose shares are still held. See options_ledger.committed_collateral.
    """
    return ol.committed_collateral(ol.read_rows(OPTION_LEDGER_PATH))


_PENDING_ORDERS_PATH = os.path.join(DATA_DIR, "pending_orders.json")


def _pending_collateral() -> float:
    """
    Cash locked by options orders notified via Slack but not yet confirmed
    filled (pending_orders.json).  Must be counted to prevent overdraft when
    a second signal arrives before the first is filled.
    SELL_PUT  → strike × 100 per contract (cash-secured)
    BUY_CALL  → limit × 100 per contract  (debit)
    SELL_CALL → 0 (covered by shares the user already holds)
    """
    if not os.path.exists(_PENDING_ORDERS_PATH):
        return 0.0
    try:
        import json as _json
        with open(_PENDING_ORDERS_PATH) as _f:
            orders = _json.load(_f)
    except Exception:
        return 0.0
    total = 0.0
    for o in orders:
        sig = o.get("signal", "")
        if sig == "SELL_PUT":
            total += float(o.get("strike", 0)) * 100
        elif sig == "BUY_CALL":
            total += float(o.get("limit", o.get("premium", 0))) * 100
    return total


def _collateral_required(s: dict) -> float:
    """Return cash required for a signal.
    SELL_PUT: strike × 100 (collateral locked).
    BUY_CALL: cost_per_ct (debit paid upfront).
    Other: 0.
    """
    if s.get("signal") == "SELL_PUT":
        return float(s.get("strike", 0)) * 100
    if s.get("signal") == "BUY_CALL":
        return float(s.get("cost_per_ct", 0))
    return 0.0


def _budget_check(s: dict, portfolio_cash: float) -> tuple[bool, str]:
    """
    Returns (ok, message).
    ok=False means auto-skip — not enough cash to cover without margin.
    """
    required = _collateral_required(s)
    if required == 0:
        return True, ""
    committed = _committed_collateral()
    pending   = _pending_collateral()
    ach       = _schwab_pending_deposits or 0    # unsettled ACH — fenced from trading
    available = portfolio_cash - committed - pending - ach
    if required > available:
        extra = ""
        if pending: extra += f" − ${pending:,.0f} order-pending"
        if ach:     extra += f" − ${ach:,.0f} unsettled-ACH"
        msg = (f"BUDGET BLOCK: need ${required:,.0f} collateral "
               f"(strike ${s.get('strike')} × 100), "
               f"only ${available:,.0f} free "
               f"(${portfolio_cash:,.0f} cash − ${committed:,.0f} committed"
               f"{extra}). "
               f"No margin / no naked contracts.")
        return False, msg
    extra = ""
    if pending: extra += f", ${pending:,.0f} order-pending"
    if ach:     extra += f", ${ach:,.0f} unsettled-ACH"
    return True, (f"Collateral OK: ${required:,.0f} of ${available:,.0f} available"
                  + (f" (held back{extra})" if extra else ""))


# ---------------------------------------------------------------------------
# Startup banner
# ---------------------------------------------------------------------------

def _open_options_lines(client=None) -> tuple[list[str], list[str]]:
    """
    Formatted lines for open option positions + wheel holdings.
    When client is provided, fetches live Schwab option prices to show
    current mark and profit target on each line.
    Returns (terminal_lines, slack_lines).
    """
    fetch_option_quote = None
    if client is not None:
        def fetch_option_quote(occ_sym: str):
            try:
                resp = client.get_quotes([occ_sym])
                resp.raise_for_status()
                return resp.json().get(occ_sym, {}).get("quote")
            except Exception:
                return None

    return ol.position_lines(ol.read_rows(OPTION_LEDGER_PATH),
                             ol.load_holdings(WHEEL_HOLDINGS_PATH),
                             fetch_option_quote=fetch_option_quote)


def _send_pipboy(title: str, detail: str):
    try:
        import urllib.request, json as _json, base64 as _b64
        key     = os.environ.get("ABLY_API_KEY", "")
        channel = os.environ.get("PIPBOY_CHANNEL", "pip2026boy:20260708m5paper")
        if not key:
            return
        url  = f"https://rest.ably.io/channels/{urllib.request.quote(channel, safe='')}/messages"
        body = _json.dumps({"name": "notice",
                            "data": {"title": title, "detail": detail}}).encode()
        req  = urllib.request.Request(url, data=body,
                                      headers={"Content-Type": "application/json"})
        req.add_header("Authorization",
                       "Basic " + _b64.b64encode(key.encode()).decode())
        urllib.request.urlopen(req, timeout=5)
    except Exception as exc:
        print(f"  [Pipboy] failed: {exc}")


def _send_m5paper(message: str):
    try:
        import urllib.request, json as _json
        url  = "https://rest.ably.io/channels/m5paper/messages"
        key  = "bdk-Qg.v8dBtQ:cJ3tCFyMNrmHJJy_gSx6ujcobWUDoKXUJPlD6lem-KA"
        body = _json.dumps({"name": "message", "data": message}).encode()
        req  = urllib.request.Request(url, data=body,
                                      headers={"Content-Type": "application/json"})
        import base64 as _b64
        req.add_header("Authorization",
                       "Basic " + _b64.b64encode(key.encode()).decode())
        urllib.request.urlopen(req, timeout=5)
    except Exception as exc:
        print(f"  [M5Paper] failed: {exc}")


def _send_slack(message: str):
    try:
        import sys as _sys
        _sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from notify_slack import send
        send(f"{_slack_prefix}{message}" if _slack_prefix else message)
    except Exception as exc:
        print(f"  [Slack] failed to send: {exc}")


def _scan_summary_message(scan_count: int, global_scan: int, n_scanned: int,
                          n_signals: int, events: list[str], free: float,
                          committed: float, open_positions: list[str]) -> str:
    """Build the compact per-scan Slack summary (pure — no I/O)."""
    try:
        now = _now_et().strftime("%H:%M ET")
    except Exception:
        now = ""
    lines = [
        f"📡 *Scan #{scan_count}* (G{global_scan}) · {now}",
        f"{n_scanned} scanned · {n_signals} signal(s) · "
        f"free ${free:,.0f} / committed ${committed:,.0f}",
    ]
    lines.append("\n".join(f"  {e}" for e in events) if events
                 else "  no actionable signals")
    lines.append(f"open: {', '.join(open_positions) if open_positions else 'none'}")
    return "\n".join(lines)


def _send_scan_summary(scan_count: int, global_scan: int, n_scanned: int,
                       signals, events: list[str], client, portfolio):
    """Gather the cash/positions picture and send the per-scan summary to
    Slack. Sent every scan (begin/end-of-day messages are separate) so the
    book can be followed from a phone."""
    try:
        committed = _committed_collateral()
        if portfolio is not None:
            free = portfolio.cash - committed
        elif _schwab_available is not None:
            free = _schwab_available
        elif _schwab_cash is not None:
            free = _schwab_cash - committed
        else:
            free = 30_000.0 - committed
        try:
            opens = list(ol.open_options(ol.read_rows(OPTION_LEDGER_PATH)))
            open_positions = [
                f"{o['symbol']} ${o['strike']}"
                f"{'P' if o.get('signal') == 'SELL_PUT' else 'C'}"
                for o in opens
            ]
        except Exception:
            open_positions = []
        msg = _scan_summary_message(
            scan_count, global_scan, n_scanned,
            len(signals) if signals else 0, events, free, committed,
            open_positions)
        _send_slack(msg)
    except Exception as exc:
        print(f"  [Slack] scan summary failed: {exc}")


def _check_token_age() -> tuple[str, str]:
    """
    Return (terminal_line, slack_line) warning if the Schwab token is
    approaching its 7-day hard expiry.  Empty strings when age is fine.

    Uses creation_timestamp from the JSON — the SDK sets this once during
    manual OAuth and does NOT update it on access-token refreshes (only
    mtime changes on refresh, which would always look fresh).
    """
    TOKEN_PATH = os.path.join(os.path.dirname(__file__), "schwab_token.json")
    try:
        import json as _json
        data = _json.load(open(TOKEN_PATH))
        created = data["creation_timestamp"]
        age_days = (time.time() - created) / 86400
        if age_days >= 6:
            msg = (f"Schwab token is {age_days:.1f} days old — "
                   f"expires in ~{7 - age_days:.1f}d. Re-auth soon!")
            return f"  ⚠  TOKEN WARNING: {msg}", f":warning: *TOKEN WARNING:* {msg}"
    except Exception:
        pass
    return "", ""


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
    now_str  = _display_now()
    mode     = "PAPER TRADING" if paper else "LIVE"
    mode_ico = "📄" if paper else "💰"

    try:
        regime = _fetch_regime(client)
    except Exception:
        regime = Overseer.WASTELAND

    regime_label = _overseer.describe(regime)
    cards        = _overseer.recommend_roles(regime)
    cards_str    = ", ".join(c.upper() for c in cards) if cards else "NONE — stand down"

    pos_term,  pos_slack  = _position_lines(portfolio, price_fetcher)
    opts_term, opts_slack = _open_options_lines(client)

    # ── Terminal banner ──────────────────────────────────────────────────────
    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  VAULT 76 — DAILY BRIEFING  {now_str}")
    print(sep)
    print(f"  Mode:            {mode_ico}  {mode}")
    print(f"  Regime:          {regime_label}")
    print(f"  Active cards:    {cards_str}")
    print(f"  Watchlist ({len(WATCHLIST)}):   {', '.join(WATCHLIST)}")
    print(f"  Scan interval:   {SCAN_INTERVAL_MIN} min  |  Equity sizing: {int(BUDGET_PCT_MIN*100)}–{int(BUDGET_PCT_MAX*100)}% of capital by confidence")
    if portfolio:
        opt_bal = _cash_ledger.balance() if _cash_ledger else portfolio.cash
        total   = portfolio.cash + (opt_bal - portfolio.starting_capital)
        print(f"  Equity cash:     ${portfolio.cash:,.2f}")
        print(f"  Options P&L:     ${opt_bal - portfolio.starting_capital:+,.2f}"
              f"  (total ${opt_bal:,.2f})")
        print(f"  Account total:   ${total:,.2f}")
    elif _cash_ledger:
        opt_bal   = _cash_ledger.balance()
        invested  = _cash_ledger.invested_capital()   # start + deposits
        opt_pnl   = opt_bal - invested                # P&L vs money actually in
        committed = _committed_collateral()
        pending   = _pending_collateral()
        # Use real Schwab cash when available; fall back to ledger estimate
        if _schwab_cash is not None:
            free     = _schwab_available if _schwab_available is not None else _schwab_cash - pending
            cash_src = f"Schwab total=${_schwab_cash:,.0f} / avail=${free:,.0f}"
        else:
            free = opt_bal - committed - pending
            cash_src = f"ledger est."
        print(f"  Invested capital:${invested:,.2f}")
        print(f"  Options P&L:     ${opt_pnl:+,.2f}  (total ${opt_bal:,.2f})")
        print(f"  Cash ({cash_src}): Free: ${free:,.0f}"
              f"  |  Committed: ${committed:,.0f}"
              + (f"  (−${pending:,.0f} pending)" if pending else ""))
    for line in pos_term:
        print(line)
    for line in opts_term:
        print(line)
    if paper:
        print(f"  Paper trades:    {PAPER_TRADES_PATH}")
    token_term, token_slack = _check_token_age()
    if token_term:
        print(token_term)
    print(f"  Monitor:         /schwab watch")
    print(sep)

    # ── Slack message ────────────────────────────────────────────────────────
    slack_lines = [
        f"*VAULT 76 — Daily Briefing* {now_str}",
        f"*Mode:* {mode_ico} {mode}",
        f"*Regime:* {regime_label}",
        f"*Active cards:* {cards_str}",
        f"*Watchlist ({len(WATCHLIST)}):* {', '.join(WATCHLIST)}",
        f"*Scan interval:* {SCAN_INTERVAL_MIN} min | *Equity sizing:* {int(BUDGET_PCT_MIN*100)}–{int(BUDGET_PCT_MAX*100)}% of capital by confidence",
    ]
    if portfolio:
        opt_bal = _cash_ledger.balance() if _cash_ledger else portfolio.cash
        total   = portfolio.cash + (opt_bal - portfolio.starting_capital)
        slack_lines.append(
            f"*Equity cash:* ${portfolio.cash:,.2f} | "
            f"*Options P&L:* ${opt_bal - portfolio.starting_capital:+,.2f} | "
            f"*Total:* ${total:,.2f}"
        )
    elif _cash_ledger:
        opt_bal   = _cash_ledger.balance()
        invested  = _cash_ledger.invested_capital()
        opt_pnl   = opt_bal - invested
        committed = _committed_collateral()
        pending   = _pending_collateral()
        if _schwab_cash is not None:
            free     = _schwab_available if _schwab_available is not None else _schwab_cash - pending
            cash_lbl = f"*Schwab total:* ${_schwab_cash:,.0f} | *Available:* ${free:,.0f}"
        else:
            free     = opt_bal - committed - pending
            cash_lbl = f"*Est. cash:* ${opt_bal - committed:,.0f}"
        slack_lines.append(
            f"*Invested capital:* ${invested:,.2f} | "
            f"*Options P&L:* ${opt_pnl:+,.2f} | "
            f"*Total:* ${opt_bal:,.2f}"
        )
        slack_lines.append(
            f"{cash_lbl} | *Committed:* ${committed:,.0f} | *Free:* ${free:,.0f}"
            + (f" (−${pending:,.0f} pending)" if pending else "")
        )
    slack_lines += pos_slack
    slack_lines += opts_slack
    if token_slack:
        slack_lines.append(token_slack)
    _send_slack("\n".join(slack_lines))

    if portfolio and _cash_ledger:
        opt_bal = _cash_ledger.balance()
        opt_pnl = opt_bal - portfolio.starting_capital
        total   = portfolio.cash + opt_pnl
        _send_pipboy(
            f"Vault 76 started — {_now_et().strftime('%m/%d %H:%M ET')}",
            f"Regime: {regime_label} | "
            f"Opts P&L: ${opt_pnl:+.0f} | Total: ${total:,.0f} | "
            f"Watchlist: {len(WATCHLIST)} symbols"
        )


def _print_eod(client, portfolio, price_fetcher, scan_count: int):
    """Print end-of-day summary and post it to Slack."""
    now_str = _display_now()

    cur_prices = _get_current_prices(portfolio, price_fetcher) if portfolio else {}

    print(f"\n{'='*62}")
    print(f"  VAULT 76 — END OF DAY  {now_str}  (scans: {scan_count})")
    print(f"{'='*62}")
    if portfolio:
        portfolio.print_status(cur_prices)
        if _cash_ledger:
            opt_bal = _cash_ledger.balance()
            opt_pnl = opt_bal - portfolio.starting_capital
            total   = portfolio.cash + opt_pnl
            print(f"  Options P&L:  ${opt_pnl:+,.2f}  (total ${opt_bal:,.2f})")
            print(f"  Account total:${total:,.2f}")
    elif _cash_ledger:
        opt_bal   = _cash_ledger.balance()
        invested  = _cash_ledger.invested_capital()
        opt_pnl   = opt_bal - invested
        committed = _committed_collateral()
        pending   = _pending_collateral()
        if _schwab_cash is not None:
            free     = _schwab_available if _schwab_available is not None else _schwab_cash - pending
            cash_src = f"Schwab total=${_schwab_cash:,.0f} / avail=${free:,.0f}"
        else:
            free     = opt_bal - committed - pending
            cash_src = "ledger est."
        print(f"  Invested capital:${invested:,.2f}")
        print(f"  Options P&L:     ${opt_pnl:+,.2f}  (total ${opt_bal:,.2f})")
        print(f"  Cash ({cash_src}): Free: ${free:,.0f}"
              f"  |  Committed: ${committed:,.0f}"
              + (f"  (−${pending:,.0f} pending)" if pending else ""))

    pos_term,  pos_slack  = _position_lines(portfolio, price_fetcher)
    opts_term, opts_slack = _open_options_lines(client)

    for line in opts_term:
        print(line)
    committed = _committed_collateral()
    if portfolio:
        pending = _pending_collateral()
        free    = portfolio.cash - committed - pending
        print(f"  Committed collateral: ${committed:,.0f}  |  Free cash: ${free:,.0f}"
              + (f"  (−${pending:,.0f} pending)" if pending else ""))

    if portfolio:
        s = portfolio.summary(cur_prices)
        pnl_sign = "+" if s["total_pnl_dollar"] >= 0 else ""
        opt_bal   = _cash_ledger.balance() if _cash_ledger else s["cash"]
        opt_pnl   = opt_bal - portfolio.starting_capital
        total     = s["cash"] + opt_pnl
        slack_lines = [
            f"*VAULT 76 — End of Day* {now_str}",
            f"*Scans today:* {scan_count}",
            f"*Equity cash:* ${s['cash']:,.2f}  |  "
            f"*Options P&L:* ${opt_pnl:+,.2f}  |  "
            f"*Account total:* ${total:,.2f}",
            f"*Equity P&L:* {pnl_sign}${s['total_pnl_dollar']:,.2f} ({pnl_sign}{s['total_pnl_pct']:.2f}%)  |  "
            f"*Realized:* ${s['realized_pnl_dollar']:+,.2f}  |  "
            f"*Unrealized:* ${s['unrealized_pnl_dollar']:+,.2f}",
        ] + pos_slack + opts_slack
    elif _cash_ledger:
        opt_bal  = _cash_ledger.balance()
        invested = _cash_ledger.invested_capital()
        opt_pnl  = opt_bal - invested
        committed = _committed_collateral()
        pending  = _pending_collateral()
        if _schwab_cash is not None:
            free_eod = _schwab_available if _schwab_available is not None else _schwab_cash - pending
            cash_lbl = f"*Schwab total:* ${_schwab_cash:,.0f} | *Available:* ${free_eod:,.0f}"
        else:
            free_eod = opt_bal - committed - pending
            cash_lbl = f"*Est. cash:* ${opt_bal - committed:,.0f}"
        slack_lines = [
            f"*VAULT 76 — End of Day* {now_str}",
            f"*Scans today:* {scan_count}",
            f"*Invested capital:* ${invested:,.2f}  |  "
            f"*Options P&L:* ${opt_pnl:+,.2f}  |  "
            f"*Total:* ${opt_bal:,.2f}",
            f"{cash_lbl} | *Committed:* ${committed:,.0f} | *Free:* ${free_eod:,.0f}"
            + (f" (−${pending:,.0f} pending)" if pending else ""),
        ] + opts_slack
    else:
        slack_lines = [
            f"*VAULT 76 — End of Day* {now_str}  (scans: {scan_count})",
        ] + opts_slack

    _send_slack("\n".join(slack_lines))

    if portfolio and _cash_ledger:
        s       = portfolio.summary(cur_prices)
        opt_bal = _cash_ledger.balance()
        opt_pnl = opt_bal - portfolio.starting_capital
        total   = s["cash"] + opt_pnl
        _send_m5paper(
            f"EOD {_now_et().strftime('%m/%d')} | "
            f"Scans:{scan_count} | "
            f"Opts:{opt_pnl:+.0f} | "
            f"Total:${total:,.0f}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", action="store_true",
                        help="Paper trading mode — track fake buys/sells, show P&L")
    args = parser.parse_args()

    # Load the pip schwab-py package without being shadowed by the local schwab/ directory.
    # vault76/overseer.py already cached the local schwab package in sys.modules['schwab'];
    # temporarily swap it out, import the pip package, then restore.
    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    _removed = [p for p in sys.path if os.path.abspath(p) == _project_root]
    for p in _removed:
        sys.path.remove(p)
    _local_schwab = sys.modules.pop("schwab", None)
    import schwab as schwab_lib
    sys.modules["schwab"] = _local_schwab  # restore local package for vault76 imports
    for p in _removed:
        sys.path.insert(0, p)

    CLIENT_ID     = os.environ["SCHWAB_CLIENT_ID"]
    CLIENT_SECRET = os.environ["SCHWAB_CLIENT_SECRET"]
    TOKEN_PATH    = os.path.join(os.path.dirname(__file__), "schwab_token.json")
    client = schwab_lib.auth.client_from_token_file(TOKEN_PATH, CLIENT_ID, CLIENT_SECRET)

    global _current_client
    _current_client = client

    portfolio = None
    if args.paper:
        from paper_portfolio import PaperPortfolio
        portfolio = PaperPortfolio(PAPER_TRADES_PATH)

    global _current_portfolio
    _current_portfolio = portfolio

    global _cash_ledger
    starting = portfolio.starting_capital if portfolio else 30_000.0
    _cash_ledger = cl.CashLedger(CASH_LEDGER_PATH, starting_capital=starting)

    price_fetcher = _make_price_fetcher(client)

    # Train LGBM move-risk models from local history (seconds/symbol):
    # drop risk for SELL_PUT/BUY, rally (called-away) risk for SELL_CALL
    global _assign_models, _upside_models
    try:
        _assign_models = assignment_risk.load_models(WATCHLIST, DATA_DIR)
        aucs = {s: m.holdout_auc for s, m in _assign_models.items()
                if m.holdout_auc is not None}
        print(f"  [AssignRisk] models ready for {len(_assign_models)}/"
              f"{len(WATCHLIST)} symbols"
              + (f"  (median holdout AUC {sorted(aucs.values())[len(aucs)//2]:.2f})"
                 if aucs else ""))
    except Exception as exc:
        print(f"  [AssignRisk] disabled ({exc})")
        _assign_models = {}
    try:
        _upside_models = assignment_risk.load_models(WATCHLIST, DATA_DIR,
                                                     direction="up")
        aucs = {s: m.holdout_auc for s, m in _upside_models.items()
                if m.holdout_auc is not None}
        print(f"  [CalledAway] models ready for {len(_upside_models)}/"
              f"{len(WATCHLIST)} symbols"
              + (f"  (median holdout AUC {sorted(aucs.values())[len(aucs)//2]:.2f})"
                 if aucs else ""))
    except Exception as exc:
        print(f"  [CalledAway] disabled ({exc})")
        _upside_models = {}

    kronos_cache  = _load_kronos_cache(WATCHLIST)

    global _current_kronos_cache
    _current_kronos_cache = kronos_cache

    # Sector ETF cache for Hunter three-pillar check (loaded from disk, no API call)
    print("Loading sector ETF cache for Hunter...", flush=True)
    _load_sector_cache()
    print(f"  Sector cache: {list(_sector_cache.keys())}", flush=True)

    # TimesFM zero-shot 30-day forecasts — second opinion next to Kronos
    global _timesfm_cache
    _timesfm_cache = timesfm_advisor.load_cache(WATCHLIST, DATA_DIR)

    router_holds = wheel_router.load_holds(ROUTER_HOLDS_PATH)
    if router_holds:
        print("  [Router] active holds: "
              + ", ".join(f"{sym} {p['shares']}sh @ ${p['entry']}"
                          for sym, p in router_holds.items()))
    medic_holds = wheel_router.load_holds(MEDIC_HOLDS_PATH)
    if medic_holds:
        print("  [Medic] crisis positions: "
              + ", ".join(f"{sym} {p['shares']}sh @ ${p['entry']}"
                          for sym, p in medic_holds.items()))

    # Settle overnight expiries/assignments before the briefing
    try:
        _process_option_ledger(client, paper=args.paper)
    except Exception as exc:
        print(f"  [WHEEL] ledger processing failed: {exc}")

    # Sync live Schwab balances BEFORE the briefing so it shows real cash, not
    # the ledger's starting_capital fallback (which ignores deposits). The scan
    # hook = the overseer's reconcile; it sets _schwab_cash/_available/_pending.
    if _scan_hook_fn is not None:
        try:
            _scan_hook_fn()
        except Exception as exc:
            print(f"  [ScanHook] pre-briefing balance sync failed: {exc}")

    _print_startup(client, paper=args.paper, portfolio=portfolio,
                   price_fetcher=price_fetcher)
    scan_count = 0
    global_scan = _load_global_scan()

    while True:
        now = _now_et().strftime("%H:%M:%S ET")

        if _market_closed_for_today():
            if scan_count > 0:
                _print_eod(client, portfolio, price_fetcher, scan_count)
                if portfolio:
                    portfolio.log_scan(scan_num=scan_count, symbols_scanned=0,
                                       signals_found=0)
            now_et = _now_et()
            _sleep_until_market_open(now_et)
            scan_count = 0
            continue

        if not _is_market_hours():
            print(f"[{now}] Pre-market — waiting for open (9am ET). Sleeping 5 min.")
            time.sleep(300)
            continue

        if scan_count == 0:
            _print_startup(client, paper=args.paper, portfolio=portfolio,
                           price_fetcher=price_fetcher)

        if _scan_hook_fn is not None:
            try:
                _scan_hook_fn()
            except Exception as _hook_exc:
                print(f"  [ScanHook] error: {_hook_exc}")

        scan_count += 1
        global_scan += 1
        _save_global_scan(global_scan)
        regime, spy_df = _fetch_regime_and_spy(client)
        regime_str     = _overseer.describe(regime)
        print(f"\n{'━'*70}")
        print(f"  SCAN #{scan_count} [G:{global_scan}]  {now}  |  {regime_str}  |  {len(WATCHLIST)} symbols")
        print(f"{'━'*70}")

        # Fast risk-off check (FinRL #1): suppress new puts if SPY -3% in 3 days
        riskoff_active, riskoff_msg = _check_fast_riskoff(spy_df)
        if riskoff_active:
            print(f"  ⚡ {riskoff_msg}")

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
                    if _cash_ledger:
                        _cash_ledger.record(
                            "STOCK_SELL", ex["symbol"], round(ex["proceeds"], 2),
                            f"SELL {ex['shares']} {ex['symbol']} @ ${ex['exit']:.2f}"
                            f" [{ex['exit_reason']}] P&L ${ex['pnl_dollar']:+.2f}"
                        )

        # Wheel bookkeeping: settle expiries/assignments/early exits in the
        # options ledger. Assignments feed covered-call scans via holdings.
        try:
            _process_option_ledger(client, paper=args.paper)
        except Exception as exc:
            print(f"  [WHEEL] ledger processing failed: {exc}")

        # Scan for new signals
        signals = _scan_all(client, regime=regime, spy_df=spy_df)

        # Deterministic allocation order (plan item 4, step 2): cash-freeing
        # signals first, then puts by premium/day per collateral dollar with
        # a same-symbol concentration penalty — replaces the accidental
        # first-come-first-served WATCHLIST order.
        try:
            open_counts: dict = {}
            for p in ol.open_options(ol.read_rows(OPTION_LEDGER_PATH)):
                sym = p.get("symbol")
                open_counts[sym] = open_counts.get(sym, 0) + 1
            for sym in wheel_router.load_holds(ROUTER_HOLDS_PATH):
                open_counts[sym] = open_counts.get(sym, 0) + 1
            # score_puts=False: the portfolio backtest (2026-07-04) measured
            # premium-density put ranking at -$47K vs neutral order on $30K —
            # keep only the cash-freeing-first tier until score v2 wins.
            signals = allocator.rank_signals(signals, open_counts,
                                             score_puts=False)
        except Exception as exc:
            print(f"  [Allocator] ranking skipped ({exc})")

        # Publish the batch so the AutoOverseer can weigh each signal
        # against its competitors for the same cash (capital allocation)
        global _current_scan_signals
        _current_scan_signals = signals

        if portfolio:
            portfolio.log_scan(
                scan_num=scan_count,
                symbols_scanned=len(WATCHLIST),
                signals_found=len(signals),
            )

        scan_events: list[str] = []   # per-signal dispositions for Slack

        if not signals:
            print(f"  No signals.")
        else:
            print(f"  {len(signals)} signal(s) found.")
            for s in signals:
                _print_signal(s, paper=args.paper, kronos_cache=kronos_cache)

                # Fast risk-off gate — suppress new puts when SPY shocked
                if riskoff_active and s["signal"] == "SELL_PUT":
                    print(f"\n  ⚡ {riskoff_msg} — skipping {s['symbol']} SELL_PUT")
                    if args.paper:
                        s["reason"] = f"fast risk-off: {riskoff_msg}"
                        _log_option_trade(s, verdict="RISKOFF_BLOCK")
                    print(f"\n{VERDICT_SKIP}")
                    print(f"SKIPPED: {s['symbol']} {s['signal']} — fast risk-off")
                    print(VERDICT_SKIP)
                    scan_events.append(f"⚡ {s['symbol']} {s['signal']} risk-off skip")
                    continue

                # Kronos advisory — warn if put strike is above predicted 30d support
                k_warn, k_msg = _kronos_advisory(s, kronos_cache)
                if k_warn:
                    print(f"\n  ⚠  {k_msg}")
                    print(f"  ⚠  Strike above Kronos support floor — assignment risk elevated.")

                # Budget check — auto-skip if collateral exceeds available cash.
                # Real mode: use the Schwab cash synced by the overseer's scan
                # hook, not a constant — the account balance drifts from
                # $30,000 with every close (observed 2026-07-28).
                cash = (portfolio.cash if portfolio
                        else _schwab_cash if _schwab_cash is not None
                        else 30_000.0)
                budget_ok, budget_msg = _budget_check(s, cash)
                if not budget_ok:
                    print(f"\n  ⛔ {budget_msg}")
                    if args.paper:
                        _log_option_trade(s, verdict="BUDGET_BLOCK")
                    print(f"\n{VERDICT_SKIP}")
                    print(f"SKIPPED: {s['symbol']} {s['signal']} — budget block")
                    print(VERDICT_SKIP)
                    scan_events.append(
                        f"⛔ {s['symbol']} {s['signal']} ${s.get('strike')} budget block")
                    continue

                if budget_msg:
                    print(f"  ✓ {budget_msg}")

                # Close signals (SELL equity, BUY_TO_CLOSE options) need no approval
                _close_signals = {"SELL", "BUY_TO_CLOSE"}
                if s["signal"] in _close_signals:
                    verdict = "y"
                    print(f"  ✅ Auto-approved close — no LLM needed")
                else:
                    verdict = _decision_fn(s) if _decision_fn is not None else _wait_for_verdict()
                sig = s["signal"]

                if verdict == "q":
                    if portfolio:
                        print("\nFinal portfolio status:")
                        cur_prices = _get_current_prices(portfolio, price_fetcher)
                        portfolio.print_status(cur_prices)
                    print("Exiting scanner.")
                    sys.exit(0)

                elif verdict == "y":
                    print(f"\n{VERDICT_OK}")
                    if sig == "BUY":
                        _cap = portfolio.cash if portfolio else 30_000.0
                        _budget = _equity_budget(s.get("confidence", 50), _cap)
                        shares = max(1, int(_budget / s["entry"]))
                        print(f"  [Sizing] conf={s.get('confidence',50)} → "
                              f"${_budget:,.0f} ({_budget/_cap*100:.0f}% of ${_cap:,.0f}) → {shares} shares")
                        print(f"APPROVED: {s['symbol']} BUY @ ${s['entry']}")
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
                                if _cash_ledger:
                                    _cash_ledger.record(
                                        "STOCK_BUY", s["symbol"], -round(pos["cost"], 2),
                                        f"BUY {pos['shares']} {s['symbol']} @ ${pos['entry']:.2f}"
                                    )
                            else:
                                print("  [PAPER] Insufficient cash — trade not recorded.")
                        else:
                            print(f"  Suggested: BUY {shares} shares @ market")
                    elif sig in ("SELL_PUT", "SELL_CALL"):
                        print(f"APPROVED: {s['symbol']} {sig} strike=${s['strike']} "
                              f"premium=${s['premium']}/sh (${s['premium']*100:.0f}/contract)")
                        if args.paper:
                            _log_option_trade(s, verdict="APPROVED")
                            print(f"  [PAPER] Logged to {OPTION_LEDGER_PATH}")
                            if _cash_ledger:
                                prem_ct = round(float(s.get("premium", 0)) * 100, 2)
                                _cash_ledger.record(
                                    "OPTION_SELL", s["symbol"], prem_ct,
                                    f"{sig} {s['symbol']} strike=${s['strike']} "
                                    f"exp={s.get('dte','?')}d premium=${prem_ct:.2f}"
                                )
                        else:
                            print(f"  Suggested: {sig} {s['symbol']} ${s['strike']} strike, "
                                  f"${s['dte']}d DTE — collect ${s['premium']*100:.0f}/contract")
                    elif sig == "BUY_CALL":
                        cost = float(s.get("cost_per_ct", s.get("premium", 0) * 100))
                        print(f"APPROVED: {s['symbol']} BUY_CALL strike=${s['strike']} "
                              f"premium=${s['premium']}/sh  cost=${cost:.0f}/contract")
                        if args.paper:
                            _log_option_trade(s, verdict="APPROVED")
                            print(f"  [PAPER] Logged to {OPTION_LEDGER_PATH}")
                            if _cash_ledger:
                                _cash_ledger.record(
                                    "OPTION_BUY", s["symbol"], -cost,
                                    f"BUY_CALL {s['symbol']} strike=${s['strike']} "
                                    f"exp={s.get('dte','?')}d cost=${cost:.2f}"
                                )
                        else:
                            print(f"  Suggested: BUY 1 call {s['symbol']} ${s['strike']} "
                                  f"strike  {s.get('dte',35)}d DTE — pay ${cost:.0f}/contract")

                    elif sig == "HOLD_SHARES":
                        wheel_router.enter_hold(
                            ROUTER_HOLDS_PATH, s["symbol"], s["shares"],
                            s["close"], str(pd.Timestamp.now().date()))
                        print(f"APPROVED: {s['symbol']} HOLD_SHARES — "
                              f"{s['shares']} sh @ ${s['close']}; Scavenger "
                              f"suppressed until RESUME_WHEEL")
                        if not args.paper:
                            print(f"  Suggested: BUY {s['shares']} {s['symbol']} "
                                  f"@ market (router hold — no automated order)")
                    elif sig == "RESUME_WHEEL":
                        pnl = wheel_router.exit_hold(
                            ROUTER_HOLDS_PATH, s["symbol"], s["close"])
                        print(f"APPROVED: {s['symbol']} RESUME_WHEEL — hold "
                              f"closed @ ${s['close']}"
                              + (f"  P&L ${pnl:+,.2f}" if pnl is not None else ""))
                        if not args.paper:
                            print(f"  Suggested: SELL {s['shares']} {s['symbol']} "
                                  f"@ market (router exit — no automated order)")
                    elif sig == "BUY_ETF":
                        wheel_router.enter_hold(
                            MEDIC_HOLDS_PATH, s["symbol"], s["shares"],
                            s["close"], str(pd.Timestamp.now().date()))
                        print(f"APPROVED: {s['symbol']} BUY_ETF — Medic buys "
                              f"{s['shares']} sh @ ${s['close']} (crisis entry)")
                        if not args.paper:
                            print(f"  Suggested: BUY {s['shares']} {s['symbol']} "
                                  f"@ market (medic — no automated order)")
                    elif sig == "SELL_ETF":
                        pnl = wheel_router.exit_hold(
                            MEDIC_HOLDS_PATH, s["symbol"], s["close"])
                        print(f"APPROVED: {s['symbol']} SELL_ETF — Medic exits "
                              f"@ ${s['close']}"
                              + (f"  P&L ${pnl:+,.2f}" if pnl is not None else ""))
                        if not args.paper:
                            print(f"  Suggested: SELL {s['shares']} {s['symbol']} "
                                  f"@ market (medic — no automated order)")
                    print(VERDICT_OK)

                else:  # skip
                    print(f"\n{VERDICT_SKIP}")
                    print(f"SKIPPED: {s['symbol']} {sig}")
                    if sig == "BUY" and portfolio:
                        portfolio.log_signal(
                            s["symbol"], s.get("entry", 0), s.get("target", 0),
                            s.get("stop", 0), s.get("rsi", 0), s.get("adx", 0),
                            verdict="SKIPPED",
                        )
                    elif sig in ("SELL_PUT", "SELL_CALL", "BUY_CALL") and args.paper:
                        _log_option_trade(s, verdict="SKIPPED")
                    print(VERDICT_SKIP)

                # Record this signal's disposition for the per-scan Slack summary
                _strike = f" ${s.get('strike')}" if s.get("strike") else ""
                scan_events.append(f"✅ {s['symbol']} {sig}{_strike}" if verdict == "y"
                                   else f"➖ {s['symbol']} {sig}{_strike} skipped")

        # Show portfolio status at end of each scan
        if portfolio:
            cur_prices = _get_current_prices(portfolio, price_fetcher)
            portfolio.print_status(cur_prices)
            # Show open options + committed collateral so the display matches
            # the budget tracker (which reads from the ledger CSV).
            opts_term, _ = _open_options_lines(client)
            committed    = _committed_collateral()
            free         = portfolio.cash - committed
            print(f"\n  ┌─ Options / Collateral " + "─" * 35)
            for line in opts_term:
                print(f"  │{line[1:]}" if line.startswith(" ") else f"  │ {line}")
            print(f"  │ Committed:  ${committed:>10,.0f}  collateral locked")
            print(f"  │ Free cash:  ${free:>10,.0f}  available for new puts")
            print("  └" + "─" * 57)

        print(f"{'─'*70}")
        print(f"  ■ END SCAN #{scan_count} [G:{global_scan}]  —  next in {SCAN_INTERVAL_MIN} min.")
        print(f"{'─'*70}")

        # Send this scan's summary to Slack (every scan, per user request)
        _send_scan_summary(scan_count, global_scan, len(WATCHLIST),
                           signals, scan_events, client, portfolio)

        time.sleep(SCAN_INTERVAL_MIN * 60)


if __name__ == "__main__":
    main()
