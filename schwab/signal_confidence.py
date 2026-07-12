"""
Signal confidence scorer — composite 0–100 quality score for scanner signals.

Scores are computed from the fully-assembled signal dict (after LGBM, TimesFM,
and Kronos advisory data have been attached). Called in live_scanner._scan_all()
just before signals.append(s).

Score bands:
  ≥70  HIGH  — strong setup, lean APPROVE
  40–69 MED  — acceptable, weigh other fields
  <40  LOW   — weak setup, lean SKIP

Dispatcher:
  score(signal, kronos_buf=None) -> int

Per-type scorers:
  score_sell_put(signal, kronos_buf) -> int
  score_buy(signal)                  -> int
  score_sell_call(signal)            -> int
  score_buy_call(signal)             -> int
"""


def score(signal: dict, kronos_buf: float | None = None) -> int:
    """Dispatch to the right scorer. Returns 0 for unknown signal types."""
    sig = signal.get("signal", "NONE")
    if sig == "SELL_PUT":
        return score_sell_put(signal, kronos_buf=kronos_buf)
    if sig == "BUY":
        return score_buy(signal)
    if sig == "SELL_CALL":
        return score_sell_call(signal)
    if sig == "BUY_CALL":
        return score_buy_call(signal)
    return 0


def score_sell_put(signal: dict, kronos_buf: float | None = None) -> int:
    """
    SELL_PUT quality score (0–100).

    Weights (sum = 100 base, ±10 forward-model adjustment):
      Assignment risk  30 pts — most critical; stock must not fall below strike
      ADX sideways     20 pts — lower = more sideways = better for premium selling
      Premium yield    20 pts — reward for capital tied up
      RSI neutrality   15 pts — distance from 50; avoid falling-knife entries
      HV sweet spot    15 pts — 25-45% is ideal; too low = thin, too high = gamma
      TimesFM/Kronos  ±10 pts — forward-looking modifiers, capped
    """
    adx          = float(signal.get("adx", 15))
    rsi          = float(signal.get("rsi", 50))
    hv           = float(signal.get("hv", 25))        # already in % form
    premium_pct  = float(signal.get("premium_pct", 0.5))
    assign_risk  = signal.get("assign_risk_pct")       # float % or None
    timesfm      = signal.get("timesfm_30d_pct")       # float % or None

    s = 0

    # ── Assignment risk (30 pts) ──────────────────────────────────────────────
    if assign_risk is None:
        s += 15                     # neutral — no model for this symbol
    elif assign_risk < 20:
        s += 30
    elif assign_risk < 30:
        s += 22
    elif assign_risk < 40:
        s += 14
    elif assign_risk < 50:
        s += 7
    # >= 50: 0

    # ── ADX sideways quality (20 pts) ────────────────────────────────────────
    # Hard filter already ensures adx < 20; score how well below threshold
    if adx < 10:
        s += 20
    elif adx < 13:
        s += 16
    elif adx < 16:
        s += 12
    else:                           # 16-20
        s += 8

    # ── Premium yield (20 pts) ───────────────────────────────────────────────
    if premium_pct >= 2.0:
        s += 20
    elif premium_pct >= 1.5:
        s += 16
    elif premium_pct >= 1.0:
        s += 12
    elif premium_pct >= 0.7:
        s += 8
    else:                           # 0.5–0.7%
        s += 4

    # ── RSI proximity to 50 (15 pts) ─────────────────────────────────────────
    dist = abs(rsi - 50)
    if dist < 4:
        s += 15
    elif dist < 7:
        s += 12
    elif dist < 10:
        s += 8
    else:                           # 10–15 (edge of 35–65 band)
        s += 4

    # ── HV sweet spot (15 pts) ───────────────────────────────────────────────
    if 25 <= hv <= 45:
        s += 15
    elif 20 <= hv < 25 or 45 < hv <= 55:
        s += 10
    elif hv > 55:
        s += 3
    else:                           # low HV but above MIN_HV
        s += 7

    # ── Forward-model adjustment (capped ±10) ────────────────────────────────
    adj = 0
    if timesfm is not None:
        if timesfm > 5:
            adj += 5
        elif timesfm > 2:
            adj += 3
        elif timesfm < -5:
            adj -= 5
        elif timesfm < -2:
            adj -= 3

    if kronos_buf is not None:
        if kronos_buf > 8:
            adj += 5
        elif kronos_buf > 3:
            adj += 3
        elif kronos_buf < -2:
            adj -= 5
        elif kronos_buf < 0:
            adj -= 2

    s += max(-10, min(10, adj))
    return max(0, min(100, s))


def score_buy(signal: dict) -> int:
    """
    BUY (Raider pullback-in-trend) quality score (0–100).

    Weights (sum = 100 base, ±5 TimesFM adjustment):
      Trend strength   25 pts — ADX; higher = more confirmed trend
      Pullback quality 25 pts — RSI at bounce; ideal 40-52
      EMA alignment    25 pts — EMA20 above EMA50 gap %
      Drop risk        25 pts — LGBM P(>5% drop); lower = real dip not breakdown
      TimesFM          ±5 pts — forward hint on trend continuation
    """
    adx       = float(signal.get("adx", 22))
    rsi       = float(signal.get("rsi", 45))
    ema20     = float(signal.get("ema20", 0))
    ema50     = float(signal.get("ema50", 0))
    drop_risk = signal.get("drop_risk_pct")
    timesfm   = signal.get("timesfm_30d_pct")

    s = 0

    # ── Trend strength (25 pts) ──────────────────────────────────────────────
    if adx > 40:
        s += 25
    elif adx > 32:
        s += 20
    elif adx > 25:
        s += 15
    else:                           # 20–25 (at minimum threshold)
        s += 10

    # ── Pullback quality (25 pts) ────────────────────────────────────────────
    if 40 <= rsi <= 52:
        s += 25                     # ideal bounce zone
    elif 35 <= rsi < 40 or 52 < rsi <= 56:
        s += 18
    elif 30 <= rsi < 35 or 56 < rsi <= 60:
        s += 10
    else:
        s += 5

    # ── EMA alignment gap (25 pts) ───────────────────────────────────────────
    if ema50 > 0:
        gap = (ema20 - ema50) / ema50 * 100
        if gap > 5:
            s += 25
        elif gap > 3:
            s += 20
        elif gap > 1.5:
            s += 14
        elif gap > 0:
            s += 8

    # ── Drop risk (25 pts) ───────────────────────────────────────────────────
    if drop_risk is None:
        s += 12                     # neutral
    elif drop_risk < 20:
        s += 25
    elif drop_risk < 30:
        s += 20
    elif drop_risk < 40:
        s += 14
    elif drop_risk < 50:
        s += 7
    # >= 50: 0

    # ── TimesFM adjustment (capped ±5) ───────────────────────────────────────
    adj = 0
    if timesfm is not None:
        if timesfm > 8:
            adj += 5
        elif timesfm > 4:
            adj += 3
        elif timesfm < -5:
            adj -= 5
        elif timesfm < -2:
            adj -= 3
    s += max(-5, min(5, adj))
    return max(0, min(100, s))


def score_sell_call(signal: dict) -> int:
    """
    SELL_CALL (Scavenger covered call) quality score (0–100).

    Weights (sum = 100 base, ±5 TimesFM adjustment):
      ADX (low)        25 pts — low = stock not running away; higher = risky to cap
      Premium yield    30 pts — reward per capital month
      Profit cushion   20 pts — price above cost basis; more cushion = less risk
      Called-away risk 25 pts — moderate is ok (locks in profit); very high = capping a run
      TimesFM          ±5 pts — bullish forecast penalises (stock may run past strike)
    """
    adx         = float(signal.get("adx", 15))
    premium_pct = float(signal.get("premium_pct", 0.5))
    close       = float(signal.get("close", 1))
    cost_basis  = float(signal.get("cost_basis", close))
    called_away = signal.get("called_away_pct")
    timesfm     = signal.get("timesfm_30d_pct")

    s = 0

    # ── ADX — low is better for covered calls (25 pts) ───────────────────────
    if adx < 13:
        s += 25
    elif adx < 20:
        s += 20
    elif adx < 26:
        s += 12                     # moderate — wider strike used
    else:
        s += 5                      # strong trend — risky to sell call

    # ── Premium yield (30 pts) ───────────────────────────────────────────────
    if premium_pct >= 1.5:
        s += 30
    elif premium_pct >= 1.0:
        s += 24
    elif premium_pct >= 0.7:
        s += 16
    elif premium_pct >= 0.5:
        s += 8

    # ── Profit cushion (20 pts) ──────────────────────────────────────────────
    gain_pct = (close - cost_basis) / cost_basis * 100 if cost_basis > 0 else 0
    if gain_pct > 10:
        s += 20
    elif gain_pct > 5:
        s += 15
    elif gain_pct > 0:
        s += 10
    elif gain_pct > -5:
        s += 5                      # slightly underwater but within allowed range

    # ── Called-away risk (25 pts) ────────────────────────────────────────────
    if called_away is None:
        s += 12
    elif called_away < 20:
        s += 25                     # unlikely to be called away
    elif called_away < 35:
        s += 20
    elif called_away < 50:
        s += 12                     # moderate — premium collected, may be called
    elif called_away < 65:
        s += 6                      # likely called away — still profitable if cushion is there
    else:
        s += 3                      # almost certain to be called — capping a strong run

    # ── TimesFM adjustment (capped ±5) ───────────────────────────────────────
    # Bullish forecast = stock likely runs past call strike → penalise
    adj = 0
    if timesfm is not None:
        if timesfm > 8:
            adj -= 5
        elif timesfm > 4:
            adj -= 3
        elif timesfm < -4:
            adj += 3                # weak outlook → selling call is smart
    s += max(-5, min(5, adj))
    return max(0, min(100, s))


def score_buy_call(signal: dict) -> int:
    """
    BUY_CALL (Hunter VCP breakout) quality score (0–100).

    Calibrated against backtest distribution (n=32, 2019–2026):
      vcp_tight_pct  p25=3.5%, p50=5.4%, p75=6.7%
      breakout_vol   p25=1.51×, p50=1.68×, p75=1.99×
      adx            p25=23.8, p50=26.6, p75=37.1
      rsi            p25=70.2, p50=73.0, p75=76.5  ← high by design at breakout
      premium_pct    p25=3.7%, p50=4.8%, p75=7.2%

    Weights (sum = 100 base, ±5 TimesFM adjustment):
      VCP base quality    30 pts — vcp_tight_pct; tighter = more compressed = better
      Breakout conviction 25 pts — vol_ratio; volume surge confirms real breakout
      Trend strength      20 pts — ADX before base; stronger = cleaner momentum
      RSI momentum zone   15 pts — 65-78 is the sweet spot at VCP breakout day
      Premium cost        10 pts — premium_pct of close; cheaper = more asymmetry
      TimesFM             ±5 pts — bullish bias boosts; bearish penalises
    """
    adx          = float(signal.get("adx", 25))
    rsi          = float(signal.get("rsi", 72))
    premium_pct  = float(signal.get("premium_pct", 5.0))
    vcp_tight    = float(signal.get("vcp_tight_pct", 5.5))
    breakout_vol = float(signal.get("breakout_vol", 1.6))
    timesfm      = signal.get("timesfm_30d_pct")

    s = 0

    # ── VCP base quality (30 pts) ─────────────────────────────────────────────
    # Tighter 5-bar rolling range in the consolidation zone = more compressed base
    if vcp_tight < 2.5:
        s += 30                    # very tight — institutional accumulation
    elif vcp_tight < 3.5:
        s += 24                    # above p25: solid base
    elif vcp_tight < 5.5:
        s += 16                    # near median: acceptable
    elif vcp_tight < 7.0:
        s += 8                     # loose — VCP still valid but lower quality
    # >= 7.0: base too wide, 0 pts

    # ── Breakout conviction (25 pts) ─────────────────────────────────────────
    # vol_ratio: today's volume vs 20-day average
    if breakout_vol >= 2.5:
        s += 25                    # strong institutional buying
    elif breakout_vol >= 2.0:
        s += 20                    # above p75: convincing breakout
    elif breakout_vol >= 1.6:
        s += 13                    # near p50: reasonable
    else:                          # 1.4–1.6: at minimum threshold
        s += 6

    # ── Trend strength (20 pts) ───────────────────────────────────────────────
    if adx > 40:
        s += 20
    elif adx > 32:
        s += 16
    elif adx > 25:
        s += 10                    # near p50: trend confirmed but not strong
    else:                          # 20–25: minimum ADX; marginal trend
        s += 5

    # ── RSI momentum zone (15 pts) ────────────────────────────────────────────
    # VCP breakouts fire with RSI 65-80 by design; sweet spot is 65-78
    if 65 <= rsi <= 78:
        s += 15
    elif 60 <= rsi < 65 or 78 < rsi <= 80:
        s += 9
    elif 55 <= rsi < 60:
        s += 4
    # < 55 or > 80 (blocked by filter): 0

    # ── Premium cost (10 pts) ────────────────────────────────────────────────
    # Cheaper call relative to stock price = more leverage room
    if premium_pct < 3.0:
        s += 10
    elif premium_pct < 5.0:
        s += 7                     # near p50: reasonable
    elif premium_pct < 7.5:
        s += 4                     # above p75: getting expensive
    elif premium_pct < 10.0:
        s += 1
    # >= 10%: near max allowed, 0

    # ── TimesFM adjustment (capped ±5) ────────────────────────────────────────
    adj = 0
    if timesfm is not None:
        if timesfm > 8:
            adj += 5
        elif timesfm > 4:
            adj += 3
        elif timesfm < -5:
            adj -= 5
        elif timesfm < -2:
            adj -= 3
    s += max(-5, min(5, adj))
    return max(0, min(100, s))


def tier(confidence: int) -> str:
    """Return 'HIGH', 'MED', or 'LOW' band label."""
    if confidence >= 70:
        return "HIGH"
    if confidence >= 40:
        return "MED"
    return "LOW"
