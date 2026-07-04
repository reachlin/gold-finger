"""
The Overseer — Vault 76 market regime classifier.

Reads the wasteland conditions and decides which roles to deploy.

Regimes:
  RECLAMATION  — Bull market. SPY trending up, VIX calm. Rebuilding is possible.
  WASTELAND    — Bear or sideways. Hostile terrain. Survival tools needed.
  NUKED_ZONE   — Crash / extreme volatility. Blast radius active. Stay out.
"""
import sys
import os
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from schwab.trend_scanner import compute_indicators

VIX_NUKED   = 30.0   # VIX ≥ this → NUKED_ZONE
MIN_BARS    = 60     # minimum SPY bars needed to classify
ADX_RUNNER  = 28     # per-stock ADX ≥ this → trending runner, route to Raider


class Overseer:
    RECLAMATION = "RECLAMATION"
    WASTELAND   = "WASTELAND"
    NUKED_ZONE  = "NUKED_ZONE"

    _DESCRIPTIONS = {
        RECLAMATION: "Reclamation Day — bull market, rebuilding in progress",
        WASTELAND:   "The Wasteland — bear/sideways, survival mode engaged",
        NUKED_ZONE:  "Nuked Zone — blast radius active, all roles benched",
    }

    # Which roles are active per regime
    _ROLES = {
        RECLAMATION: ["maggie", "raider", "scavenger"],  # Maggie hunts breakouts; Raider attacks pullbacks; Scavenger sells puts on sideways stocks
        WASTELAND:   ["scavenger", "raider"],   # Scavenger primary for income; Raider opportunistic on any trending names
        NUKED_ZONE:  ["chemist", "medic"],      # Blast radius — Chemist harvests the chaos, Medic buys quality ETFs at panic prices
    }

    def classify(self, spy_df: pd.DataFrame, vix: float = 20.0) -> str:
        """
        Classify current market regime.

        spy_df : daily OHLCV DataFrame for SPY (or any broad index)
        vix    : current VIX level (fetch from signal_verifier.fetch_vix())
        """
        if vix >= VIX_NUKED:
            return self.NUKED_ZONE

        if len(spy_df) < MIN_BARS:
            return self.WASTELAND   # can't assess → play it safe

        ind  = compute_indicators(spy_df).dropna().reset_index(drop=True)
        if len(ind) < MIN_BARS:
            return self.WASTELAND

        last = ind.iloc[-1]
        prev = ind.iloc[-6] if len(ind) > 6 else ind.iloc[0]
        return self.classify_row(last, prev, vix)

    def classify_row(self, last, prev, vix: float = 20.0) -> str:
        """
        Fast-path regime classification from precomputed indicator rows.

        last : indicator row for the current bar (needs close, ema50)
        prev : indicator row 5 bars earlier (needs ema50)

        Walk-forward backtests precompute indicators once over the full
        series and call this per bar instead of classify(), which re-runs
        compute_indicators on a growing slice. The regime rule lives ONLY
        here so live and backtest classification cannot drift apart.
        """
        if vix >= VIX_NUKED:
            return self.NUKED_ZONE

        above_ema50  = last["close"] > last["ema50"]
        ema50_rising = last["ema50"] > prev["ema50"]

        if above_ema50 and ema50_rising:
            return self.RECLAMATION
        return self.WASTELAND

    def describe(self, regime: str) -> str:
        return self._DESCRIPTIONS.get(regime, regime)

    def recommend_roles(self, regime: str,
                        stock_ind: "dict | pd.Series | None" = None) -> list[str]:
        """
        Return role codenames for this regime.

        stock_ind : optional per-stock indicators (must contain 'adx').
                    When provided, routes by the stock's own trend strength:
                      RECLAMATION + ADX >= ADX_RUNNER → ["maggie", "raider"]   (breakout scan + ride the trend)
                      RECLAMATION + ADX <  ADX_RUNNER → ["maggie", "scavenger"] (breakout scan + collect premium)
                      WASTELAND (any ADX)             → ["scavenger"] (income focus)
                      NUKED_ZONE (any ADX)            → ["chemist"]   (unchanged)
                    Maggie is always offered in RECLAMATION regardless of ADX —
                    a breakout can fire before ADX has caught up to confirm
                    the trend; her own scan() gates the entry.
                    Without stock_ind, returns the static regime mapping.
        """
        if stock_ind is None:
            return list(self._ROLES.get(regime, []))

        if regime == self.NUKED_ZONE:
            return list(self._ROLES[self.NUKED_ZONE])

        adx = float(stock_ind.get("adx", 0.0) if hasattr(stock_ind, "get")
                    else getattr(stock_ind, "adx", 0.0))

        if regime == self.RECLAMATION:
            return ["maggie", "raider"] if adx >= ADX_RUNNER else ["maggie", "scavenger"]

        if regime == self.WASTELAND:
            return ["scavenger"]

        return []


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

    parser = argparse.ArgumentParser()
    parser.add_argument("--spy-csv", default="data/spy_history.csv")
    args = parser.parse_args()

    spy_df = pd.read_csv(args.spy_csv, parse_dates=["datetime"])

    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "schwab"))
        from signal_verifier import fetch_vix
        vix = fetch_vix()
    except Exception:
        vix = 20.0
        print("  (VIX fetch failed — defaulting to 20.0)")

    overseer = Overseer()
    regime   = overseer.classify(spy_df, vix=vix)
    roles    = overseer.recommend_roles(regime)

    print(f"\n{'='*50}")
    print(f"  VAULT 76 — OVERSEER REPORT")
    print(f"{'='*50}")
    print(f"  VIX:     {vix:.1f}")
    print(f"  Regime:  {overseer.describe(regime)}")
    if roles:
        print(f"  Deploy:  {', '.join(roles)}")
    else:
        print(f"  Deploy:  NONE — stand down, Dweller")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
