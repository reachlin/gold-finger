"""
The Overseer — Vault 76 market regime classifier.

Reads the wasteland conditions and decides which perk cards to deploy.

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


class Overseer:
    RECLAMATION = "RECLAMATION"
    WASTELAND   = "WASTELAND"
    NUKED_ZONE  = "NUKED_ZONE"

    _DESCRIPTIONS = {
        RECLAMATION: "Reclamation Day — bull market, rebuilding in progress",
        WASTELAND:   "The Wasteland — bear/sideways, survival mode engaged",
        NUKED_ZONE:  "Nuked Zone — blast radius active, all cards benched",
    }

    # Which perk cards are active per regime
    _PERK_CARDS = {
        RECLAMATION: ["raider", "scavenger"],   # Raider attacks pullbacks; Scavenger sells puts on sideways stocks
        WASTELAND:   ["scavenger", "raider"],   # Scavenger primary for income; Raider opportunistic on any trending names
        NUKED_ZONE:  [],                        # Blast radius — all Dwellers stand down
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

        above_ema50  = last["close"] > last["ema50"]
        ema50_rising = last["ema50"] > prev["ema50"]

        if above_ema50 and ema50_rising:
            return self.RECLAMATION
        return self.WASTELAND

    def describe(self, regime: str) -> str:
        return self._DESCRIPTIONS.get(regime, regime)

    def recommend_perk_cards(self, regime: str) -> list[str]:
        """Return list of perk card codenames to deploy for this regime."""
        return list(self._PERK_CARDS.get(regime, []))


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
    cards    = overseer.recommend_perk_cards(regime)

    print(f"\n{'='*50}")
    print(f"  VAULT 76 — OVERSEER REPORT")
    print(f"{'='*50}")
    print(f"  VIX:     {vix:.1f}")
    print(f"  Regime:  {overseer.describe(regime)}")
    if cards:
        print(f"  Deploy:  {', '.join(cards)}")
    else:
        print(f"  Deploy:  NONE — stand down, Dweller")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
