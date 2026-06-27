"""
PerkCard — base class for all Vault 76 trading strategies.

Each weapon in the armory is a PerkCard with:
  - A codename and human name
  - A declared list of optimal regimes
  - A scan() method that returns a signal dict
  - A should_deploy() method checked by the Overseer
"""
from abc import ABC, abstractmethod
import pandas as pd


class PerkCard(ABC):
    codename:        str        # machine identifier, e.g. "scavenger"
    name:            str        # human name, e.g. "The Scavenger"
    optimal_regimes: list[str]  # regimes where this card performs best

    def should_deploy(self, regime: str) -> bool:
        """True if this card is active in the given regime."""
        return regime in self.optimal_regimes

    @abstractmethod
    def scan(self, symbol: str, df: pd.DataFrame,
             regime: str | None = None) -> dict:
        """
        Run signal detection on df (pre-computed indicators).
        Returns dict with at minimum: {"symbol", "signal": "BUY"|"NONE", "reason"}.
        BUY signals also include: entry, target, stop, rsi, adx.
        """

    def describe(self) -> str:
        return f"{self.name} [{self.codename}] — optimal: {', '.join(self.optimal_regimes)}"
