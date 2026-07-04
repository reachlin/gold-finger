"""
Deterministic capital allocator — plan item 4, step 2.

Before this, signals were processed in WATCHLIST order: first-come-first-
served claimed the collateral, so "two small caps vs one large" was decided
by list position, not by merit. This module gives each scan's batch a
deterministic processing order:

  1. Cash-freeing / zero-cash signals first — RESUME_WHEEL releases capital,
     SELL_CALL writes against shares already held.
  2. Cash-consuming option signals ranked by capital efficiency:
     premium per day per collateral dollar, halved for every position
     already open on the same symbol (concentration penalty).
  3. Everything else (BUY, HOLD_SHARES — small fixed budgets) in original
     order.

The AutoOverseer still approves/rejects each signal (with the full peer
list in its prompt — step 1); this module only decides who gets first
claim on the cash. Scores are deterministic so the policy is backtestable.
"""

CONCENTRATION_PENALTY = 0.5   # score multiplier per open position on the symbol
DEFAULT_BUDGET        = 600   # matches live_scanner.BUDGET_PER_TRADE

_FREE_SIGNALS = ("RESUME_WHEEL", "SELL_CALL")
_SCORED       = ("SELL_PUT",)


def cash_needed(signal: dict, budget_per_trade: float = DEFAULT_BUDGET) -> float:
    """Cash a signal would consume if approved."""
    sig = signal.get("signal")
    if sig == "SELL_PUT":
        return float(signal.get("strike", 0)) * 100
    if sig == "HOLD_SHARES":
        return float(signal.get("shares", 0)) * float(signal.get("close", 0))
    if sig == "BUY":
        return float(budget_per_trade)
    return 0.0   # SELL_CALL (covered by held shares), RESUME_WHEEL (frees cash)


def score(signal: dict, open_count: int = 0) -> float:
    """
    Capital efficiency: premium per day per collateral dollar, with a
    concentration penalty per position already open on the symbol.
    """
    strike  = float(signal.get("strike", 0) or 0)
    premium = float(signal.get("premium", 0) or 0)
    dte     = float(signal.get("dte", 30) or 30)
    if strike <= 0 or dte <= 0:
        return 0.0
    return (premium / (strike * dte)) * (CONCENTRATION_PENALTY ** open_count)


def rank_signals(signals: list[dict],
                 open_counts: "dict[str, int] | None" = None) -> list[dict]:
    """
    Deterministic processing order for a scan's signal batch. Never drops
    a signal — the AutoOverseer still judges every one; this only decides
    who gets first claim on the cash.
    """
    open_counts = open_counts or {}

    free, scored, rest = [], [], []
    for s in signals:
        sig = s.get("signal")
        if sig in _FREE_SIGNALS:
            free.append(s)
        elif sig in _SCORED:
            scored.append(s)
        else:
            rest.append(s)

    scored.sort(key=lambda s: score(s, open_counts.get(s.get("symbol"), 0)),
                reverse=True)
    return free + scored + rest
