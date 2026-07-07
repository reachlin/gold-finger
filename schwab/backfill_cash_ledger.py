"""
One-time backfill: replay the existing options ledger into cash_ledger.csv.

Run once after deploying cash_ledger.py to populate historical entries.
Safe to re-run — it checks if cash_ledger.csv already has non-STARTING_CAPITAL
rows and exits early.

Usage:
  /Users/lincai/anaconda3/envs/gold-finger/bin/python schwab/backfill_cash_ledger.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import options_ledger as ol
import cash_ledger as cl

LEDGER_PATH      = os.path.join(os.path.dirname(__file__), "..", "data", "paper_options_ledger.csv")
CASH_LEDGER_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "cash_ledger.csv")
STARTING_CAPITAL = 30_000.0


def main():
    ledger = cl.CashLedger(CASH_LEDGER_PATH, starting_capital=STARTING_CAPITAL)
    existing = [r for r in ledger.rows() if r["event_type"] != "STARTING_CAPITAL"]
    if existing:
        print(f"cash_ledger.csv already has {len(existing)} non-starting entries — skipping backfill.")
        print(f"Current balance: ${ledger.balance():,.2f}")
        return

    rows   = ol.read_rows(LEDGER_PATH)
    states = {}
    opens  = {}

    for r in rows:
        verdict = r.get("verdict", "")
        sig     = r.get("signal", "")

        if verdict == "APPROVED" and sig in ("SELL_PUT", "SELL_CALL"):
            key = ol.row_key(r)
            opens[key] = r
            try:
                prem_ct = float(r["premium_ct"])
            except (ValueError, KeyError):
                prem_ct = float(r.get("premium_sh", 0)) * 100
            ledger.record(
                "OPTION_SELL", r["symbol"], round(prem_ct, 2),
                f"{sig} {r['symbol']} strike={r.get('strike','?')} "
                f"opened {r['date'][:10]} premium=${prem_ct:.2f}"
            )
            print(f"  +${prem_ct:.2f}  OPTION_SELL  {r['symbol']} {r['date'][:10]}")

        elif verdict in ("CLOSED", "ASSIGNED"):
            ref     = r.get("ref") or ol.row_key(r)
            opening = opens.get(ref)
            pnl     = float(r.get("pnl") or 0)
            reason  = r.get("reason", "")
            sym     = r["symbol"]

            if "early_exit" in reason or "bought_back" in reason:
                premium_ct = float(opening["premium_ct"]) if opening else 0.0
                buyback    = premium_ct - pnl
                ledger.record("OPTION_BUYBACK", sym, -round(buyback, 2),
                              f"buyback {sym} {r['date'][:10]}: {reason}")
                print(f"  -${buyback:.2f}  OPTION_BUYBACK  {sym} {r['date'][:10]}")

            elif "assigned" in reason:
                if opening:
                    strike_cash = float(opening["strike"]) * 100
                    ledger.record("OPTION_ASSIGNED", sym, -round(strike_cash, 2),
                                  f"assigned {sym} at ${opening['strike']} {r['date'][:10]}")
                    print(f"  -${strike_cash:.2f}  OPTION_ASSIGNED  {sym} {r['date'][:10]}")

            elif "called_away" in reason or "shares_called_away" in reason:
                if opening:
                    strike_cash = float(opening["strike"]) * 100
                    ledger.record("OPTION_CALLED_AWAY", sym, round(strike_cash, 2),
                                  f"called away {sym} at ${opening['strike']} {r['date'][:10]}")
                    print(f"  +${strike_cash:.2f}  OPTION_CALLED_AWAY  {sym} {r['date'][:10]}")

            elif "expired" in reason:
                ledger.record("OPTION_EXPIRED", sym, 0.0,
                              f"expired worthless {sym} {r['date'][:10]}")
                print(f"  $0.00  OPTION_EXPIRED  {sym} {r['date'][:10]}")

    print(f"\nBackfill complete. Balance: ${ledger.balance():,.2f}")


if __name__ == "__main__":
    main()
