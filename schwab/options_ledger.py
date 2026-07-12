"""
Options ledger — single source of truth for paper option trades.

The ledger CSV (data/paper_options_ledger.csv) is append-only. Row lifecycle:

  APPROVED   → short option opened (SELL_PUT / SELL_CALL), 1 contract per row
  CLOSED     → position closed. `ref` links to the opening row's key and
               `reason` says why: expired_worthless, early_exit, called_away,
               shares_called_away. `pnl` carries the realized P&L per contract.
  ASSIGNED   → put expired ITM; 100 shares acquired at (strike - premium).
               The put is no longer an open option but its collateral stays
               committed — the cash became shares — until a CLOSED row with
               the same ref releases it (shares called away or sold).
  SKIPPED / BUDGET_BLOCK / RISKOFF_BLOCK → informational, never open.

Assigned shares awaiting covered calls ("wheel holdings") live in
data/paper_wheel_holdings.json:  {symbol: {contracts, cost_basis, put_key,
assigned_date}}. cost_basis already includes the collected put premium.

P&L accounting is additive across a wheel cycle and matches
backtest_scavenger.py:
  put expired worthless   pnl = put premium
  put early exit          pnl = premium - buyback value
  put assigned            pnl = 0 (premium baked into cost_basis)
  call expired worthless  pnl = call premium
  call early exit         pnl = premium - buyback value
  called away             pnl = (call_strike - cost_basis)·100 + call premium
"""
import os
import csv
import json
from datetime import date, datetime, timedelta

from options_pricer import (
    black_scholes_put, black_scholes_call,
    adaptive_profit_target, RISK_FREE_RATE,
)

FIELDNAMES = [
    "date", "symbol", "signal", "close", "strike",
    "premium_sh", "premium_ct", "premium_pct", "dte", "hv", "adx",
    "confidence", "regime", "verdict", "reason", "pnl", "ref",
]

_OPEN_VERDICT = "APPROVED"
SHARES_PER_CONTRACT = 100


def _now_et() -> datetime:
    from zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo("America/New_York"))


def row_key(row: dict) -> str:
    """Stable identifier of an opening row: symbol + open timestamp."""
    return f"{row['symbol']}_{row['date']}"


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------

def read_rows(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    # Backfill columns that predate this module
    for r in rows:
        for col in FIELDNAMES:
            if r.get(col) is None:
                r[col] = ""
        r.pop(None, None)
    return rows


def _migrate_header(path: str):
    """Rewrite a legacy-header ledger so it carries all FIELDNAMES columns."""
    rows = read_rows(path)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def append_row(path: str, row: dict):
    if os.path.exists(path):
        with open(path, newline="") as f:
            header = next(csv.reader(f), [])
        if header != FIELDNAMES:
            _migrate_header(path)
        write_header = False
    else:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        write_header = True

    out = {col: row.get(col, "") for col in FIELDNAMES}
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            w.writeheader()
        w.writerow(out)


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

def _states(rows: list[dict]) -> dict[str, dict]:
    """
    Map opening-row key → {"row": opening row, "status": open|assigned|closed}.
    CLOSED/ASSIGNED rows locate their opening row via `ref`; rows written
    before `ref` existed fall back to their own symbol_date key (legacy).
    """
    states: dict[str, dict] = {}
    for r in rows:
        verdict = r.get("verdict", "")
        if verdict == _OPEN_VERDICT and r.get("signal") in ("SELL_PUT", "SELL_CALL"):
            states[row_key(r)] = {"row": r, "status": "open"}
        elif verdict in ("CLOSED", "ASSIGNED"):
            key = r.get("ref") or row_key(r)
            if key in states:
                states[key]["status"] = ("assigned" if verdict == "ASSIGNED"
                                         else "closed")
    return states


def open_options(rows: list[dict], symbol: str | None = None) -> list[dict]:
    """Opening rows whose option is still live (not closed, not assigned)."""
    out = [s["row"] for s in _states(rows).values() if s["status"] == "open"]
    if symbol is not None:
        out = [r for r in out if r["symbol"] == symbol]
    return out


def committed_collateral(rows: list[dict]) -> float:
    """
    Cash locked by short puts: open puts (may be assigned any time) plus
    assigned puts (cash became shares, released only when shares leave).
    Covered calls need shares, not cash.
    """
    total = 0.0
    for s in _states(rows).values():
        r = s["row"]
        if r.get("signal") != "SELL_PUT" or s["status"] == "closed":
            continue
        try:
            total += float(r["strike"]) * SHARES_PER_CONTRACT
        except (ValueError, KeyError):
            pass
    return total


# ---------------------------------------------------------------------------
# Wheel holdings (assigned shares)
# ---------------------------------------------------------------------------

def load_holdings(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def save_holdings(path: str, holdings: dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(holdings, f, indent=2)


# ---------------------------------------------------------------------------
# Expiry / assignment / early-exit processing
# ---------------------------------------------------------------------------

def _expiry_date(row: dict) -> date:
    opened = date.fromisoformat(row["date"][:10])
    return opened + timedelta(days=int(float(row["dte"])))


def _close_row(opening: dict, reason: str, pnl: float, verdict: str = "CLOSED",
               now: datetime | None = None) -> dict:
    now = now or _now_et()
    return {
        "date":    now.strftime("%Y-%m-%d %H:%M:%S"),
        "symbol":  opening["symbol"],
        "signal":  opening["signal"],
        "strike":  opening["strike"],
        "dte":     opening["dte"],
        "verdict": verdict,
        "reason":  reason,
        "pnl":     round(pnl, 2),
        "ref":     row_key(opening),
    }


def process_expirations(ledger_path: str, holdings_path: str,
                        fetch_quote, today: date | None = None,
                        r: float = RISK_FREE_RATE) -> list[dict]:
    """
    Walk all open options once and settle whatever is due:

      - expiry reached, put OTM  → CLOSED expired_worthless (keep premium)
      - expiry reached, put ITM  → ASSIGNED; add shares to wheel holdings
      - expiry reached, call OTM → CLOSED expired_worthless (keep shares)
      - expiry reached, call ITM → called away: CLOSED for the call (share
        gain + premium) and CLOSED for the originating put (frees collateral)
      - before expiry: buy back early when the option's mark-to-market falls
        to adaptive_profit_target(entry_iv, entry_dte) of the entry premium

    fetch_quote(symbol) → {"close": float, "hv": float}  (hv annualized,
    fraction). A fetch failure leaves that symbol's positions untouched.

    Returns a list of event dicts for display/Slack:
      {"action", "symbol", "pnl", "detail"}
    """
    rows     = read_rows(ledger_path)
    holdings = load_holdings(holdings_path)
    today    = today or _now_et().date()
    events: list[dict] = []
    quotes: dict[str, dict | None] = {}

    for opening in open_options(rows):
        sym = opening["symbol"]
        if sym not in quotes:
            try:
                quotes[sym] = fetch_quote(sym)
            except Exception:
                quotes[sym] = None
        q = quotes[sym]
        if q is None:
            continue

        close  = float(q["close"])
        hv     = float(q.get("hv") or 0.0) or 0.30
        expiry = _expiry_date(opening)

        if today >= expiry:
            event = _settle_expiry(opening, close, holdings, today)
        else:
            event = _try_early_exit(opening, close, hv, expiry, today, r)

        if event is None:
            continue
        events.append(event)
        for out_row in event.pop("_rows"):
            append_row(ledger_path, out_row)

    save_holdings(holdings_path, holdings)
    return events


def _settle_expiry(opening: dict, close: float, holdings: dict,
                   today: date) -> dict:
    sym     = opening["symbol"]
    strike  = float(opening["strike"])
    premium = float(opening["premium_sh"])
    prem_ct = premium * SHARES_PER_CONTRACT

    if opening["signal"] == "SELL_PUT":
        if close > strike:
            row = _close_row(opening, "expired_worthless", prem_ct)
            return {"action": "put_expired", "symbol": sym, "pnl": prem_ct,
                    "detail": f"put ${strike} expired worthless, keep ${prem_ct:.0f}",
                    "_rows": [row]}

        # Assigned — shares acquired at (strike - premium) effective cost
        cost_basis = strike - premium
        h = holdings.get(sym)
        if h:   # stack onto an existing holding: average the cost basis
            n = h["contracts"]
            h["cost_basis"] = round((h["cost_basis"] * n + cost_basis) / (n + 1), 4)
            h["contracts"]  = n + 1
        else:
            holdings[sym] = {"contracts": 1, "cost_basis": round(cost_basis, 4),
                             "put_key": row_key(opening),
                             "assigned_date": today.isoformat()}
        row = _close_row(opening,
                         f"assigned at ${strike} — cost basis ${cost_basis:.2f}",
                         0.0, verdict="ASSIGNED")
        return {"action": "put_assigned", "symbol": sym, "pnl": 0.0,
                "detail": (f"assigned {SHARES_PER_CONTRACT} sh at ${strike}, "
                           f"cost basis ${cost_basis:.2f} — Scavenger phase 2"),
                "_rows": [row]}

    # SELL_CALL
    if close < strike:
        row = _close_row(opening, "expired_worthless", prem_ct)
        return {"action": "call_expired", "symbol": sym, "pnl": prem_ct,
                "detail": f"call ${strike} expired worthless, keep ${prem_ct:.0f}",
                "_rows": [row]}

    # Called away — shares sold at strike, wheel cycle complete
    h          = holdings.pop(sym, None)
    cost_basis = h["cost_basis"] if h else strike
    pnl        = (strike - cost_basis) * SHARES_PER_CONTRACT + prem_ct
    out_rows   = [_close_row(opening, "called_away", pnl)]
    if h and h.get("put_key"):
        release = _close_row(opening, "shares_called_away", 0.0)
        release["ref"]    = h["put_key"]
        release["signal"] = "SELL_PUT"
        out_rows.append(release)
    return {"action": "called_away", "symbol": sym, "pnl": pnl,
            "detail": (f"called away at ${strike} "
                       f"(cost basis ${cost_basis:.2f}) — wheel complete"),
            "_rows": out_rows}


def _try_early_exit(opening: dict, close: float, hv: float,
                    expiry: date, today: date, r: float) -> dict | None:
    strike     = float(opening["strike"])
    premium    = float(opening["premium_sh"])
    entry_dte  = int(float(opening["dte"]))
    # Entry IV: the hv column is stored as a percentage (e.g. "24.6")
    try:
        entry_iv = float(opening["hv"]) / 100
    except (ValueError, KeyError):
        entry_iv = hv

    days_left = (expiry - today).days
    T         = days_left / 365
    pricer    = (black_scholes_put if opening["signal"] == "SELL_PUT"
                 else black_scholes_call)
    cur_val   = pricer(close, strike, T, r, hv)
    target    = adaptive_profit_target(entry_iv, entry_dte)

    if cur_val > premium * target:
        return None

    pnl = (premium - cur_val) * SHARES_PER_CONTRACT
    row = _close_row(opening,
                     f"early_exit at {cur_val/premium*100:.0f}% of premium "
                     f"(target {target*100:.0f}%)", pnl)
    return {"action": "early_exit", "symbol": opening["symbol"], "pnl": pnl,
            "detail": (f"{opening['signal']} ${strike} bought back at "
                       f"${cur_val:.2f} vs ${premium:.2f} entry "
                       f"({days_left}d left)"),
            "_rows": [row]}


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def position_lines(rows: list[dict], holdings: dict,
                   today: date | None = None) -> tuple[list[str], list[str]]:
    """(terminal_lines, slack_lines) for open options + wheel holdings."""
    today = today or _now_et().date()
    opens = open_options(rows)

    term, slack = [], []
    if not opens:
        term.append("  Options:         none")
        slack.append("*Options:* none")
    else:
        term.append(f"  Options ({len(opens)} open):")
        slack.append(f"*Options ({len(opens)} open):*")
        for r in opens:
            sym, sig, strike = r["symbol"], r["signal"], r.get("strike", "?")
            prem   = r.get("premium_ct", "?")
            opened = r["date"][:10]
            try:
                exp = _expiry_date(r)
                exp_str = f"exp {exp.isoformat()} ({(exp - today).days}d left)"
            except Exception:
                exp_str = f"{r.get('dte', '?')} DTE at open"
            try:
                collat = (f"${float(strike) * SHARES_PER_CONTRACT:,.0f}"
                          if sig == "SELL_PUT" else "covered")
            except Exception:
                collat = "?"
            try:
                prem_str = f"${float(prem):.0f}"
            except Exception:
                prem_str = str(prem)
            term.append(f"    {sym:<6} {sig}  strike ${strike}  prem {prem_str}"
                        f"  collat {collat}  {exp_str}  [opened {opened}]")
            slack.append(f"  • {sym} {sig} ${strike} | prem {prem_str}"
                         f" | collat {collat} | {exp_str}")

    if holdings:
        term.append(f"  Wheel holdings ({len(holdings)}):")
        slack.append(f"*Wheel holdings ({len(holdings)}):*")
        for sym, h in holdings.items():
            shares = h["contracts"] * SHARES_PER_CONTRACT
            term.append(f"    {sym:<6} {shares} sh @ ${h['cost_basis']:.2f}"
                        f"  (assigned {h.get('assigned_date', '?')})"
                        f"  → selling covered calls")
            slack.append(f"  • {sym} {shares}sh @${h['cost_basis']:.2f}"
                         f" [assigned {h.get('assigned_date', '?')}]")
    return term, slack
