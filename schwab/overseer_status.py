"""Reconciled overseer status — the correct way to answer "check overseer".

Motivation: reading `tail -n30 overseer.log` and inferring state is unreliable.
A single scan is ~80 log lines, so position opens/closes scroll out of any tail
window within minutes, and the operator ends up guessing. This tool instead
reads GROUND TRUTH — the live Schwab account (positions + cash) — and augments
it with state-change events pulled from the WHOLE log plus resting GTC orders.

Usage:
    python schwab/overseer_status.py
"""
import os, sys, json, subprocess, re
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(ROOT, "data", "overseer.log")
PENDING = os.path.join(ROOT, "data", "pending_orders.json")

# Events that change money/positions — the things a status check MUST surface.
# Kept to specific event phrases: loose words like "assigned"/"P&L" match the
# per-signal "MAX LOSS ... if assigned" and EOD "P&L $" boilerplate in EVERY
# scan and would bury the real events. Genuine assignment shows as a position
# change in the live account (ground truth) anyway.
EVENT_RE = re.compile(
    r"(closed on Schwab|Fill booked|REAL ORDER PLACED|BUY_TO_CLOSE fill detected|"
    r"resting cover|BUY_TO_CLOSE placed \(GTC\)|was ASSIGNED|"
    r"AUTH ERROR|invalid_grant|Traceback)")
# Defensive: never treat these recurring advisory lines as events.
EVENT_EXCLUDE_RE = re.compile(r"MAX LOSS|assignment risk|Kronos")


def parse_occ(sym: str):
    """'IBM   260918P00220000' -> ('IBM','2026-09-18','P',220.0)."""
    root = sym[:6].strip()
    body = sym[6:]
    yy, mm, dd = body[0:2], body[2:4], body[4:6]
    typ = body[6]
    strike = int(body[7:15]) / 1000.0
    return root, f"20{yy}-{mm}-{dd}", typ, strike


def short_options(positions: list[dict]) -> list[dict]:
    """Open short option legs (shortQuantity>0 and symbol is an OCC contract)."""
    out = []
    for p in positions:
        sym = p.get("symbol", "")
        if (p.get("shortQuantity") or 0) > 0 and len(sym) >= 15 and sym[6:].strip():
            out.append(p)
    return out


def committed_from_positions(positions: list[dict], covered_calls: bool = True) -> float:
    """Cash-secured collateral = sum(strike*100*qty) over short PUTS.

    Short calls are assumed covered by stock (no cash) when covered_calls=True.
    """
    total = 0.0
    for p in short_options(positions):
        _, _, typ, strike = parse_occ(p["symbol"])
        qty = p.get("shortQuantity") or 0
        if typ == "P":
            total += strike * 100 * qty
        elif typ == "C" and not covered_calls:
            total += strike * 100 * qty
    return total


def recent_events(lines: list[str], limit: int = 12) -> list[str]:
    """State-change events from the WHOLE log (not a tail), newest last."""
    hits = [ln.strip() for ln in lines
            if EVENT_RE.search(ln) and not EVENT_EXCLUDE_RE.search(ln)]
    return hits[-limit:]


def last_scan(lines: list[str]) -> dict:
    """Most recent 'END SCAN #N' marker."""
    num = None
    for ln in lines:
        m = re.search(r"END SCAN #(\d+)", ln)
        if m:
            num = int(m.group(1))
    return {"num": num}


def _proc_health() -> dict:
    def run(cmd):
        return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()
    alive = run("tmux has-session -t overseer 2>/dev/null && echo yes || echo no") == "yes"
    pid = run("ps aux | grep real_overseer.py | grep -v grep | grep -v tmux "
              "| grep -v caffeinate | grep -v tee | awk '{print $2}' | head -1")
    caff = run("ps aux | grep 'caffeinate -is' | grep -v grep | awk '{print $2}' | head -1")
    mtime = ""
    if os.path.exists(LOG):
        mtime = datetime.fromtimestamp(os.path.getmtime(LOG)).strftime("%Y-%m-%d %H:%M:%S")
    return {"alive": alive, "pid": pid, "caffeinate": caff, "log_mtime": mtime}


def main():
    from schwab_account import get_client
    h = _proc_health()
    print("=" * 60)
    print("  OVERSEER STATUS (reconciled: live account + log + pending)")
    print("=" * 60)
    status = "✅ ALIVE" if h["alive"] and h["pid"] else "⛔ DOWN"
    print(f"  Process : {status}  pid={h['pid'] or '-'}  caffeinate={h['caffeinate'] or 'MISSING'}")
    print(f"  Log     : last write {h['log_mtime']}")

    lines = open(LOG).read().splitlines() if os.path.exists(LOG) else []
    ls = last_scan(lines)
    print(f"  Scan    : #{ls['num']}")

    # GROUND TRUTH: live account
    try:
        client = get_client()
        r = client.get_accounts(fields=client.Account.Fields.POSITIONS)
        r.raise_for_status()
        sa = r.json()[0]["securitiesAccount"]
        bal = sa.get("currentBalances", {})
        positions = [dict(symbol=p["instrument"]["symbol"],
                          shortQuantity=p.get("shortQuantity", 0),
                          longQuantity=p.get("longQuantity", 0),
                          marketValue=p.get("marketValue"))
                     for p in sa.get("positions", [])]
        cash = bal.get("cashBalance") or 0
        committed = committed_from_positions(positions)
        print(f"\n  LIVE ACCOUNT {sa.get('accountNumber')}:")
        print(f"    Cash ${cash:,.2f}   committed ${committed:,.0f}   free ${cash-committed:,.0f}")
        shorts = short_options(positions)
        print(f"    Open option positions: {len(shorts)}")
        for p in shorts:
            root, exp, typ, strike = parse_occ(p["symbol"])
            print(f"      {root} {'PUT' if typ=='P' else 'CALL'} ${strike:g}  "
                  f"x{int(p['shortQuantity'])}  exp {exp}  mktval ${p.get('marketValue',0):,.0f}")
        longs = [p for p in positions if (p.get("longQuantity") or 0) > 0]
        for p in longs:
            print(f"      LONG {p['symbol']} x{int(p['longQuantity'])}  mktval ${p.get('marketValue',0):,.0f}")
    except Exception as exc:
        print(f"\n  ⚠ LIVE ACCOUNT FETCH FAILED: {exc}")

    # Resting GTC covers
    if os.path.exists(PENDING):
        try:
            pend = json.load(open(PENDING))
            print(f"\n  Resting orders ({len(pend)}):")
            for o in pend:
                print(f"    {o.get('trade_id')}  {o.get('symbol')} {o.get('signal')} "
                      f"${o.get('strike')}  limit ${o.get('limit')}  {o.get('duration')}")
        except Exception:
            pass

    print("\n  Recent state-change events (whole log):")
    for ev in recent_events(lines):
        print(f"    {ev}")
    print("=" * 60)


if __name__ == "__main__":
    main()
