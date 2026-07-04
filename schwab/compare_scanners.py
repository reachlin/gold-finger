"""
Compare two auto-overseer paper-trading runs side by side.

Built for the A/B experiment started 2026-07-04: `overseer-main` (main
worktree, ~/dev/private/gold-finger-main) vs `overseer-pm` (this repo,
portfolio-management branch). Both start from identical ledgers; the only
code difference is the allocation work, so ledger/decision divergence is
the measurement.

Reads each side's data dir:
  - paper_options_ledger.csv  → verdict counts, approved premiums, positions
  - overseer.log              → every LLM decision with its reason

Reports per-side summaries, then day/symbol/action-matched decisions where
the two scanners DISAGREED, and signals only one side saw.

Usage:
  /Users/lincai/anaconda3/envs/gold-finger/bin/python schwab/compare_scanners.py
  /Users/lincai/anaconda3/envs/gold-finger/bin/python schwab/compare_scanners.py <dirA> <dirB>
"""
import os
import re
import sys
import argparse

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import options_ledger as ol

DEFAULT_A = os.path.join(os.path.dirname(__file__), "..", "data")
DEFAULT_B = os.path.join(os.path.dirname(__file__), "..", "..",
                         "gold-finger-main", "data")

_VERDICT_RE = re.compile(
    r"AutoOverseer \[[^\]]*\]:\s+\S+\s+(?P<verdict>[YN])\s+—\s+(?P<reason>.*\S)")


def parse_decisions(log_text: str) -> list[dict]:
    """
    Extract LLM decisions from an overseer.log: each ===SIGNAL_START===
    block's TIME/SYMBOL/ACTION, paired with the next AutoOverseer verdict
    line. Blocks without a verdict (interactive runs, crashes) are dropped.
    """
    decisions, current = [], None
    for line in log_text.splitlines():
        if "===SIGNAL_START===" in line:
            current = {}
            continue
        if current is not None and "verdict" not in current:
            if line.startswith("TIME:"):
                m = re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", line)
                current["time"] = m.group(0) if m else line[8:].strip()
            elif line.startswith("SYMBOL:"):
                current["symbol"] = line.split()[1]
            elif line.startswith("ACTION:"):
                current["action"] = line.split()[1]
        m = _VERDICT_RE.search(line)
        if m and current and "symbol" in current and "action" in current:
            decisions.append({"time":    current.get("time", "?"),
                              "symbol":  current["symbol"],
                              "action":  current["action"],
                              "verdict": m.group("verdict"),
                              "reason":  m.group("reason")})
            current = None
    return decisions


def ledger_summary(rows: list[dict]) -> dict:
    by_verdict: dict = {}
    approved_premium = 0.0
    approved_symbols = []
    for r in rows:
        v = r.get("verdict", "?")
        by_verdict[v] = by_verdict.get(v, 0) + 1
        if v == "APPROVED":
            approved_symbols.append(r.get("symbol", "?"))
            try:
                approved_premium += float(r.get("premium_ct", 0) or 0)
            except ValueError:
                pass
    return {"by_verdict": by_verdict,
            "approved_premium": round(approved_premium, 2),
            "approved_symbols": approved_symbols}


def _key(d: dict) -> tuple:
    return (d.get("time", "?")[:10], d["symbol"], d["action"])


def diff_decisions(a: list[dict], b: list[dict]) -> dict:
    """Match decisions by (day, symbol, action); report disagreements and
    signals only one side saw. Last decision wins if a side re-decided."""
    a_by = {_key(d): d for d in a}
    b_by = {_key(d): d for d in b}
    disagreements, only_a, only_b = {}, {}, {}
    for k in a_by.keys() | b_by.keys():
        da, db = a_by.get(k), b_by.get(k)
        if da and db:
            if da["verdict"] != db["verdict"]:
                disagreements[k] = (da, db)
        elif da:
            only_a[k] = da
        else:
            only_b[k] = db
    return {"disagreements": disagreements,
            "only_a": only_a, "only_b": only_b}


def _load_side(data_dir: str) -> dict:
    ledger = ol.read_rows(os.path.join(data_dir, "paper_options_ledger.csv"))
    log_path = os.path.join(data_dir, "overseer.log")
    log_text = ""
    if os.path.exists(log_path):
        with open(log_path, errors="replace") as f:
            log_text = f.read()
    return {"ledger": ledger,
            "summary": ledger_summary(ledger),
            "decisions": parse_decisions(log_text),
            "open": ol.open_options(ledger)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir_a", nargs="?", default=DEFAULT_A,
                    help="side A data dir (default: this repo's data/)")
    ap.add_argument("dir_b", nargs="?", default=DEFAULT_B,
                    help="side B data dir (default: ../gold-finger-main/data)")
    args = ap.parse_args()

    name_a = "A: " + os.path.abspath(args.dir_a)
    name_b = "B: " + os.path.abspath(args.dir_b)
    a, b = _load_side(args.dir_a), _load_side(args.dir_b)

    print("═" * 90)
    print("  SCANNER A/B COMPARISON")
    print(f"  {name_a}")
    print(f"  {name_b}")
    print("═" * 90)

    for name, side in (("A", a), ("B", b)):
        s = side["summary"]
        print(f"\n  ── Side {name} "
              f"({len(side['decisions'])} LLM decisions, "
              f"{len(side['ledger'])} ledger rows) ──")
        for v, n in sorted(s["by_verdict"].items()):
            print(f"     {v:14s} {n}")
        print(f"     approved premium  ${s['approved_premium']:,.2f}"
              f"   symbols: {', '.join(s['approved_symbols']) or '—'}")
        print(f"     open positions    {len(side['open'])}")

    d = diff_decisions(a["decisions"], b["decisions"])
    print(f"\n  ── Decision diff (matched by day/symbol/action) ──")
    print(f"     disagreements: {len(d['disagreements'])}"
          f"   only-A: {len(d['only_a'])}   only-B: {len(d['only_b'])}")
    for k, (da, db) in sorted(d["disagreements"].items()):
        print(f"\n     {k[0]}  {k[1]} {k[2]}")
        print(f"       A: {da['verdict']} — {da['reason']}")
        print(f"       B: {db['verdict']} — {db['reason']}")
    for label, side_only in (("only-A", d["only_a"]), ("only-B", d["only_b"])):
        for k, dd in sorted(side_only.items()):
            print(f"     {label}: {k[0]}  {k[1]} {k[2]}"
                  f"  → {dd['verdict']} — {dd['reason']}")
    print()
    print("  Note: one week of LLM decisions is noisy — treat differences as")
    print("  suggestive. Ledgers started identical on 2026-07-04.")


if __name__ == "__main__":
    main()
