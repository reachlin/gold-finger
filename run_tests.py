#!/usr/bin/env python3
"""Run every test in the repo, and exit non-zero if anything fails.

Two suites, run differently for a reason:

  schwab/  -- one pytest process. Fast, and schwab/conftest.py grafts schwab-py
              onto the local schwab package so the whole directory collects.

  repo root -- ONE PROCESS PER FILE. These cannot share an interpreter: lightgbm
              and torch each bring their own OpenMP runtime, and loading both
              segfaults mid-run (lightgbm/basic.py _lazy_init). Every file passes
              alone, so the isolation is the fix. Setting KMP_DUPLICATE_LIB_OK to
              paper over it is deliberately NOT done -- duplicate OpenMP runtimes
              can silently produce wrong numbers, which is worse than a crash in
              code that prices real trades.

Why this script exists at all: the schwab suite was run with --ignore and the
root suite was not run, which let a genuinely failing test
(test_cover_skipped_when_already_covered, guarding against duplicate covers) sit
red until the bug reached production and sent 45 rejected orders on 2026-09-04.
A suite nobody can run in one command is a suite nobody runs.

    python run_tests.py            # everything
    python run_tests.py --schwab   # just the trading system
"""
import argparse
import glob
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
_SUMMARY = re.compile(r"(\d+) (passed|failed|skipped|error)")


def _tally(output: str) -> dict:
    out = {}
    for n, kind in _SUMMARY.findall(output):
        out[kind] = out.get(kind, 0) + int(n)
    return out


def _run(args: list, label: str) -> tuple:
    r = subprocess.run([PY, "-m", "pytest", *args, "-q"], cwd=HERE,
                       capture_output=True, text=True, timeout=1800)
    text = r.stdout + r.stderr
    crashed = bool(re.search(r"Segmentation fault|Fatal Python error", text))
    counts = _tally(text)
    # pytest exits 5 for "no tests collected", which is exactly what a
    # module-level pytest.importorskip produces. That is a skip, not a failure.
    ok = (r.returncode in (0, 5)) and not crashed
    if not ok:
        print(f"  ✗ {label}")
        tail = [ln for ln in text.splitlines() if ln.strip()][-12:]
        for ln in tail:
            print(f"      {ln}")
    return ok, counts, crashed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--schwab", action="store_true",
                    help="only the schwab/ trading suite")
    a = ap.parse_args()

    total = {"passed": 0, "failed": 0, "skipped": 0, "error": 0}
    failures = []

    print("== schwab/ (single process) ==")
    ok, counts, crashed = _run(["schwab"], "schwab/")
    for k, v in counts.items():
        total[k] = total.get(k, 0) + v
    if not ok:
        failures.append("schwab/")
    print(f"  {counts.get('passed', 0)} passed, {counts.get('skipped', 0)} skipped")

    if not a.schwab:
        print("\n== repo root (one process per file: OpenMP conflict) ==")
        for path in sorted(glob.glob(os.path.join(HERE, "test_*.py"))):
            name = os.path.basename(path)
            ok, counts, crashed = _run([name], name)
            for k, v in counts.items():
                total[k] = total.get(k, 0) + v
            if not ok:
                failures.append(name + (" [SEGFAULT]" if crashed else ""))
        print(f"  {total.get('passed', 0)} passed cumulative, "
              f"{total.get('skipped', 0)} skipped")

    print("\n" + "=" * 52)
    print(f"  passed {total.get('passed', 0)}   "
          f"skipped {total.get('skipped', 0)}   "
          f"failed {total.get('failed', 0) + total.get('error', 0)}")
    if failures:
        print(f"  FAILING: {', '.join(failures)}")
        return 1
    print("  ALL GREEN")
    return 0


if __name__ == "__main__":
    sys.exit(main())
