"""Let both `schwab` packages resolve in one test session.

This repo has schwab/__init__.py, so the local package is named exactly like the
installed schwab-py library. Only one name can win in sys.modules, and pytest
puts the repo root on sys.path (schwab/ is a package, so pytest prepends its
parent) -- so the local one wins and `import schwab.orders.options` fails.

That is why test_gtc_close.py was excluded from the suite with --ignore and run
by hand as a script instead, where sys.path[0] is schwab/ itself and `schwab`
therefore resolves to site-packages. The cost of that workaround was real: it
hid a genuinely failing test (test_cover_skipped_when_already_covered, a
duplicate-cover assertion) for long enough that the bug it guarded reached
production and sent 45 rejected orders on 2026-09-04.

Swapping sys.modules["schwab"] to schwab-py is not an option -- code here
imports the LOCAL package by that name too (schwab/market_intel.py does
`from schwab.nvda_trader import ...`, and the repo-root tests use
`from schwab.real_overseer import ...`). Both are needed.

So instead of choosing, graft schwab-py's directory onto the local package's
__path__. Submodule lookup then searches both: `schwab.orders` comes from
schwab-py, `schwab.nvda_trader` from this directory. Nothing is shadowed and no
test needs to know about it.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import schwab as _pkg  # noqa: E402  (the local package; repo root is on sys.path)


def _graft_schwab_py() -> None:
    """Append the installed schwab-py directory to the local package's path."""
    for _p in list(_pkg.__path__):
        if os.path.isdir(os.path.join(_p, "orders")):
            return                      # schwab-py already reachable
    for entry in sys.path:
        if not entry:
            continue
        cand = os.path.join(entry, "schwab")
        if (os.path.isdir(os.path.join(cand, "orders"))
                and os.path.abspath(cand) != _HERE):
            _pkg.__path__.append(cand)
            return


_graft_schwab_py()
