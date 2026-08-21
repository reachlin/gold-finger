"""
Fully automatic REAL-money Vault 76 Overseer — LLM-driven trading decisions.

Successor to auto_overseer.py with paper/semi modes removed: the LLM approves
or rejects each signal, and the overseer places the Schwab order, confirms the
fill, books it to the options + cash ledgers, watches profit targets, and
closes positions — no human in the loop.

Safety gate: REALLY_REAL=true must be set in the environment (.env) or the
process refuses to start.

Bookkeeping rules (single source of truth — see cash_ledger.py header):
    open fill        APPROVED row      OPTION_SELL      +premium x 100
    buyback fill     CLOSED row        OPTION_BUYBACK   -(fill x 100)
    expired OTM      CLOSED row        OPTION_EXPIRED   $0
    assigned         ASSIGNED row      OPTION_ASSIGNED  -(strike x 100)
                     + shares recorded in wheel holdings at (strike - premium)

Usage:
    python schwab/real_overseer.py
    python schwab/real_overseer.py --provider deepseek
    python schwab/real_overseer.py --provider openai_compatible --model llama3 \
        --base-url http://localhost:11434/v1
"""
import os
import sys
import json
import time
import argparse
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo

# LightGBM and PyTorch each bundle their own OpenMP on macOS; training the
# LGBM advisories before loading Kronos/TimesFM deadlocked torch forever on
# an OMP barrier (kmp_flag_64::wait, 0% CPU — observed 2026-07-06, wedged
# both scanners all morning). Must be set before either library initializes.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_ET = ZoneInfo("America/New_York")

_DATA_DIR            = os.path.join(os.path.dirname(__file__), "..", "data")
PENDING_ORDERS_PATH  = os.path.join(_DATA_DIR, "pending_orders.json")
_TRADE_COUNTER_PATH  = os.path.join(_DATA_DIR, "trade_counter.json")
_VAULT8_SIGNALS_PATH = os.path.join(_DATA_DIR, "vault8_weekly_signals.json")
SLACK_MENTION        = "<@U02DQJ9KKFZ>"   # user's Slack ID for trade notifications

# Schwab order states that mean "still live on the book"
WORKING_STATUSES = ("WORKING", "QUEUED", "ACCEPTED", "PENDING_ACTIVATION")
# Terminal states that mean "this order will never fill"
DEAD_STATUSES    = ("REJECTED", "CANCELED", "EXPIRED", "REPLACED")


def _now_et() -> datetime:
    return datetime.now(_ET)


def _next_trade_id() -> str:
    """Return next sequential trade ID like T0001, T0002, …"""
    os.makedirs(_DATA_DIR, exist_ok=True)
    try:
        with open(_TRADE_COUNTER_PATH) as f:
            n = json.load(f).get("counter", 0)
    except Exception:
        n = 0
    n += 1
    with open(_TRADE_COUNTER_PATH, "w") as f:
        json.dump({"counter": n}, f)
    return f"T{n:04d}"


def _load_pending() -> list:
    if not os.path.exists(PENDING_ORDERS_PATH):
        return []
    with open(PENDING_ORDERS_PATH) as f:
        return json.load(f)


def _save_pending(orders: list):
    os.makedirs(_DATA_DIR, exist_ok=True)
    with open(PENDING_ORDERS_PATH, "w") as f:
        json.dump(orders, f, indent=2)


# ---------------------------------------------------------------------------
# System prompt for the Overseer LLM
# ---------------------------------------------------------------------------

OVERSEER_SYSTEM = """You are the Vault 76 Overseer AI, reviewing options trade signals for a wheel strategy (cash-secured puts → assignment → covered calls).

## Your role
Approve or reject each SELL_PUT signal. Goal: capture steady premium income while avoiding assignment risk.

## Hard rules (auto-skip — do not override)
- Budget check already confirmed collateral is available — you do not need to recheck
- Fast risk-off conditions are already filtered — trust the pre-screening

## Duplicate positions
Multiple open positions on the same symbol are ALLOWED — the budget check
guarantees cash covers every contract. Do not reject a signal merely because
a position on the symbol already exists. The prompt lists existing open
positions on this symbol; use them only to judge concentration risk (e.g.
several near-strike puts stacked before the same expiry on a volatile name).

## Soft rules (use your judgment)
- SKIP if HV > 60% (too volatile for premium selling — gamma risk too high)
- SKIP if ADX > 28 (stock is trending, not sideways — wrong strategy)
- SKIP if Kronos buffer < -2% (stock predicted to fall below strike — elevated assignment risk)
- SKIP if yield/DTE < 0.02% per day (not enough premium for the capital tie-up)
- APPROVE if HV 15–50%, ADX < 22, yield > 0.03%/day, quality company

## Capital allocation (when "Other pending signals" are listed)
Cash may not cover every candidate this scan — approving this signal can
forfeit a better one. It is VALID to reject an otherwise-acceptable signal
to keep collateral free for a stronger peer. Prefer:
- higher capital efficiency: premium/day per collateral dollar
- proven-edge names: UNH, AMZN, IBM top the backtests; KO and PG are
  cheap collateral but negative-edge — cheap is not good
- diversification: avoid stacking correlated/same-sector names whose
  puts would all be assigned in the same drawdown
- laddered expiries and smaller positions over one big lump of collateral
- real-chain quotes (quote_source: schwab_chain) over model estimates
Keep roughly 10% of cash unreserved for rolls and adjustments.

## Router signals: HOLD_SHARES / RESUME_WHEEL
These come from the backtested wheel-vs-hold router: when the TimesFM 30d
forecast crosses the threshold, the scanner proposes holding shares uncapped
instead of wheeling (HOLD_SHARES), or selling the held shares and resuming
the wheel (RESUME_WHEEL). This mechanical policy is the validated one
(2015-2026 and 2024+ backtests) — the default is APPROVE. Veto ONLY for
context the models cannot see: earnings within ~5 trading days, pending
corporate action or trading halt, or an obvious macro shock today. Do not
veto merely because the forecast seems optimistic — that judgment is
already priced into the threshold.

## Medic signals: BUY_ETF / SELL_ETF
The Medic buys quality dividend ETFs (SCHD, VIG, VYM) at NUKED_ZONE panic
prices and sells them when RECLAMATION confirms recovery. Backtested
2015-2026: 15-16 winning episodes of 19 per ETF. Default is APPROVE —
veto only for context the rule can't see (ETF-specific halt, obviously
broken market plumbing). These are small budget positions, not collateral.

## Signal field guide
- quote_source: "schwab_chain" = real premium/IV/delta from the live chain
  (trust these numbers); "model" = Black-Scholes estimate on historical vol
  (treat premium as approximate)
- assign_risk_pct: on SELL_PUT — LGBM probability (%) of a >5% drop within
  30 trading days, trained per symbol on daily history. Advisory. Weigh it
  together with the Kronos buffer: both bearish → lean SKIP; if they
  disagree, judge from IV/ADX/context. Above ~50% is elevated.
- drop_risk_pct: on BUY — same drop model. A pullback entry with high drop
  risk may be a breakdown, not a dip → lean SKIP above ~50%.
- called_away_pct: on SELL_CALL — LGBM probability (%) of a >8% rally within
  30 trading days, i.e. shares called away at the strike. Being called away
  locks in the wheel profit, so moderate values are fine; very high values
  mean the call likely caps a strong run — prefer SKIP if cost basis is far
  below the strike and the trend is strong.
- model_auc: chronological holdout AUC of the LGBM model behind the field
  above (0.5 = coin flip). Discount the advisory when AUC is below ~0.6;
  trust it more above ~0.7.
- timesfm_30d_pct: TimesFM (zero-shot foundation model) forecast of the
  SMA5 change over the next 30 trading days, in %. Independent second
  opinion next to Kronos: both bearish on a SELL_PUT → stronger SKIP;
  strongly positive on a SELL_CALL → shares likely called away / upside
  capped. Treat as a trend hint, not a price target.

## Confidence score (pre-computed)
Each signal includes a composite confidence score (0–100) already weighing
ADX, RSI, HV, assignment risk, premium yield, and forward models:
- ≥70 HIGH — strong setup across all indicators, lean YES
- 40–69 MED — acceptable but imperfect, use other fields to decide
- <40 LOW  — weak setup, lean NO unless a compelling override exists
Use the confidence score as your primary input. Do not re-derive it from scratch.

BUY_CALL exception: backtest shows HIGH-scored BUY_CALL signals can fail when
macro timing is poor (technically perfect VCP in a deteriorating environment).
For BUY_CALL, treat MED (40–69) as the sweet spot and HIGH (≥70) as a caution
flag — verify the regime is firmly RECLAMATION and sector is also trending.

## Response format — ONLY valid JSON, no other text
{"decision": "yes", "reason": "brief reason under 15 words"}
or
{"decision": "no", "reason": "brief reason under 15 words"}"""


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

# Fields worth showing for a competing signal — enough to judge capital
# efficiency and risk, without the full signal dump
PEER_FIELDS = ("signal", "strike", "premium", "premium_pct", "dte",
               "assign_risk_pct", "called_away_pct", "timesfm_30d_pct",
               "shares", "entry", "cost_per_ct", "sector_ok")


def peer_summaries(all_signals: list[dict], current: dict) -> list[dict]:
    """Compact summaries of the scan's OTHER signals — the candidates
    competing with `current` for the same cash."""
    peers = []
    for s in all_signals:
        if s is current:
            continue
        p = {"symbol": s.get("symbol", "?")}
        for k in PEER_FIELDS:
            if s.get(k) is not None:
                p[k] = s[k]
        if s.get("signal") == "SELL_PUT" and s.get("strike"):
            p["collateral"] = float(s["strike"]) * 100
        peers.append(p)
    return peers


def build_prompt(signal: dict, portfolio_state: dict, kronos: dict,
                 open_positions: list[dict] | None = None,
                 peer_signals: list[dict] | None = None,
                 vault8_range: dict | None = None) -> str:
    lines = ["## Signal"]
    # Confidence score at the top — primary decision input
    conf = signal.get("confidence")
    if conf is not None:
        from signal_confidence import tier as _tier
        lines.append(f"  confidence: {conf}/100  [{_tier(conf)}]")
    for k, v in signal.items():
        if k == "confidence":
            continue                # already shown above
        lines.append(f"  {k}: {v}")

    lines.append("\n## Portfolio state")
    lines.append(f"  Cash available for collateral: ${portfolio_state.get('available', 0):,.0f}")
    lines.append(f"  Collateral required:           ${portfolio_state.get('required', 0):,.0f}")

    if peer_signals:
        lines.append(f"\n## Other pending signals this scan ({len(peer_signals)})"
                     f" — competing for the same cash")
        for p in peer_signals:
            parts = [f"{p.get('symbol', '?')} {p.get('signal', '?')}"]
            if p.get("strike") is not None:
                parts.append(f"strike ${p['strike']}")
            if p.get("premium") is not None:
                parts.append(f"premium ${p['premium']}/sh")
            if p.get("premium_pct") is not None:
                parts.append(f"({p['premium_pct']}%)")
            if p.get("dte") is not None:
                parts.append(f"{p['dte']} DTE")
            if p.get("collateral") is not None:
                parts.append(f"collateral ${p['collateral']:,.0f}")
            if p.get("assign_risk_pct") is not None:
                parts.append(f"assign_risk {p['assign_risk_pct']}%")
            if p.get("timesfm_30d_pct") is not None:
                parts.append(f"timesfm_30d {p['timesfm_30d_pct']:+.1f}%")
            lines.append("  - " + "  ".join(parts))

    sym = signal.get("symbol", "?")
    if open_positions:
        lines.append(f"\n## Existing open positions on {sym} ({len(open_positions)})")
        for p in open_positions:
            lines.append(f"  {p.get('signal', '?')} strike ${p.get('strike', '?')}"
                         f"  premium ${p.get('premium_ct', '?')}/ct"
                         f"  opened {str(p.get('date', '?'))[:10]}"
                         f"  {p.get('dte', '?')} DTE at open")
    else:
        lines.append(f"\n## Existing open positions on {sym}: none")

    if kronos:
        lines.append("\n## Kronos 30-day AI price prediction")
        lines.append(f"  Predicted support floor: ${kronos.get('support', 0):.2f}")
        lines.append(f"  Predicted resistance:    ${kronos.get('resistance', 0):.2f}")
        lines.append(f"  Strike buffer vs floor:  {kronos.get('buf_pct', 0):+.1f}%")
        warn = kronos.get("warn", False)
        lines.append(f"  Assignment warning:      {'YES — strike above predicted support' if warn else 'NO'}")

    if vault8_range:
        lines.append("\n## Vault 8 — Weekly range prediction (BiLSTM)")
        lines.append(f"  Predicted low  (entry):  ${vault8_range.get('entry', 0):.2f}")
        lines.append(f"  Predicted high (target): ${vault8_range.get('target', 0):.2f}")
        lines.append(f"  Predicted range:         {vault8_range.get('pred_range_pct', 0):.1f}%")
        lines.append(f"  Base compression:        {vault8_range.get('base_range', 0):.1f}%")
        lines.append(f"  Vault 8 confidence:      {vault8_range.get('confidence', 0)}/100")
        lines.append(f"  Stop:                    ${vault8_range.get('stop', 0):.2f}")

    lines.append("\nApprove or skip this trade?")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def parse_llm_response(text: str) -> tuple[str, str]:
    """Parse LLM JSON response. Returns ("no", reason) on any failure.

    Handles common LLM wrapping patterns:
      - Raw JSON: {"decision": "yes", ...}
      - Markdown fenced: ```json\\n{...}\\n```
      - Preamble text followed by JSON on its own line
    """
    import re
    if not text:
        return "no", "empty response"

    # Strip markdown fences and surrounding whitespace
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip()

    # Try the whole cleaned string first
    candidates = [cleaned]

    # Also try extracting the first {...} block in case of preamble/postamble
    m = re.search(r"\{[^{}]*\}", cleaned, re.DOTALL)
    if m:
        candidates.append(m.group())

    for candidate in candidates:
        try:
            parsed = json.loads(candidate.strip())
            decision = str(parsed.get("decision", "no")).lower().strip()
            reason   = str(parsed.get("reason", "no reason given")).strip()
            if decision not in ("yes", "no"):
                return "no", f"invalid decision value '{decision}' — defaulting to no"
            return decision, reason
        except (json.JSONDecodeError, Exception):
            continue

    return "no", f"JSON parse failed — defaulting to no (raw: {text[:80]!r})"


# ---------------------------------------------------------------------------
# OCC option symbols
# ---------------------------------------------------------------------------

def build_occ_symbol(symbol: str, exp_date, signal: str, strike: float) -> str:
    """
    OCC option symbol: {ROOT:<6}{YYMMDD}{C|P}{strike*1000:08d}
    signal: SELL_PUT → P, SELL_CALL → C.
    """
    root     = symbol.ljust(6)
    exp_str  = exp_date.strftime("%y%m%d")
    put_call = "C" if signal == "SELL_CALL" else "P"
    return f"{root}{exp_str}{put_call}{int(strike * 1000):08d}"


_OCC_RE = None


def parse_occ_symbol(occ: str) -> tuple[str, date, str, float]:
    """Inverse of build_occ_symbol: returns (root, expiry, "P"|"C", strike).
    Accepts both the padded 21-char form ("KO    260814P00078000") and the
    space-stripped form ("KO260814P00078000") — roots are variable length,
    so fixed offsets don't work on normalized symbols."""
    global _OCC_RE
    if _OCC_RE is None:
        import re
        _OCC_RE = re.compile(r"^([A-Z.$]{1,6})\s*(\d{6})([CP])(\d{8})$")
    m = _OCC_RE.match(occ.strip())
    if not m:
        raise ValueError(f"unparseable OCC symbol: {occ!r}")
    root, ymd, pc, strike_raw = m.groups()
    expiry = datetime.strptime(ymd, "%y%m%d").date()
    return root, expiry, pc, int(strike_raw) / 1000.0


# ---------------------------------------------------------------------------
# Schwab API helpers (shared by ordering / reconcile / auto-close)
# ---------------------------------------------------------------------------

def get_account_hash(client) -> str | None:
    try:
        accts = client.get_account_numbers().json()
        if not accts:
            return None
        return accts[0]["hashValue"]
    except Exception as exc:
        print(f"  [Overseer] ⚠ get_account_numbers failed: {exc}")
        return None


def fetch_account(client, account_hash: str) -> dict | None:
    """securitiesAccount dict (positions + balances), or None on error."""
    try:
        resp = client.get_account(account_hash,
                                  fields=client.Account.Fields.POSITIONS)
        resp.raise_for_status()
        return resp.json().get("securitiesAccount", {})
    except Exception as exc:
        print(f"  [Overseer] ⚠ Could not fetch Schwab account: {exc}")
        return None


def available_funds(balances: dict) -> float | None:
    """
    Free cash for new collateral. availableFunds is already net of option
    collateral requirements; cashBalance is NOT (it includes locked
    collateral) and must never be used for a can-we-afford-this check —
    auto_overseer checked cashBalance first and would have overdrafted.

    This account reports availableFundsNonMarginableTrade=0.0 alongside the
    real availableFunds value, so a 0.0 must not shadow a nonzero sibling;
    all-zero means genuinely $0 free.
    """
    values = [float(balances[k])
              for k in ("availableFundsNonMarginableTrade", "availableFunds")
              if balances.get(k) is not None]
    if not values:
        return None
    avail = next((v for v in values if v > 0), values[0])
    # Trade SETTLED cash only. Schwab grants provisional buying power the instant
    # an ACH deposit is initiated, so availableFunds includes still-in-flight
    # money (pendingDeposits). Subtract it; when the transfer lands pendingDeposits
    # drops to 0 and the funds become usable automatically — no restart needed.
    pending = float(balances.get("pendingDeposits", 0) or 0)
    return avail - pending


def fetch_orders(client, account_hash: str, days_back: int = 7) -> list:
    """All orders entered in the last `days_back` days (one call per scan)."""
    try:
        now  = datetime.now()
        resp = client.get_orders_for_account(
            account_hash,
            from_entered_datetime=now - timedelta(days=days_back),
            to_entered_datetime=now + timedelta(minutes=5),
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        print(f"  [Overseer] ⚠ Could not fetch Schwab orders: {exc}")
        return []


def place_order_with_retry(client, account_hash: str, order,
                           attempts: int = 4, base_delay: float = 1.5):
    """
    Place an order, retrying on HTTP 429 (rate limit) with exponential backoff.

    Opening + immediately resting a GTC close fires several order/quote calls
    within a couple of seconds, which trips Schwab's rate limiter (observed:
    the close order after an XOM open failed with 429). A 429 means the request
    was rejected, not accepted, so retrying is safe — it cannot double-place.

    Returns the successful response. Re-raises the last error on other failures
    or once attempts are exhausted.
    """
    last_exc = None
    for i in range(attempts):
        try:
            resp = client.place_order(account_hash, order)
            resp.raise_for_status()
            return resp
        except Exception as exc:
            last_exc = exc
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status == 429 and i < attempts - 1:
                delay = base_delay * (2 ** i)
                print(f"  [Overseer] ⏳ 429 rate-limited placing order — "
                      f"retry {i + 1}/{attempts - 1} in {delay:.1f}s")
                time.sleep(delay)
                continue
            raise
    raise last_exc     # pragma: no cover — loop always returns or raises


def parse_entered_time(order: dict) -> datetime | None:
    """
    Schwab returns enteredTime as an ISO8601 string like
    "2026-07-28T13:00:08+0000" — NOT epoch milliseconds (auto_overseer
    divided it by 1000, which raised and silently disabled recency
    filtering). Returns an aware datetime or None.
    """
    raw = order.get("enteredTime")
    if not raw or not isinstance(raw, str):
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S.%f%z"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def order_legs(order: dict):
    for leg in order.get("orderLegCollection", []):
        yield (leg.get("instruction", ""),
               leg.get("instrument", {}).get("symbol", "").replace(" ", ""))


def order_fill_price(order: dict) -> float | None:
    acts = order.get("orderActivityCollection", [])
    if acts:
        exec_legs = acts[0].get("executionLegs", [])
        if exec_legs and exec_legs[0].get("price") is not None:
            return float(exec_legs[0]["price"])
    if order.get("price") is not None:
        return float(order["price"])
    return None


def find_order(orders: list, occ_norm: str | None = None,
               order_id: str | None = None,
               instruction: str | None = None,
               since: datetime | None = None) -> dict | None:
    """First order matching the given filters (order_id wins when set)."""
    for o in orders:
        if order_id is not None:
            if str(o.get("orderId", "")) == str(order_id):
                return o
            continue
        if since is not None:
            entered = parse_entered_time(o)
            if entered is not None and entered < since:
                continue
        for inst, leg_sym in order_legs(o):
            if occ_norm is not None and leg_sym != occ_norm:
                continue
            if instruction is not None and inst != instruction:
                continue
            return o
    return None


# ---------------------------------------------------------------------------
# Bookkeeping — the ONLY functions that write CLOSED/ASSIGNED/APPROVED rows
# and cash-ledger entries. Amounts are cash movements, never P&L.
# ---------------------------------------------------------------------------

def book_open_fill(ledger_path: str, cash_ledger, s: dict, fill_price: float):
    """Confirmed SELL_TO_OPEN fill: APPROVED ledger row + premium credited."""
    import options_ledger as ol
    premium_ct = round(fill_price * 100, 2)
    ol.append_row(ledger_path, {
        "date":        _now_et().strftime("%Y-%m-%d %H:%M:%S"),
        "symbol":      s["symbol"],
        "signal":      s["signal"],
        "close":       s.get("close", ""),
        "strike":      s.get("strike", ""),
        "premium_sh":  fill_price,
        "premium_ct":  premium_ct,
        "premium_pct": s.get("premium_pct", ""),
        "dte":         s.get("dte", ""),
        "hv":          s.get("hv", ""),
        "adx":         s.get("adx", ""),
        "confidence":  s.get("confidence", ""),
        "regime":      s.get("regime", ""),
        "verdict":     "APPROVED",
        "reason":      f"real fill confirmed @ ${fill_price:.2f}",
    })
    if cash_ledger is not None:
        cash_ledger.record(
            "OPTION_SELL", s["symbol"], premium_ct,
            f"{s['signal']} {s['symbol']} strike=${s.get('strike')} "
            f"exp={s.get('dte', '?')}d premium=${premium_ct:.2f} (real fill)"
        )


def book_buyback(ledger_path: str, cash_ledger, opening: dict, fill_price: float):
    """BUY_TO_CLOSE fill: CLOSED row carries P&L; cash entry is the COST
    paid (negative) — the premium was already credited at open."""
    import options_ledger as ol
    from options_ledger import _close_row
    entry_prem = float(opening["premium_sh"])
    pnl  = round((entry_prem - fill_price) * 100, 2)
    cost = round(fill_price * 100, 2)
    ol.append_row(ledger_path, _close_row(
        opening,
        f"early_exit bought back at ${fill_price:.2f} vs ${entry_prem:.2f} entry",
        pnl,
    ))
    if cash_ledger is not None:
        cash_ledger.record(
            "OPTION_BUYBACK", opening["symbol"], -cost,
            f"BUY_TO_CLOSE {opening['signal']} ${opening['strike']} "
            f"bought back at ${fill_price:.2f} vs ${entry_prem:.2f} entry"
        )
    return pnl


def book_expired(ledger_path: str, cash_ledger, opening: dict):
    """Option expired worthless: full premium is the P&L; no cash moves."""
    import options_ledger as ol
    from options_ledger import _close_row
    pnl = round(float(opening.get("premium_ct", 0) or 0), 2)
    ol.append_row(ledger_path, _close_row(opening, "expired_worthless", pnl))
    if cash_ledger is not None:
        cash_ledger.record(
            "OPTION_EXPIRED", opening["symbol"], 0.0,
            f"{opening['signal']} ${opening['strike']} expired worthless "
            f"(premium already credited)"
        )
    return pnl


def book_assigned(ledger_path: str, cash_ledger, opening: dict,
                  holdings_path: str):
    """Put assigned: ASSIGNED row, collateral spent on shares, shares
    recorded in wheel holdings at (strike - premium) cost basis."""
    import options_ledger as ol
    from options_ledger import _close_row, row_key, SHARES_PER_CONTRACT
    strike  = float(opening["strike"])
    premium = float(opening["premium_sh"])
    ol.append_row(ledger_path, _close_row(
        opening,
        f"assigned at ${strike:.2f} — {SHARES_PER_CONTRACT} shares acquired",
        0.0, verdict="ASSIGNED",
    ))
    if cash_ledger is not None:
        cash_ledger.record(
            "OPTION_ASSIGNED", opening["symbol"], -round(strike * 100, 2),
            f"{opening['signal']} ${strike} assigned — paid for "
            f"{SHARES_PER_CONTRACT} shares"
        )
    # Record shares for the covered-call side of the wheel
    try:
        with open(holdings_path) as f:
            holdings = json.load(f)
    except Exception:
        holdings = {}
    sym        = opening["symbol"]
    cost_basis = round(strike - premium, 4)
    h = holdings.get(sym)
    if h:   # stack onto an existing holding: average the cost basis
        n = h["contracts"]
        h["cost_basis"] = round((h["cost_basis"] * n + cost_basis) / (n + 1), 4)
        h["contracts"]  = n + 1
    else:
        holdings[sym] = {"contracts": 1, "cost_basis": cost_basis,
                         "put_key": row_key(opening),
                         "assigned_date": str(_now_et().date())}
    with open(holdings_path, "w") as f:
        json.dump(holdings, f, indent=2)


# ---------------------------------------------------------------------------
# RealOverseer
# ---------------------------------------------------------------------------

class RealOverseer:
    """
    LLM decision engine + order placement + bookkeeping for real trading.

    Wire it up:
        ov = RealOverseer(...)
        live_scanner.set_decision_fn(ov.decide)
        live_scanner.set_scan_hook_fn(ov.scan_hook)
    """

    def __init__(self, provider: str | None = None, model: str | None = None,
                 api_key: str | None = None, base_url: str | None = None):
        from llm_client import LLMClient
        self.llm = LLMClient(provider=provider, model=model,
                             api_key=api_key, base_url=base_url)
        print(f"  [Overseer] LLM: {self.llm}  mode=real (fully automated)")

    # ------------------------------------------------------------------ #
    # Decision                                                           #
    # ------------------------------------------------------------------ #

    def decide(self, s: dict) -> str:
        """Decision callback for live_scanner.set_decision_fn()."""
        import live_scanner as scanner

        # Cash: the Schwab-synced available funds (set every scan by
        # scan_hook). Never a hardcoded constant — auto_overseer assumed
        # $30,000 forever in real mode.
        available = getattr(scanner, "_schwab_available", None)
        if available is None:
            cl = getattr(scanner, "_cash_ledger", None)
            total     = cl.balance() if cl else 30_000.0
            available = total - scanner._committed_collateral()
        required = scanner._collateral_required(s)
        portfolio_state = {"available": available, "required": required}

        # Kronos advisory
        kronos_cache = getattr(scanner, "_current_kronos_cache", {})
        kronos = {}
        sym = s.get("symbol", "")
        if sym in kronos_cache:
            entry  = kronos_cache[sym]
            strike = float(s.get("strike", 0))
            kronos = {
                "support":    entry.get("support", 0),
                "resistance": entry.get("resistance", 0),
                "buf_pct":    entry.get("buf_pct", 0),
                "warn":       strike > entry.get("support", 0),
            }

        # Existing open positions on this symbol (concentration context)
        try:
            import options_ledger as ol
            open_positions = ol.open_options(
                ol.read_rows(scanner.OPTION_LEDGER_PATH), symbol=sym)
        except Exception:
            open_positions = []

        peers        = peer_summaries(getattr(scanner, "_current_scan_signals", []), s)
        vault8_range = _load_vault8_range(sym)

        prompt = build_prompt(s, portfolio_state, kronos,
                              open_positions=open_positions,
                              peer_signals=peers,
                              vault8_range=vault8_range)
        raw = self.llm.chat(system=OVERSEER_SYSTEM, user=prompt)
        decision, reason = parse_llm_response(raw) if raw else ("no", "LLM returned empty response")

        verdict = "y" if decision == "yes" else "n"
        icon    = "✅" if verdict == "y" else "⛔"
        print(f"\n  🤖 Overseer [{self.llm.provider}/{self.llm.model}]: "
              f"{icon} {verdict.upper()}  —  {reason}")

        if verdict == "y":
            self._place_order(s)
        return verdict

    # ------------------------------------------------------------------ #
    # Order placement                                                    #
    # ------------------------------------------------------------------ #

    def _pre_trade_check(self, client, account_hash: str, s: dict) -> tuple[bool, str]:
        """Verify AVAILABLE funds cover collateral and no duplicate short
        exists. Uses availableFunds (net of locked collateral) — never
        cashBalance."""
        sec = fetch_account(client, account_hash)
        if sec is None:
            return False, "Cannot fetch Schwab account"

        avail = available_funds(sec.get("currentBalances", {}))
        if avail is None:
            return False, "Schwab balances missing availableFunds — abort"
        collateral_needed = float(s.get("strike", 0)) * 100
        if avail < collateral_needed:
            return False, (f"Schwab available ${avail:,.0f} < "
                           f"collateral needed ${collateral_needed:,.0f} — abort")

        # Duplicate short position check
        signal   = s.get("signal", "SELL_PUT")
        put_call = "PUT" if signal == "SELL_PUT" else "CALL"
        strike   = float(s["strike"])
        symbol   = s["symbol"].upper()
        for pos in sec.get("positions", []):
            inst = pos.get("instrument", {})
            if inst.get("assetType") != "OPTION":
                continue
            if (inst.get("putCall", "").upper() == put_call
                    and abs(float(inst.get("strikePrice", -1)) - strike) < 0.01
                    and float(pos.get("shortQuantity", 0)) > 0
                    and symbol in inst.get("symbol", "").upper()):
                return False, (f"Duplicate: Schwab already has short "
                               f"{inst.get('symbol')} — skipping")

        return True, (f"Pre-check OK — Schwab available ${avail:,.0f}, "
                      f"collateral needed ${collateral_needed:,.0f}")

    def _place_order(self, s: dict):
        """
        Place a real Schwab SELL_TO_OPEN order, then either book the
        confirmed fill or leave it tracked in pending_orders.json for the
        scan hook to confirm on a later cycle — never fire-and-forget.

        Limit price: live BID from the chain quote (guaranteed fill);
        falls back to premium x 0.95 for model-priced signals.
        """
        if s.get("signal") not in ("SELL_PUT", "SELL_CALL"):
            print(f"  [Overseer] {s.get('signal')} — no automated real "
                  f"order; manage the share trade manually")
            return
        if os.environ.get("REALLY_REAL", "").lower() != "true":
            print("  [Overseer] Real order suppressed — REALLY_REAL is not 'true'")
            return

        import live_scanner as scanner
        client = getattr(scanner, "_current_client", None)
        if client is None:
            print("  [Overseer] ⚠ No Schwab client available — cannot place order")
            return
        account_hash = get_account_hash(client)
        if not account_hash:
            return

        ok, msg = self._pre_trade_check(client, account_hash, s)
        print(f"  [Pre-trade] {msg}")
        if not ok:
            return

        try:
            from schwab.orders.options import option_sell_to_open_limit
            from schwab.orders.common import Duration, Session

            if s.get("expiry"):
                exp_date = date.fromisoformat(s["expiry"])
            else:
                exp_date = date.today() + timedelta(days=int(s.get("dte", 30)))
            occ_sym = build_occ_symbol(s["symbol"], exp_date,
                                       s.get("signal", "SELL_PUT"),
                                       float(s["strike"]))

            bid     = s.get("bid")
            premium = float(s.get("premium", 0))
            if bid and float(bid) > 0 and s.get("quote_source") == "schwab_chain":
                limit = round(float(bid), 2)
                price_src = f"bid ${limit:.2f}"
            else:
                limit = round(premium * 0.95, 2)
                price_src = f"model×0.95 ${limit:.2f}"

            order = (
                option_sell_to_open_limit(occ_sym, 1, limit)
                .set_duration(Duration.DAY)
                .set_session(Session.NORMAL)
                .build()
            )

            collat = int(float(s.get("strike", 0)) * 100)
            scanner._send_slack(
                f"*🤖 AUTO ORDER SUBMITTING*\n"
                f"  {occ_sym}  limit {price_src}  collat ~${collat:,}\n"
                f"  conf={s.get('confidence', '?')}  reason: {s.get('reason', '—')}"
            )

            resp = place_order_with_retry(client, account_hash, order)
            location        = resp.headers.get("Location", "")
            schwab_order_id = location.rstrip("/").split("/")[-1] if location else None
            print(f"  [Overseer] ✅ REAL ORDER PLACED: {occ_sym}  "
                  f"limit {price_src}  (mid ${premium:.2f}  ask ${s.get('ask', '?')})")
        except Exception as exc:
            print(f"  [Overseer] ⚠ Real order failed: {exc}")
            scanner._send_slack(f"{SLACK_MENTION} ❌ *Order placement failed* — "
                                f"{s['symbol']} {s['signal']}\n{exc}")
            return

        # Track it — the scan hook confirms + books whenever it fills
        trade_id = _next_trade_id()
        pending  = _load_pending()
        pending.append({
            "trade_id":        trade_id,
            "symbol":          s["symbol"],
            "signal":          s["signal"],
            "strike":          float(s["strike"]),
            "expiry":          exp_date.isoformat(),
            "dte":             int(s.get("dte", 30)),
            "limit":           limit,
            "occ_sym":         occ_sym,
            "schwab_order_id": schwab_order_id,
            "notified_at":     _now_et().strftime("%Y-%m-%d %H:%M:%S"),
            "signal_data":     {k: v for k, v in s.items()
                                if isinstance(v, (str, int, float, bool, type(None)))},
        })
        _save_pending(pending)

        # Poll briefly for an immediate fill (bid-priced orders usually fill
        # in seconds); no alarm if it doesn't — the scan hook keeps watching.
        for _ in range(6):   # 6 x 5s = 30s
            time.sleep(5)
            o = find_order(fetch_orders(client, account_hash, days_back=1),
                           order_id=schwab_order_id) if schwab_order_id else None
            if o and o.get("status") == "FILLED":
                self._confirm_open_fill(client, account_hash, o)
                return
        print(f"  [Overseer] order working — will confirm on a later scan")

    def _confirm_open_fill(self, client, account_hash: str, order: dict):
        """Book a filled opening order and drop it from pending."""
        import live_scanner as scanner
        order_id = str(order.get("orderId", ""))
        pending  = _load_pending()
        entry    = next((e for e in pending
                         if str(e.get("schwab_order_id", "")) == order_id), None)
        if entry is None:
            return
        fill = order_fill_price(order) or float(entry["limit"])
        book_open_fill(scanner.OPTION_LEDGER_PATH,
                       getattr(scanner, "_cash_ledger", None),
                       entry["signal_data"] | {"symbol": entry["symbol"],
                                               "signal": entry["signal"],
                                               "strike": entry["strike"],
                                               "dte":    entry["dte"]},
                       fill)
        _save_pending([e for e in pending
                       if str(e.get("schwab_order_id", "")) != order_id])
        premium_ct = fill * 100
        print(f"  [Overseer] ✅ Fill booked: {entry['symbol']} {entry['signal']} "
              f"${entry['strike']} @ ${fill:.2f}")
        scanner._send_slack(
            f"{SLACK_MENTION} ✅ *ORDER FILLED — {entry.get('trade_id', '')}*\n"
            f"*{entry['symbol']} {entry['signal']}* ${entry['strike']} "
            f"exp {entry['expiry']}\n"
            f"*Fill:* ${fill:.2f}/sh  (${premium_ct:.0f}/contract)\n"
            f"*Collateral committed:* ${entry['strike'] * 100:,.0f}"
        )

        # Rest a GTC buy-to-close at the profit target immediately, so decay is
        # captured with no monitoring gap and no daily DAY-order churn. The
        # profit-target sweep is now just a fallback for positions without one.
        if entry["signal"] in ("SELL_PUT", "SELL_CALL"):
            try:
                from options_pricer import adaptive_profit_target
                sd = entry.get("signal_data", {}) or {}
                try:
                    entry_iv = float(sd.get("hv", 30)) / 100
                except (ValueError, TypeError):
                    entry_iv = 0.30
                try:                       # true entry IV (%) for v2 IV-crush TP
                    entry_iv_pct = float(sd.get("iv")) if sd.get("iv") else None
                except (ValueError, TypeError):
                    entry_iv_pct = None
                dte          = int(float(entry.get("dte", 30)))
                target_price = round(fill * adaptive_profit_target(entry_iv, dte), 2)
                try:
                    days_left = max(
                        (date.fromisoformat(entry["expiry"]) - _now_et().date()).days, 0)
                except Exception:
                    days_left = dte
                self._submit_close_order(
                    scanner, client, account_hash,
                    symbol=entry["symbol"], strike=float(entry["strike"]),
                    expiry=entry["expiry"], days_left=days_left,
                    occ_sym=entry["occ_sym"], entry_prem=fill,
                    target_price=target_price, opening_signal=entry["signal"],
                    opening_ref=entry.get("trade_id"), trigger="open",
                    entry_iv=entry_iv_pct,
                )
            except Exception as exc:
                print(f"  [Overseer] ⚠ could not rest GTC close for "
                      f"{entry['symbol']}: {exc}")

    def _submit_close_order(self, scanner, client, account_hash: str, *,
                            symbol: str, strike: float, expiry, days_left: int,
                            occ_sym: str, entry_prem: float, target_price: float,
                            mark: float | None = None,
                            opening_signal: str = "SELL_PUT",
                            opening_ref=None, trigger: str = "target",
                            entry_iv: float | None = None):
        """
        Place a GOOD_TILL_CANCEL BUY_TO_CLOSE limit at ``target_price`` and
        track it in the pending list.

        GTC (not DAY) means the take-profit order rests at Schwab across
        sessions until it fills — no nightly cancel/re-place churn. Callers:
          • _confirm_open_fill  (trigger="open") — rest it the moment the
            short option fills, so decay is captured with no monitoring gap.
          • _check_profit_targets (trigger="target") — fallback for any open
            position that has no resting close yet (e.g. positions opened
            before this feature, or a prior placement that failed).

        Idempotent: if a BUY_TO_CLOSE for this contract is already pending it
        does nothing and returns None. Returns the trade_id on success.
        """
        occ_norm = occ_sym.replace(" ", "")
        for e in _load_pending():
            if (e.get("signal") == "BUY_TO_CLOSE"
                    and e.get("occ_sym", "").replace(" ", "") == occ_norm):
                return None    # already have a resting close for this contract

        try:
            from schwab.orders.options import option_buy_to_close_limit
            from schwab.orders.common import Duration, Session
            order = (
                option_buy_to_close_limit(occ_sym, 1, f"{target_price:.2f}")
                .set_duration(Duration.GOOD_TILL_CANCEL)
                .set_session(Session.NORMAL)
                .build()
            )
            resp = place_order_with_retry(client, account_hash, order)
            location = resp.headers.get("Location", "")
            schwab_order_id = location.rstrip("/").split("/")[-1] if location else None
        except Exception as exc:
            print(f"  [AutoClose] ❌ BUY_TO_CLOSE failed: {exc}")
            scanner._send_slack(f"{SLACK_MENTION} ❌ *BUY_TO_CLOSE failed* — "
                                f"{symbol} ${strike}\n{exc}")
            return None

        trade_id = _next_trade_id()
        pending  = _load_pending()
        pending.append({
            "trade_id":        trade_id,
            "symbol":          symbol,
            "signal":          "BUY_TO_CLOSE",
            "strike":          strike,
            "expiry":          expiry if isinstance(expiry, str) else expiry.isoformat(),
            "dte":             days_left,
            "limit":           target_price,
            "occ_sym":         occ_sym,
            "schwab_order_id": schwab_order_id,
            "notified_at":     _now_et().strftime("%Y-%m-%d %H:%M:%S"),
            "duration":        "GTC",     # rests across sessions — do not day-expire
            "opening_ref":     opening_ref,
            "entry_iv":        entry_iv,  # % at open; used by v2 IV-crush early-TP
        })
        _save_pending(pending)

        est_pnl  = round((entry_prem - target_price) * 100, 2)
        where    = "resting on open" if trigger == "open" else "target near"
        markline = f"*Mark:* ${mark:.2f}  " if mark else ""
        scanner._send_slack(
            f"{SLACK_MENTION} 🎯 *BUY_TO_CLOSE placed (GTC) — {trade_id}*\n"
            f"*{symbol} ${strike} ({opening_signal})* — {where}\n"
            f"{markline}*Target:* ${target_price:.2f}  *Entry:* ${entry_prem:.2f}\n"
            f"*Limit:* ${target_price:.2f} (GTC)  |  DTE: {days_left}\n"
            f"*Est. P&L at fill:* +${est_pnl:,.2f}"
        )
        print(f"  [AutoClose] ✅ BUY_TO_CLOSE placed (GTC): {trade_id}  "
              f"{symbol} ${strike}  limit ${target_price:.2f}  "
              f"est P&L +${est_pnl:.2f}")
        return trade_id

    # ------------------------------------------------------------------ #
    # Scan hook — runs at the start of every scan cycle                  #
    # ------------------------------------------------------------------ #

    def scan_hook(self):
        import live_scanner as scanner
        client = getattr(scanner, "_current_client", None)
        if client is None:
            return
        account_hash = get_account_hash(client)
        if not account_hash:
            return

        sec = fetch_account(client, account_hash)
        if sec is None:
            return
        orders = fetch_orders(client, account_hash, days_back=7)

        try:
            occ_map = self._reconcile(scanner, sec, orders)
        except Exception as exc:
            print(f"  [Reconcile] ⚠ reconcile failed: {exc}")
            occ_map = {}
        try:
            self._process_pending(scanner, client, account_hash, orders)
        except Exception as exc:
            print(f"  [Overseer] ⚠ pending check failed: {exc}")
        try:
            self._check_profit_targets(scanner, client, account_hash,
                                       orders, occ_map)
        except Exception as exc:
            print(f"  [Overseer] ⚠ auto-close check failed: {exc}")

    # -- 1. Schwab is the source of truth ------------------------------ #

    def _reconcile(self, scanner, sec: dict, orders: list) -> dict:
        """
        Sync cash, detect closes/expiry/assignment, alert on unknown
        positions. Returns {ledger row_key: actual Schwab OCC symbol} for
        every still-open position (used to quote/close with the REAL symbol
        instead of reconstructing it from date+dte).
        """
        import options_ledger as ol

        balances = sec.get("currentBalances", {})
        avail    = available_funds(balances)
        total    = balances.get("cashBalance")
        scanner._schwab_cash      = float(total) if total is not None else None
        scanner._schwab_available = avail

        positions = sec.get("positions", [])
        option_positions = {}     # occ_norm -> original (spaced) Schwab symbol
        equity_long      = {}     # symbol   -> long shares
        for pos in positions:
            inst = pos.get("instrument", {})
            if inst.get("assetType") == "OPTION":
                qty = float(pos.get("shortQuantity", 0)) + float(pos.get("longQuantity", 0))
                if qty > 0:
                    raw = inst.get("symbol", "")
                    option_positions[raw.replace(" ", "")] = raw
            elif inst.get("assetType") in ("EQUITY", "COLLECTIVE_INVESTMENT"):
                equity_long[inst.get("symbol", "")] = float(pos.get("longQuantity", 0))

        working = [o for o in orders if o.get("status") in WORKING_STATUSES]
        if working:
            descs = [f"{inst} {sym} @${o.get('price', '?')}"
                     for o in working for inst, sym in order_legs(o)]
            print(f"  [Reconcile] Schwab total=${scanner._schwab_cash or 0:,.0f}  "
                  f"available=${avail or 0:,.0f}  open orders: {', '.join(descs)}")
        else:
            print(f"  [Reconcile] Schwab total=${scanner._schwab_cash or 0:,.0f}  "
                  f"available=${avail or 0:,.0f}  no open orders")

        rows  = ol.read_rows(scanner.OPTION_LEDGER_PATH)
        opens = list(ol.open_options(rows))
        cash_ledger = getattr(scanner, "_cash_ledger", None)
        today_et    = _now_et().date()

        occ_map: dict[str, str] = {}
        for opening in opens:
            sym      = opening["symbol"]
            strike   = float(opening["strike"])
            put_call = "C" if opening["signal"] == "SELL_CALL" else "P"

            # Match by root+type+strike against REAL Schwab symbols — the
            # date+dte reconstruction drifts when a fill was logged late.
            match = None
            for occ_norm, occ_raw in option_positions.items():
                root, _, pc, pos_strike = parse_occ_symbol(occ_norm)
                if root == sym and pc == put_call and abs(pos_strike - strike) < 0.01:
                    match = occ_raw
                    break
            if match:
                occ_map[ol.row_key(opening)] = match
                continue     # still open on Schwab — all good

            # Gone from Schwab. Closed by our order? Assigned? Expired?
            expiry = (date.fromisoformat(opening["date"][:10])
                      + timedelta(days=int(float(opening["dte"]))))
            occ_guess = build_occ_symbol(sym, expiry, opening["signal"],
                                         strike).replace(" ", "")
            closing = find_order(orders, occ_norm=occ_guess,
                                 instruction="BUY_TO_CLOSE")
            if closing and closing.get("status") == "FILLED":
                fill = order_fill_price(closing) or 0.0
                pnl  = book_buyback(scanner.OPTION_LEDGER_PATH, cash_ledger,
                                    opening, fill)
                print(f"  [Reconcile] {sym} ${strike} closed on Schwab "
                      f"(fill ${fill:.2f}, P&L ${pnl:+.2f}) — ledger updated")
                scanner._send_slack(
                    f"{SLACK_MENTION} 📋 *Closed: {sym} ${strike} "
                    f"{opening['signal']}*\nBought back at ${fill:.2f} — "
                    f"P&L ${pnl:+,.2f}"
                )
            elif today_et >= expiry:
                if equity_long.get(sym, 0) >= 100 and opening["signal"] == "SELL_PUT":
                    book_assigned(scanner.OPTION_LEDGER_PATH, cash_ledger,
                                  opening, scanner.WHEEL_HOLDINGS_PATH)
                    print(f"  [Reconcile] {sym} ${strike} ASSIGNED — "
                          f"100 shares acquired, ledger + holdings updated")
                    scanner._send_slack(
                        f"{SLACK_MENTION} 📥 *ASSIGNED: {sym} ${strike}*\n"
                        f"100 shares acquired at cost basis "
                        f"${strike - float(opening['premium_sh']):.2f} — "
                        f"covered calls now available"
                    )
                else:
                    pnl = book_expired(scanner.OPTION_LEDGER_PATH, cash_ledger,
                                       opening)
                    print(f"  [Reconcile] {sym} ${strike} expired worthless "
                          f"(P&L ${pnl:+.2f}) — ledger updated")
                    scanner._send_slack(
                        f"📋 *Expired worthless: {sym} ${strike} "
                        f"{opening['signal']}* — premium ${pnl:,.2f} kept"
                    )
            else:
                # Vanished mid-life with no closing order of ours — do NOT
                # guess a booking (auto_overseer wrote "expired worthless"
                # here, corrupting the ledger). Flag for manual review.
                print(f"  [Reconcile] ⚠ {sym} ${strike} gone from Schwab "
                      f"before expiry with no closing order — manual review")
                scanner._send_slack(
                    f"{SLACK_MENTION} ⚠ *{sym} ${strike} {opening['signal']} "
                    f"missing on Schwab* before {expiry} with no closing "
                    f"order — review manually; ledger NOT updated."
                )

        # Schwab option positions we don't know about (skip ones our own
        # pending opening orders explain — the fill may simply not be
        # confirmed yet this cycle)
        known   = {v.replace(" ", "") for v in occ_map.values()}
        pending = {e.get("occ_sym", "").replace(" ", "") for e in _load_pending()}
        for occ in option_positions:
            if occ not in known and occ not in pending:
                print(f"  [Reconcile] ⚠ Schwab has option {occ} not in local "
                      f"ledger — manual trade?")
                scanner._send_slack(
                    f"{SLACK_MENTION} ⚠ *Unknown Schwab position:* `{occ}`\n"
                    f"Not in local ledger — was this placed manually?"
                )
        return occ_map

    # -- 2. Pending orders: confirm fills, drop dead orders ------------ #

    def _process_pending(self, scanner, client, account_hash: str, orders: list):
        pending = _load_pending()
        if not pending:
            return

        keep = []
        for entry in pending:
            order = find_order(orders, order_id=entry.get("schwab_order_id"))
            if order is None:
                # Fall back to OCC + recency matching (order id lost)
                notified = None
                try:
                    notified = datetime.strptime(
                        entry["notified_at"], "%Y-%m-%d %H:%M:%S"
                    ).replace(tzinfo=_ET)
                except Exception:
                    pass
                order = find_order(orders,
                                   occ_norm=entry["occ_sym"].replace(" ", ""),
                                   since=notified)

            status = order.get("status", "") if order else ""
            if status == "FILLED":
                if entry["signal"] in ("SELL_PUT", "SELL_CALL"):
                    self._confirm_open_fill(client, account_hash, order)
                else:   # BUY_TO_CLOSE — booked by _reconcile when the
                        # position vanishes; just stop tracking it here
                    print(f"  [Overseer] ✅ BUY_TO_CLOSE fill detected: "
                          f"{entry['symbol']} ${entry['strike']} — "
                          f"booked by Schwab sync")
                continue
            if status in DEAD_STATUSES:
                reason = order.get("statusDescription", status)
                print(f"  [Overseer] ❌ Order {status}: {entry['symbol']} "
                      f"{entry['signal']} — {reason}")
                scanner._send_slack(
                    f"{SLACK_MENTION} ❌ *Order {status} — "
                    f"{entry.get('trade_id', '')}*\n{entry['symbol']} "
                    f"{entry['signal']} ${entry['strike']}\nReason: {reason}"
                )
                continue
            keep.append(entry)

        # Auto-expire entries notified before today's 4am ET cutoff (DAY
        # orders died at yesterday's close; Schwab may purge them entirely)
        cutoff = _now_et().replace(hour=4, minute=0, second=0, microsecond=0)
        live, expired = [], []
        for entry in keep:
            # GTC orders (e.g. resting buy-to-close targets) do not die at the
            # close — keep tracking them until Schwab reports fill/cancel.
            if entry.get("duration") == "GTC":
                live.append(entry)
                continue
            try:
                notified = datetime.strptime(
                    entry["notified_at"], "%Y-%m-%d %H:%M:%S"
                ).replace(tzinfo=_ET)
                (expired if notified < cutoff else live).append(entry)
            except Exception:
                live.append(entry)

        if expired:
            lines = "\n".join(f"  • {e['symbol']} {e['signal']} ${e['strike']}"
                              f"  (placed {e['notified_at']})" for e in expired)
            scanner._send_slack(f"{SLACK_MENTION} ⏰ *Orders expired unfilled:*\n"
                                f"{lines}\nRemoved from watch list.")
            print(f"  [Overseer] {len(expired)} order(s) expired unfilled")

        _save_pending(live)
        if live:
            print(f"  [Overseer] {len(live)} order(s) awaiting fill: "
                  + ", ".join(f"{e['symbol']} {e['signal']} ${e['strike']}"
                              for e in live))

    # -- 3. Profit targets → BUY_TO_CLOSE ------------------------------ #

    def _check_profit_targets(self, scanner, client, account_hash: str,
                              orders: list, occ_map: dict):
        import options_ledger as ol
        from options_pricer import adaptive_profit_target

        rows  = ol.read_rows(scanner.OPTION_LEDGER_PATH)
        opens = list(ol.open_options(rows))
        if not opens:
            return

        pending_close = {e["occ_sym"].replace(" ", "") for e in _load_pending()
                         if e.get("signal") == "BUY_TO_CLOSE"}
        working_close = {sym for o in orders
                         if o.get("status") in WORKING_STATUSES
                         for inst, sym in order_legs(o)
                         if inst == "BUY_TO_CLOSE"}
        today_et = _now_et().date()

        for opening in opens:
            key        = ol.row_key(opening)
            occ_spaced = occ_map.get(key)    # real (spaced) Schwab symbol
            if occ_spaced is None:
                continue    # not confirmed open on Schwab this cycle
            occ_norm = occ_spaced.replace(" ", "")
            if occ_norm in pending_close or occ_norm in working_close:
                # Already protected by a resting GTC. Most days: leave it alone
                # (no churn). But if it's a FAST winner, tighten that GTC to a
                # marketable price to bank the IV-crush gain early instead of
                # grinding out the last of the decay. Atomic replace only.
                self._maybe_early_take_profit(
                    scanner, client, account_hash, opening, occ_spaced, occ_norm)
                continue

            # Every open short should always carry a resting GTC buy-to-close
            # at its profit target. It can only ever fill in our favor (we buy
            # back cheaper than we sold), and it stays live at Schwab even if
            # this process is down or the OAuth token has expired — so a
            # weekend price spike gets captured with no human in the loop.
            # Therefore place one for ANY uncovered position, regardless of how
            # far the mark is from target (no "near target" gate). The
            # pending_close/working_close checks above already prevent dupes.
            entry_prem = float(opening["premium_sh"])
            try:
                entry_iv = float(opening.get("hv", 30)) / 100
            except (ValueError, TypeError):
                entry_iv = 0.30
            dte          = int(float(opening["dte"]))
            target_price = round(entry_prem * adaptive_profit_target(entry_iv, dte), 2)

            # Live mark is best-effort — for the log line only. A quote failure
            # must NOT stop us from placing the protective cover.
            mark = None
            try:
                resp = client.get_quotes([occ_spaced])
                resp.raise_for_status()
                q    = resp.json().get(occ_spaced, {}).get("quote", {})
                mark = float(q.get("mark") or q.get("markPrice") or 0) or None
            except Exception as exc:
                print(f"  [AutoClose] ⚠ quote fetch failed for {occ_spaced}: {exc}")

            _, expiry, _, strike = parse_occ_symbol(occ_norm)
            days_left = max((expiry - today_et).days, 0)
            if mark is not None:
                where = "target hit" if mark <= target_price else "resting cover"
                progress = (f"mark=${mark:.2f} "
                            f"({(1 - mark / entry_prem) * 100:.0f}% profit so far)")
            else:
                where, progress = "resting cover", "mark=n/a"
            print(f"\n  [AutoClose] 🎯 {where}: {opening['symbol']} "
                  f"${strike} target=${target_price:.2f} entry=${entry_prem:.2f} "
                  f"{progress}")

            self._submit_close_order(
                scanner, client, account_hash,
                symbol=opening["symbol"], strike=strike, expiry=expiry,
                days_left=days_left, occ_sym=occ_spaced, entry_prem=entry_prem,
                target_price=target_price, mark=mark,
                opening_signal=opening["signal"], opening_ref=key,
                trigger="target",
            )

    def _maybe_early_take_profit(self, scanner, client, account_hash: str,
                                 opening: dict, occ_spaced: str, occ_norm: str):
        """Dynamic early take-profit for a position that already has a resting
        GTC cover: if it's a FAST winner (>= EARLY_TP_MIN_PROFIT captured within
        EARLY_TP_MAX_DAYS), tighten that GTC to a marketable price to bank the
        IV-crush gain now instead of waiting out the full decay.

        Safety: uses an ATOMIC replace_order (never a bare cancel), so the
        protective cover is never absent — if the swap fails, the original
        deep-target GTC stays live. One-shot per position (early_tp flag) and
        only ever raises the buy-back price, so it never churns or loosens the
        floor. See options_pricer.should_take_early_profit (v1 fixed thresholds,
        made market/stock-aware later)."""
        from options_pricer import should_take_early_profit

        pending = _load_pending()
        entry = next((e for e in pending
                      if e.get("signal") == "BUY_TO_CLOSE"
                      and e.get("occ_sym", "").replace(" ", "") == occ_norm), None)
        if entry is None or entry.get("early_tp"):
            return                          # nothing to tighten, or already done
        old_order_id = entry.get("schwab_order_id")
        if not old_order_id:
            return                          # can't replace without an order id

        entry_prem = float(opening["premium_sh"])
        try:                                # ledger 'date' = open timestamp
            open_dt   = datetime.strptime(opening["date"][:10], "%Y-%m-%d").date()
            days_held = max((_now_et().date() - open_dt).days, 0)
        except Exception:
            return                          # unknown open date → don't risk it

        mark = cur_iv = None
        try:
            resp = client.get_quotes([occ_spaced])
            resp.raise_for_status()
            q    = resp.json().get(occ_spaced, {}).get("quote", {})
            mark   = float(q.get("mark") or q.get("markPrice") or 0) or None
            cur_iv = float(q.get("volatility") or 0) or None   # current IV (%)
        except Exception:
            return                          # no quote → leave the cover as-is
        entry_iv = entry.get("entry_iv")    # % stored at open (None for old covers)
        if not should_take_early_profit(entry_prem, mark, days_held,
                                        entry_iv=entry_iv, current_iv=cur_iv):
            return

        new_target = round(mark, 2)
        old_limit  = float(entry.get("limit", 0) or 0)
        if new_target <= old_limit + 0.05:  # only tighten UP; never churn/loosen
            return

        try:
            from schwab.orders.options import option_buy_to_close_limit
            from schwab.orders.common import Duration, Session
            new_order = (
                option_buy_to_close_limit(occ_spaced, 1, f"{new_target:.2f}")
                .set_duration(Duration.GOOD_TILL_CANCEL)
                .set_session(Session.NORMAL)
                .build()
            )
            resp   = client.replace_order(account_hash, old_order_id, new_order)
            loc    = resp.headers.get("Location", "")
            new_id = loc.rstrip("/").split("/")[-1] if loc else old_order_id
        except Exception as exc:
            print(f"  [EarlyTP] ❌ replace failed for {opening['symbol']} "
                  f"(original GTC intact): {exc}")
            return

        entry["limit"]           = new_target
        entry["schwab_order_id"] = new_id
        entry["early_tp"]        = True
        _save_pending(pending)

        profit_pct = (1 - mark / entry_prem) * 100
        est_pnl    = round((entry_prem - new_target) * 100, 2)
        print(f"  [EarlyTP] ⚡ fast winner: {opening['symbol']} "
              f"{profit_pct:.0f}% in {days_held}d — tightened GTC "
              f"${old_limit:.2f} → ${new_target:.2f} (bank +${est_pnl:.0f})")
        scanner._send_slack(
            f"{SLACK_MENTION} ⚡ *Early take-profit — {entry.get('trade_id','?')}*\n"
            f"*{opening['symbol']} ${entry.get('strike','?')}* reached "
            f"{profit_pct:.0f}% profit in {days_held}d (IV crush).\n"
            f"Tightened GTC ${old_limit:.2f} → ${new_target:.2f} to bank it now "
            f"(+${est_pnl:,.0f}) instead of waiting for full decay."
        )


# ---------------------------------------------------------------------------
# Market-open check
# ---------------------------------------------------------------------------

def market_open_on(day) -> "bool | None":
    """
    Deterministic NYSE calendar check via pandas_market_calendars.
    Returns True/False, or None when the library is unavailable
    (caller falls back to the LLM check).
    """
    try:
        import pandas_market_calendars as mcal
    except ImportError:
        return None
    try:
        nyse  = mcal.get_calendar("NYSE")
        sched = nyse.schedule(start_date=day, end_date=day)
        return not sched.empty
    except Exception:
        return None


def seconds_until_market_check(now_et) -> float:
    """
    Seconds until the next 09:00 ET. On a closed day the overseer sleeps
    this long before exiting — otherwise the container's restart policy
    hot-loops the process every few seconds, re-sending the "market
    closed" Slack notification on every cycle (observed 2026-07-04).
    """
    target = now_et.replace(hour=9, minute=0, second=0, microsecond=0)
    if now_et >= target:
        target += timedelta(days=1)
    return (target - now_et).total_seconds()


def _sleep_until_next_check(now_et) -> None:
    sleep_s = seconds_until_market_check(now_et)
    print(f"  [MarketCheck] sleeping {sleep_s / 3600:.1f}h until the next "
          f"09:00 ET calendar check")
    time.sleep(sleep_s)


MARKET_CHECK_SYSTEM = """You are a US stock market calendar expert.
Answer ONLY with valid JSON: {"open": true, "reason": "brief reason"}
or {"open": false, "reason": "brief reason"}
No other text."""


def check_market_open(llm, send_slack) -> bool:
    """
    Deterministic NYSE calendar first, LLM fallback when the calendar
    library is unavailable. Sends Slack + sleeps until the next 09:00 ET
    check when closed. Returns True when open (or when all checks fail —
    fail open so the scanner decides).
    """
    now_et  = _now_et()
    today   = now_et.date()
    weekday = now_et.strftime("%A")

    cal_open = market_open_on(today)
    if cal_open is not None:
        status = "OPEN" if cal_open else "CLOSED"
        print(f"  [MarketCheck] {today} ({weekday}): market {status} (NYSE calendar)")
        if not cal_open:
            send_slack(f"*Market closed today* ({today}, {weekday}): NYSE calendar."
                       f"\nOverseer will not scan.")
            _sleep_until_next_check(now_et)
        return cal_open

    # Calendar library unavailable — ask the LLM
    try:
        raw = llm.chat(system=MARKET_CHECK_SYSTEM,
                       user=f"Is the US stock market open today, {today} ({weekday})?")
        parsed  = json.loads(raw)
        is_open = bool(parsed.get("open", True))
        reason  = parsed.get("reason", "")
        print(f"  [MarketCheck] {today} ({weekday}): "
              f"{'OPEN' if is_open else 'CLOSED'} (LLM: {reason})")
        if not is_open:
            send_slack(f"*Market closed today* ({today}, {weekday}): {reason}\n"
                       f"Overseer will not scan.")
            _sleep_until_next_check(now_et)
        return is_open
    except Exception as exc:
        print(f"  [MarketCheck] ⚠ LLM check failed ({exc}) — assuming open")
        return True


# ---------------------------------------------------------------------------
# Vault 8 weekly range scan
# ---------------------------------------------------------------------------

def _load_vault8_range(symbol: str) -> dict | None:
    """Return this week's Vault 8 range prediction for symbol, or None."""
    try:
        with open(_VAULT8_SIGNALS_PATH) as f:
            data = json.load(f)
    except Exception:
        return None
    iso_week = date.today().isocalendar()
    week_key = f"{iso_week.year}-W{iso_week.week:02d}"
    for sig in data.get(week_key, []):
        if sig.get("symbol", "").upper() == symbol.upper():
            return sig
    return None


def _refresh_bluechip_history_if_stale(send_slack) -> None:
    """
    Refresh data/*_history.csv (the price history the Vault8 Responder
    scans) if they're missing recent trading sessions. The Responder
    otherwise silently reuses week-old CSVs and produces identical
    signals week over week (observed 2026-07-27: W30 and W31 signals
    were byte-for-byte identical because the CSVs hadn't been refreshed
    since 2026-07-21).
    """
    import pandas as pd

    ref_path = os.path.join(_DATA_DIR, "spy_history.csv")
    last_date = None
    try:
        last_date = pd.read_csv(
            ref_path, usecols=["datetime"], parse_dates=["datetime"]
        )["datetime"].max().date()
    except Exception:
        pass

    today    = date.today()
    expected = today - timedelta(days=1)
    try:
        import pandas_market_calendars as mcal
        nyse  = mcal.get_calendar("NYSE")
        sched = nyse.schedule(start_date=today - timedelta(days=10), end_date=today)
        prior_days = [d.date() for d in sched.index if d.date() < today]
        if prior_days:
            expected = prior_days[-1]
    except Exception:
        pass

    if last_date is not None and last_date >= expected:
        return  # fresh enough — last row covers the most recent completed session

    print(f"  [Vault8] Blue-chip history stale (last={last_date}, expected>={expected})"
          f" — refreshing via download_bluechips.py...")
    try:
        from vault8.download_bluechips import main as _download_bluechips_main
        _download_bluechips_main()
        print("  [Vault8] Blue-chip history refreshed.")
    except Exception as e:
        print(f"  [Vault8] History refresh failed ({e}) — proceeding with existing CSVs.")
        send_slack(f"*Vault 8 Weekly Scan* — history refresh failed ({e}); using stale CSVs.")


def _maybe_run_vault8_weekly(send_slack) -> None:
    """
    If vault8 weekly signals don't exist for the current ISO week,
    run the Responder across all blue chips and post results to Slack.
    Called at every overseer startup; idempotent — skips if week already scanned.
    """
    import pandas as pd

    now_et   = _now_et()
    today    = now_et.date()
    iso      = today.isocalendar()
    week_key = f"{iso.year}-W{iso.week:02d}"

    try:
        with open(_VAULT8_SIGNALS_PATH) as f:
            existing = json.load(f)
        if week_key in existing:
            print(f"  [Vault8] Weekly scan already done for {week_key} — skipping.")
            return
    except Exception:
        existing = {}

    print(f"\n  [Vault8] No prediction for {week_key} — running weekly range scan ({today.strftime('%A')})...")
    _refresh_bluechip_history_if_stale(send_slack)

    # Determine regime from VIX
    regime = "RECLAMATION"
    try:
        import yfinance as yf
        vix_hist = yf.Ticker("^VIX").history(period="2d")
        if not vix_hist.empty:
            vix = float(vix_hist["Close"].iloc[-1])
            if vix >= 30.0:
                regime = "NUKED_ZONE"
            elif vix >= 20.0:
                regime = "WASTELAND"
            print(f"  [Vault8] VIX={vix:.1f} → regime={regime}")
    except Exception as e:
        print(f"  [Vault8] VIX fetch failed ({e}) — defaulting to RECLAMATION")

    if regime == "NUKED_ZONE":
        send_slack(f"*Vault 8 Weekly Scan — {week_key}*\n⚠ VIX ≥ 30, NUKED_ZONE "
                   f"— Responder benched. No weekly range signals.")
        existing[week_key] = []
        with open(_VAULT8_SIGNALS_PATH, "w") as f:
            json.dump(existing, f, indent=2)
        return

    from vault8.armory.responder import Responder
    responder = Responder()

    # Blue chip universe — same as download_bluechips.py (skip BRK-B filename quirk)
    BLUE_CHIPS = [
        "AAPL","MSFT","UNH","GS","HD","MCD","CAT","CRM","V","AMGN",
        "HON","AXP","TRV","JPM","IBM","JNJ","WMT","PG","CVX","MRK",
        "DIS","NKE","MMM","KO","BA","CSCO","VZ","INTC","DOW",
        "NVDA","GOOGL","AMZN","META","TSLA","AMD","AVGO","QCOM","TXN",
        "LLY","ABT","TMO","ABBV","BRKB","BAC","WFC","COST","XOM",
        "SPY","QQQ","GLD",
    ]

    results = []
    for ticker in BLUE_CHIPS:
        csv_path = os.path.join(_DATA_DIR, f"{ticker.lower()}_history.csv")
        if not os.path.exists(csv_path):
            continue
        try:
            df  = pd.read_csv(csv_path, parse_dates=["datetime"])
            sig = responder.scan(ticker, df, regime=regime)
            if sig.get("signal") == "BUY_WEEK_LOW":
                results.append(sig)
                print(f"  [Vault8]   {ticker:<6} entry=${sig['entry']:.2f}"
                      f" target=${sig['target']:.2f}"
                      f" range={sig['pred_range_pct']:.1f}%"
                      f" conf={sig['confidence']}")
        except Exception as e:
            print(f"  [Vault8]   {ticker} error: {e}")

    results.sort(key=lambda x: x.get("confidence", 0), reverse=True)
    existing[week_key] = results
    os.makedirs(_DATA_DIR, exist_ok=True)
    with open(_VAULT8_SIGNALS_PATH, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"  [Vault8] {len(results)} signals saved → {_VAULT8_SIGNALS_PATH}")

    if not results:
        send_slack(f"*Vault 8 Weekly Scan — {week_key}*\nRegime: {regime}\n"
                   f"No tradeable weekly ranges found.")
        return
    top_n = results[:10]
    lines = [f"*Vault 8 — Weekly Range Signals ({week_key})*",
             f"Regime: {regime}  |  {len(results)} tradeable stocks\n"]
    for r in top_n:
        lines.append(
            f"• `{r['symbol']:<6}` entry ${r['entry']:.2f} → sell ${r['target']:.2f}"
            f"  ({r['pred_range_pct']:.1f}% range)  conf={r['confidence']}"
        )
    if len(results) > 10:
        lines.append(f"_(+{len(results) - 10} more — see {_VAULT8_SIGNALS_PATH})_")
    send_slack("\n".join(lines))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Real Overseer — fully autonomous LLM-driven Vault 76 scanner "
                    "(real money only; requires REALLY_REAL=true)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--provider", default=None,
                        help="LLM provider: anthropic | openai | deepseek | openai_compatible")
    parser.add_argument("--model", default=None, help="LLM model name")
    parser.add_argument("--base-url", default=None, dest="base_url",
                        help="Base URL for openai_compatible endpoints")
    return parser


def main():
    args = build_arg_parser().parse_args()

    if os.environ.get("REALLY_REAL", "").lower() != "true":
        print("⛔ REALLY_REAL is not 'true' — this overseer places REAL orders "
              "and refuses to start without the safety gate. Set REALLY_REAL=true "
              "in .env to proceed.")
        sys.exit(1)

    ov = RealOverseer(provider=args.provider, model=args.model,
                      base_url=args.base_url)

    # live_scanner.main() parses sys.argv — leave it bare (real mode)
    sys.argv = [sys.argv[0]]

    import live_scanner as scanner
    scanner.set_decision_fn(ov.decide)
    scanner.set_scan_hook_fn(ov.scan_hook)
    scanner._slack_prefix = f"`({ov.llm.provider})`\n"

    if not check_market_open(ov.llm, scanner._send_slack):
        return

    _maybe_run_vault8_weekly(scanner._send_slack)

    scanner.main()


if __name__ == "__main__":
    main()
