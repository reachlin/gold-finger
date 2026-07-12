"""
Fully automatic Vault 76 Overseer — LLM-driven trading decisions.

Replaces the interactive y/n prompt in live_scanner.py with a configurable
LLM call. Supports Anthropic, OpenAI, DeepSeek, and any OpenAI-compatible
endpoint (Ollama, local models, etc.).

Paper mode:  logs decisions to the paper options ledger (safe default)
Real  mode:  places actual Schwab options orders (requires REALLY_REAL=true)

Usage:
    python schwab/auto_overseer.py --paper
    python schwab/auto_overseer.py --real
    python schwab/auto_overseer.py --paper --provider openai --model gpt-4o-mini
    python schwab/auto_overseer.py --paper --provider deepseek
    python schwab/auto_overseer.py --paper --provider openai_compatible --model llama3 --base-url http://localhost:11434/v1

Environment variables (alternative to CLI flags):
    LLM_PROVIDER   anthropic | openai | deepseek | openai_compatible
    LLM_MODEL      model name
    LLM_API_KEY    override API key
    LLM_BASE_URL   base URL for openai_compatible
    REALLY_REAL    set to "true" to enable real order placement
"""
import os
import sys
import json
import time
import argparse

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
               "shares", "entry")


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
                 peer_signals: list[dict] | None = None) -> str:
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
# AutoOverseer
# ---------------------------------------------------------------------------

class AutoOverseer:
    """
    LLM-powered decision engine for the live scanner.

    Wire it up:
        overseer = AutoOverseer(...)
        live_scanner.set_decision_fn(overseer.decide)
    """

    def __init__(self, provider: str | None = None, model: str | None = None,
                 api_key: str | None = None, base_url: str | None = None,
                 paper: bool = True):
        from llm_client import LLMClient
        self.llm    = LLMClient(provider=provider, model=model,
                                api_key=api_key, base_url=base_url)
        self.paper  = paper
        print(f"  [AutoOverseer] LLM: {self.llm}  paper={paper}")

    def decide(self, s: dict) -> str:
        """
        Decision callback for live_scanner.set_decision_fn().
        Returns "y" (approve) or "n" (skip).
        """
        import live_scanner as scanner

        # Portfolio state — read live from scanner's module-level vars
        portfolio  = getattr(scanner, "_current_portfolio", None)
        kronos_cache = getattr(scanner, "_current_kronos_cache", {})

        cash      = portfolio.cash if portfolio else 30_000.0
        committed = scanner._committed_collateral()
        required  = scanner._collateral_required(s)
        available = cash - committed

        portfolio_state = {"available": available, "required": required}

        # Kronos advisory
        kronos = {}
        sym = s.get("symbol", "")
        if sym in kronos_cache:
            entry  = kronos_cache[sym]
            strike = float(s.get("strike", 0))
            warn   = strike > entry.get("support", 0)
            kronos = {
                "support":    entry.get("support", 0),
                "resistance": entry.get("resistance", 0),
                "buf_pct":    entry.get("buf_pct", 0),
                "warn":       warn,
            }

        # Existing open positions on this symbol — duplicates are allowed,
        # the LLM just sees them for concentration-risk judgment
        try:
            import options_ledger as ol
            open_positions = ol.open_options(
                ol.read_rows(scanner.OPTION_LEDGER_PATH), symbol=sym)
        except Exception:
            open_positions = []

        # Competing signals from the same scan — set by live_scanner's loop
        peers = peer_summaries(getattr(scanner, "_current_scan_signals", []), s)

        prompt   = build_prompt(s, portfolio_state, kronos,
                                open_positions=open_positions,
                                peer_signals=peers)
        raw      = self.llm.chat(system=OVERSEER_SYSTEM, user=prompt)
        decision, reason = parse_llm_response(raw) if raw else ("no", "LLM returned empty response")

        verdict = "y" if decision == "yes" else "n"
        icon    = "✅" if verdict == "y" else "⛔"
        print(f"\n  🤖 AutoOverseer [{self.llm.provider}/{self.llm.model}]: "
              f"{icon} {verdict.upper()}  —  {reason}")

        if verdict == "y" and not self.paper:
            self._place_real_order(s)

        return verdict

    def _place_real_order(self, s: dict):
        """
        Place a real Schwab options SELL_TO_OPEN order.
        Requires REALLY_REAL=true env var as a hard safety gate.
        """
        if s.get("signal") not in ("SELL_PUT", "SELL_CALL"):
            # Router/equity signals are strategy-state changes, not options
            # orders — the scanner prints the suggested share trade instead.
            print(f"  [AutoOverseer] {s.get('signal')} — no automated real "
                  f"order; manage the share trade manually")
            return
        if os.environ.get("REALLY_REAL", "").lower() != "true":
            print("  [AutoOverseer] Real order suppressed — set REALLY_REAL=true to enable")
            return

        import live_scanner as scanner
        client = getattr(scanner, "_current_client", None)
        if client is None:
            print("  [AutoOverseer] ⚠ No Schwab client available — cannot place real order")
            return

        try:
            from datetime import date, timedelta
            exp_date = date.today() + timedelta(days=int(s.get("dte", 30)))
            occ_sym  = build_occ_symbol(s["symbol"], exp_date,
                                        s.get("signal", "SELL_PUT"),
                                        float(s["strike"]))

            # Schwab order spec for SELL_TO_OPEN limit
            from schwab.orders.options import option_sell_to_open_limit
            from schwab.orders.common import Duration, Session
            premium = float(s.get("premium", 0))
            # 5% below the signal's theoretical premium to improve fill odds
            limit   = round(premium * 0.95, 2)

            order = (
                option_sell_to_open_limit(occ_sym, 1, limit)
                .set_duration(Duration.DAY)
                .set_session(Session.NORMAL)
                .build()
            )

            # Get account hash
            accts = client.get_account_numbers().json()
            if not accts:
                print("  [AutoOverseer] ⚠ No accounts found")
                return
            account_hash = accts[0]["hashValue"]

            resp = client.place_order(account_hash, order)
            resp.raise_for_status()
            print(f"  [AutoOverseer] ✅ REAL ORDER PLACED: {occ_sym}  limit ${limit:.2f}")

        except Exception as exc:
            print(f"  [AutoOverseer] ⚠ Real order failed: {exc}")


# ---------------------------------------------------------------------------
# OCC option symbol
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
    from datetime import timedelta
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


def _check_market_open(llm: "LLMClient", send_slack) -> bool:
    """
    Check whether the US stock market is open today — deterministic NYSE
    calendar first, LLM fallback when the calendar library is unavailable.
    Sends a Slack notification and returns False if closed so the overseer exits early.
    Returns True if market is open (or if all checks fail — fail open so scanner decides).
    """
    from datetime import datetime
    from zoneinfo import ZoneInfo
    now_et  = datetime.now(ZoneInfo("America/New_York"))
    today   = now_et.date()
    weekday = now_et.strftime("%A")

    # Deterministic calendar check — cheap and cannot hallucinate
    cal_open = market_open_on(today)
    if cal_open is not None:
        status = "OPEN" if cal_open else "CLOSED"
        print(f"  [MarketCheck] {today} ({weekday}): market {status} (NYSE calendar)")
        if not cal_open:
            send_slack(f"*Market closed today* ({today}, {weekday}): NYSE calendar."
                       f"\nOverseer will not scan.")
            _sleep_until_next_check(now_et)
        return cal_open

    prompt = (f"Today in US Eastern Time is {today.isoformat()} ({weekday}). "
              f"Is the US stock market (NYSE/NASDAQ) open for regular trading today? "
              f"Consider weekends and all US public holidays.")
    raw = llm.chat(system=MARKET_CHECK_SYSTEM, user=prompt)
    if not raw:
        print("  [MarketCheck] LLM unavailable — proceeding anyway")
        return True
    try:
        import json as _json
        parsed  = _json.loads(raw.strip())
        is_open = bool(parsed.get("open", True))
        reason  = parsed.get("reason", "")
    except Exception:
        print(f"  [MarketCheck] Could not parse response: {raw!r} — proceeding anyway")
        return True

    status = "OPEN" if is_open else "CLOSED"
    print(f"  [MarketCheck] {today} ({weekday}): market {status} — {reason}")
    if not is_open:
        send_slack(f"*Market closed today* ({today}, {weekday}): {reason}\nOverseer will not scan.")
        _sleep_until_next_check(now_et)
    return is_open


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Auto Overseer — fully autonomous LLM-driven Vault 76 scanner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--paper", action="store_true", default=True,
                            help="Paper trading mode (default, safe)")
    mode_group.add_argument("--real", action="store_true",
                            help="Real trading mode (requires REALLY_REAL=true env var)")

    parser.add_argument("--provider", default=None,
                        help="LLM provider: anthropic | openai | deepseek | openai_compatible")
    parser.add_argument("--model", default=None,
                        help="LLM model name")
    parser.add_argument("--base-url", default=None, dest="base_url",
                        help="Base URL for openai_compatible endpoints")
    args = parser.parse_args()

    paper = not args.real

    if not paper and os.environ.get("REALLY_REAL", "").lower() != "true":
        print("⚠  Real mode selected but REALLY_REAL env var is not 'true'.")
        print("   Orders will be logged but NOT placed. Set REALLY_REAL=true to enable.")

    # Wire up the auto decision callback BEFORE importing scanner's main
    overseer = AutoOverseer(
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        paper=paper,
    )

    # Patch sys.argv so live_scanner.main() parses the right flags
    sys.argv = [sys.argv[0]]
    if paper:
        sys.argv.append("--paper")

    import live_scanner as scanner
    scanner.set_decision_fn(overseer.decide)
    provider_tag = overseer.llm.provider
    scanner._slack_prefix = f"`({provider_tag})`\n"

    # Market-open check via LLM before starting the scan loop
    if not _check_market_open(overseer.llm, scanner._send_slack):
        return

    scanner.main()


if __name__ == "__main__":
    main()
