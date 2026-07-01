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
import argparse

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
- Fast risk-off and duplicate positions are already filtered — trust the pre-screening

## Soft rules (use your judgment)
- SKIP if HV > 60% (too volatile for premium selling — gamma risk too high)
- SKIP if ADX > 28 (stock is trending, not sideways — wrong strategy)
- SKIP if Kronos buffer < -2% (stock predicted to fall below strike — elevated assignment risk)
- SKIP if yield/DTE < 0.02% per day (not enough premium for the capital tie-up)
- APPROVE if HV 15–50%, ADX < 22, yield > 0.03%/day, quality company

## Response format — ONLY valid JSON, no other text
{"decision": "yes", "reason": "brief reason under 15 words"}
or
{"decision": "no", "reason": "brief reason under 15 words"}"""


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_prompt(signal: dict, portfolio_state: dict, kronos: dict) -> str:
    lines = ["## Signal"]
    for k, v in signal.items():
        lines.append(f"  {k}: {v}")

    lines.append("\n## Portfolio state")
    lines.append(f"  Cash available for collateral: ${portfolio_state.get('available', 0):,.0f}")
    lines.append(f"  Collateral required:           ${portfolio_state.get('required', 0):,.0f}")

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
    """Parse LLM JSON response. Returns ("no", reason) on any failure."""
    try:
        parsed = json.loads(text.strip())
        decision = str(parsed.get("decision", "no")).lower().strip()
        reason   = str(parsed.get("reason", "no reason given")).strip()
        if decision not in ("yes", "no"):
            return "no", f"invalid decision value '{decision}' — defaulting to no"
        return decision, reason
    except (json.JSONDecodeError, Exception) as exc:
        return "no", f"JSON parse failed ({exc}) — defaulting to no"


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

        prompt   = build_prompt(s, portfolio_state, kronos)
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
        if os.environ.get("REALLY_REAL", "").lower() != "true":
            print("  [AutoOverseer] Real order suppressed — set REALLY_REAL=true to enable")
            return

        import live_scanner as scanner
        client = getattr(scanner, "_current_client", None)
        if client is None:
            print("  [AutoOverseer] ⚠ No Schwab client available — cannot place real order")
            return

        try:
            # Build OCC option symbol:  {ROOT:<6}{YYMMDD}{C/P}{strike*1000:08d}
            from datetime import date, timedelta
            sym        = s["symbol"].ljust(6)
            exp_date   = date.today() + timedelta(days=int(s.get("dte", 30)))
            exp_str    = exp_date.strftime("%y%m%d")
            strike_int = int(float(s["strike"]) * 1000)
            occ_sym    = f"{sym}{exp_str}P{strike_int:08d}"

            # Schwab order spec for SELL_TO_OPEN limit
            from schwab.orders.options import option_sell_to_open_limit
            from schwab.orders.common import Duration, Session
            premium = float(s.get("premium", 0))
            limit   = round(premium * 0.95, 2)  # 5% below mid to ensure fill

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
# Market-open check
# ---------------------------------------------------------------------------

MARKET_CHECK_SYSTEM = """You are a US stock market calendar expert.
Answer ONLY with valid JSON: {"open": true, "reason": "brief reason"}
or {"open": false, "reason": "brief reason"}
No other text."""


def _check_market_open(llm: "LLMClient", send_slack) -> bool:
    """
    Ask the LLM whether the US stock market is open today.
    Sends a Slack notification and returns False if closed so the overseer exits early.
    Returns True if market is open (or if check fails — fail open so scanner decides).
    """
    from datetime import date
    today = date.today()
    weekday = today.strftime("%A")
    prompt = (f"Today is {today.isoformat()} ({weekday}). "
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
