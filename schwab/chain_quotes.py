"""
Real option-chain quotes from the Schwab API for Scavenger signals.

The Scavenger prices premiums with Black-Scholes on historical volatility,
which drifts from reality whenever IV diverges from HV — exactly the moments
premium selling is most interesting. This module re-quotes a SELL_PUT /
SELL_CALL signal against the live chain: nearest listed strike, expiration
closest to the target DTE, mid price, real IV and delta.

If the chain fetch fails (API down, no liquid contracts), the signal keeps
its model premium and is tagged quote_source="model" so downstream consumers
(LLM prompt, ledger reason) know which price they are looking at.
"""
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(__file__))

import math

from strategy_params import SCAV_MIN_PREMIUM_PCT

MIN_DTE       = 21     # earliest expiration considered
MAX_DTE       = 45     # latest expiration considered
MIN_OPEN_INT  = 1      # skip strikes nobody holds — unquotable in practice
# Widest (ask-bid)/mid we will sell into. Orders go in at the BID, so a wide
# spread means handing the market maker the difference. Set from observation on
# 2026-10-05: the good fills that day (AMZN, GOOGL, IBM) were 3-6%; the XOM
# strike that cost -$82.66 was 250%. 25% is deliberately loose — it rejects
# unquotable strikes without second-guessing which signals qualify, a change
# that would deserve a backtest.
MAX_SPREAD_PCT = 0.25
# Order limit = the mid, rounded UP to the next tradable tick.
#
# This replaced limit = 0.95 * mid, which was a real defect: 0.95*mid sits above
# the bid only when ask/bid > 1.05/0.95, i.e. a spread wider than 10.53% of mid.
# Our median spread is 5.6%, so the limit landed BELOW the bid on 240 of 309
# sellable candidates (77.7%) measured 2026-10-06 -- and a sell limit below the
# bid crosses the book and fills at the bid. The patient pricing was therefore
# inoperative on most trades, quietly reverting to the bid-selling it replaced.
# Worst case that day: AMD $610, spread 2.1%, bid 31.35, limit 30.09.
#
# Rounding UP rather than down is load-bearing. On a one-tick market
# (bid 3.55 / ask 3.60) the mid is 3.575; rounding down returns 3.55, the bid
# again. Rounding up can never exceed the ask, because the ask is itself on a
# valid tick and mid < ask. So bid < limit <= ask holds for every quote, which
# is also the property that makes this safe in a disorderly market: an order
# that cannot cross cannot be filled at a price we did not choose.
#
# Increments follow what the market DISPLAYS, verified against 3,837 live quotes
# on 2026-10-06: $0.01 below $3.00, $0.05 at or above it (0 of 2,636 quotes
# at/above $3 were off-nickel).
#
# Be careful about why. Rounding to the displayed increment is a CHOICE, not a
# requirement -- an off-nickel limit above $3 is accepted and fills exactly as
# sent. Checked on the two we have sent: NVDA 260918P00205000 at $4.84 and
# IBM 260918P00220000 at $3.06, both 2026-08-21, both executed at precisely
# those prices, no rounding in either direction. None of the 48 rejections in
# the last 90 days was tick-related either. So this is not protection against
# rejection; an earlier version of this comment claimed it was, and that was
# wrong.
#
# It is kept because matching the displayed increment costs almost nothing and
# keeps the limit a price the book actually shows: measured across 236 live
# candidates with a mid at/above $3, nickel rounding asks a mean $0.0115/sh over
# the mid against $0.0023 for penny rounding -- about $0.90 per contract -- and
# lands at the ask on 1 of 236. Penny rounding would satisfy the same invariant
# if we ever want the extra fill probability.
TICK_BREAK_PRICE = 3.00
TICK_BELOW_BREAK = 0.01
TICK_AT_OR_ABOVE = 0.05

# How far toward the money a re-quote may move the strike, as a fraction of the
# OTM distance the strategy asked for. 0.70 of a 5%-OTM target = a 3.5% floor.
#
# Why a bound is needed at all: the liquidity filters above reject the thinner
# strikes, and the ranking below takes the nearest ELIGIBLE strike with no limit
# on the distance. On 2026-10-08 every XOM strike near the $160.17 target was
# rejected (spread 31-84%, or bid size 5) and the picker substituted $165 at a
# different expiry -- 2.1% OTM, delta -0.42, on a stock at $168.60. That is a
# different trade from the one the Scavenger decided to make: roughly double the
# assignment probability on a book whose mandate is steady income.
#
# 0.70 comes from measurement, not taste. Across 14,331 SELL_PUT signals in
# data/overseer.log the realized OTM% sits at p0.1=3.88, p1=4.00, p50=4.97,
# p100=6.51. The natural floor is ~3.86% -- NVDA, where the nearest listed
# strike on a high-priced stock simply lands there. Exactly 2 signals ever fell
# below 3.5%: the two pathological XOM re-quotes (2.14% and 2.92%), neither of
# which became a position (one budget-blocked, one rejected with a 429). So this
# rejects the defect and nothing else we have on record.
#
# Deliberately NOT a delta cap, which was the obvious first idea: delta is a bad
# discriminator here. A correct 5%-OTM strike on a high-IV name legitimately
# reaches |delta| 0.37 (INTC at 4.6% OTM), so any cap tight enough to catch the
# XOM case at 0.42 also rejects 278 sound signals (1.9%). Distance from the
# intended strike is what actually went wrong, so that is what is bounded.
MIN_OTM_FRACTION = 0.70

# A limit above where the contract actually traded today is a price nobody will
# pay. GOOGL 11-06 $335 went in at 7.55 against a session high of 7.30 and a
# last print of 7.16, and rested unfillable for 25 minutes. Nothing looked.
#
# Scope: the size-weighted limit already prevents THAT case (it prices 7.20).
# This is the backstop for the other direction -- a heavily BID book walking the
# limit up past where the contract trades -- and for any future pricing change
# that puts the limit outside reality.
#
# Both numbers measured against 1,104 live eligible contracts (126 in the 3-7%
# OTM band we actually trade) on 2026-10-08:
#
#   MIN_RANGE_VOLUME -- the session range is only a ceiling if contracts stand
#   behind it. Every overshoot above 2% in the tradeable band had volume <= 6
#   (PG +9.9% on volume 1, IBM +7.2% on volume 1): a single stale print. No
#   contract with volume >= 10 overshot by more than 1.82%.
#
#   MAX_LIMIT_OVER_SESSION -- overshoot percentiles in that band: p50 -4.31%,
#   p90 +0.68%, p95 +1.82%, p98 +3.57%. Sub-1% is ordinary drift since the last
#   print and must pass; GOOGL's failure was +3.42%. 2% sits in the empty gap
#   between the two populations, giving 0 false rejections across all 126.
#
# Known gap, left open deliberately: with no print at all there is no range to
# validate against, so this abstains rather than blocks. Correct inside RTH (32%
# of eligible contracts have no print yet, and MAX_UNTRADED_AGE_MIN covers the
# stale ones); NOT correct for a pre-market or overnight options session, where
# nothing trades. Making "no range" a refusal would reject ~40% of current
# signals, so it needs a decision and a backtest, not a quiet default.
#
# Known false positive: if the underlying moves hard, the puts reprice on the
# QUOTE before anything prints at the new level, so the session high reflects
# the old regime and an honest price gets skipped. Accepted on the same grounds
# the rest of this module uses -- a missed entry is cheaper than a badly-priced
# position, and the signal returns next scan once a print lands. But it costs
# entries exactly when premium is richest, so revisit this first if the filter
# proves expensive. Pinned by test_a_quote_that_repriced_before_printing_is_rejected.
MIN_RANGE_VOLUME        = 10
MAX_LIMIT_OVER_SESSION  = 0.02


def _tick(price: float) -> float:
    """Minimum price increment for an option trading at `price`."""
    return TICK_AT_OR_ABOVE if price >= TICK_BREAK_PRICE else TICK_BELOW_BREAK


def round_up_to_tick(price: float) -> float:
    """Round a price up to the next tradable tick.

    For model-priced signals, where there is no chain quote to take a mid from.
    Rounding up, not down, for the same reason as order_limit_for: we never ask
    for less than our own estimate of fair value.
    """
    tick = _tick(price)
    return round(math.ceil(round(price / tick, 6)) * tick, 2)


def order_limit_for(bid: float, ask: float,
                    bid_size: int | None = None,
                    ask_size: int | None = None) -> float:
    """The price we ask for: the SIZE-WEIGHTED fair value, on a tradable tick.

    Guarantees bid < limit <= ask for any bid < ask, so the order always rests
    instead of crossing. A locked market (bid == ask) returns that price.

    Why weighted and not the plain mid. The mid is only fair when the book is
    balanced, and all three mid-priced orders we placed proved it:

      XOM  11-06 155   bid 1.87(10)  ask 2.14(11)   balanced
          mid 2.005 -> filled instantly at 2.00. Right.
      AMZN 11-06 245   bid 5.65(427) ask 5.95(13)   heavily BID
          mid 5.80 -> filled instantly, but fair value was 5.94: ~14c given away.
      GOOGL 11-06 335  bid 7.15(17)  ask 7.90(509)  heavily OFFERED
          mid 7.525 -> limit 7.55, ABOVE the day's high of 7.30 and above the
          later ask of 7.35. Unfillable, and it blocked GOOGL for the session.

    A large bid size means buyers are stacked and the price is likelier to tick
    up, so fair value sits nearer the ask; a stacked offer means the reverse.
    That is the standard microprice, and the sizes are already fetched -- they
    were only being printed on the LIQUIDITY line.

    Rounding direction flips with it. The old rule rounded UP so the limit could
    never land on the bid; with a weighted value that can sit just under the ask
    that lands exactly ON the ask (AMZN: 5.9411 -> 5.95), the least fillable
    price in the spread. So round DOWN and floor at one tick above the bid,
    which states the never-cross invariant instead of relying on the rounding
    direction to imply it.

    Spreads alone will not catch this: GOOGL's was only 10%, well inside
    MAX_SPREAD_PCT. Both failures happened in the first 40 minutes after the
    open, when books are widest and most lopsided.
    """
    mid = (bid + ask) / 2
    total = (bid_size or 0) + (ask_size or 0)
    if bid_size is None or ask_size is None or total <= 0:
        fair = mid                      # no size data — the mid is all we have
    else:
        fair = (bid * ask_size + ask * bid_size) / total

    tick = _tick(fair)
    # Work in integer ticks to dodge binary-float surprises: 3.675/0.05 is
    # 73.49999... in floating point, which floor()s to the wrong step.
    limit = math.floor(round(fair / tick, 6)) * tick
    # Never cross: the limit must sit strictly above the bid. On a one-tick
    # market this forces the ask, which is the only non-crossing price there.
    limit = max(limit, bid + _tick(bid))
    return round(min(limit, ask), 2)
# Minimum contracts resting on the bid. This, not open interest, is the real
# liquidity gate: OI counts contracts somebody HOLDS, bidSize counts contracts
# somebody will BUY today. Measured 2026-10-06, KO quoted bid 0.00 / bidSize 0
# / volume 0 with open interest clearing MIN_OPEN_INT=1 -- unsellable at any
# price, and only the `bid <= 0` check kept it out. 10 is ~10x the 1-contract
# size traded, and well under the 43-635 seen on healthy strikes.
MIN_BID_SIZE = 10
# Age of the last print, in minutes, past which a contract with ZERO volume
# today is treated as an untested market rather than a merely thin one. Both
# conditions are required: zero volume alone is common and perfectly fillable
# (UNH trades a median of 6 contracts), and a stale print on something that DID
# trade today is just quiet.
#
# Measured live on Monday 2026-10-06 against 314 contracts clearing every other
# gate inside the delta band we sell: this rejects 5 of them (1.6%), all with
# open interest of 1-42 and no trade since Friday.
#
# Know what this threshold does NOT do: 1440 minutes is 24 hours, while Friday
# 16:00 ET to Monday 09:30 ET is 3930 minutes. A weekend gap therefore clears it
# on its own, so for the whole Monday session the rule collapses to "zero volume
# today" for anything untraded since Friday. That is stricter on Mondays than
# Tuesdays, which is ugly but lands on the safe side: a contract nobody has
# touched since Friday close genuinely has nothing behind its mid, and skipping
# it costs one day. Fixing the asymmetry properly means counting trading
# SESSIONS rather than wall-clock minutes, which needs the market calendar --
# not worth it for 1.6% of candidates.
#
# Do not read a stale `last` as proof of a bad quote on its own. Across a
# weekend the underlying moves, so Friday's print SHOULD diverge from Monday's
# mid; META $705 showed last 27.56 against mid 22.02 purely from the gap. The
# signal here is the absence of trading, not the size of that divergence.
MAX_UNTRADED_AGE_MIN = 1440


def _trade_age_min(opt: dict) -> float | None:
    """Minutes between the last print and the current quote, or None.

    A large value means the price you see is a quote nobody has traded against
    recently — the condition that made XOM's mid meaningless. Reported rather
    than filtered on: a stale print alongside a live two-sided quote is still
    perfectly tradeable.
    """
    q_t, t_t = opt.get("quoteTimeInLong"), opt.get("tradeTimeInLong")
    if not q_t or not t_t:
        return None
    return round(max(0.0, (float(q_t) - float(t_t)) / 1000 / 60), 1)


def _tally(reasons: dict | None, key: str) -> None:
    """Count one per-strike rejection.

    Tallied rather than printed: this loop walks every strike in the DTE window
    for 14 symbols every 5 minutes (1,104 eligible contracts in a live sample),
    so a line per rejection would bury a log that is already 20MB. requote_signal
    prints the tally once, and only when the signal actually dies.
    """
    if reasons is not None:
        reasons[key] = reasons.get(key, 0) + 1


def fetch_chain_quote(client, symbol: str, option_type: str,
                      target_strike: float, target_dte: int,
                      reasons: dict | None = None,
                      min_dte: int = MIN_DTE, max_dte: int = MAX_DTE) -> dict | None:
    """
    Fetch the chain for symbol and return the contract nearest to
    (target_strike, target_dte):

      {"strike", "premium" (the MID), "order_limit", "mid", "bid", "ask",
       "dte", "expiry", "iv" (fraction), "delta", "open_interest"}

    `premium` is the MID — a fair estimate of the fill, since a sell limit
    fills at the limit or better. `order_limit` is where the order is actually
    placed: the mid rounded up to a tradable tick, which always rests
    above the bid rather than crossing
    wider than 10% of mid. Both are shown to the LLM so it can judge the trade-off.

    Contracts wider than MAX_SPREAD_PCT are skipped entirely — that guard is
    what makes the mid trustworthy. The mid of a 250% stale quote is the number
    that cost -$82.66 on XOM on 2026-10-05.

    option_type: "PUT" or "CALL". Returns None when nothing usable is found.
    """
    try:
        kwargs = {}
        try:
            # Narrow the server-side window when the enum is importable
            from schwab.client import Client as _C
            kwargs["contract_type"] = (_C.Options.ContractType.CALL
                                       if option_type == "CALL"
                                       else _C.Options.ContractType.PUT)
        except Exception:
            pass
        resp = client.get_option_chain(
            symbol,
            include_underlying_quote=True,
            from_date=datetime.now() + timedelta(days=min_dte),
            to_date=datetime.now() + timedelta(days=max_dte),
            **kwargs,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return None

    exp_map = data.get("callExpDateMap" if option_type == "CALL"
                       else "putExpDateMap", {})
    if not exp_map:
        return None

    best = None
    for exp_key, strikes in exp_map.items():
        try:
            exp_str, dte = exp_key.split(":")[0], int(exp_key.split(":")[1])
        except (IndexError, ValueError):
            continue
        if not (min_dte <= dte <= max_dte):
            continue
        for strike_str, opts in strikes.items():
            opt    = opts[0]
            strike = float(strike_str)
            bid    = float(opt.get("bid", 0) or 0)
            ask    = float(opt.get("ask", 0) or 0)
            oi     = int(opt.get("openInterest", 0) or 0)
            if bid <= 0:
                _tally(reasons, "no_bid")
                continue
            if oi < MIN_OPEN_INT:
                _tally(reasons, "open_interest")
                continue
            # A missing size must not read as zero — some quotes omit it, and
            # treating absent data as "no liquidity" drops every signal.
            bid_size = opt.get("bidSize")
            if bid_size is not None and int(bid_size) < MIN_BID_SIZE:
                _tally(reasons, "bid_size")
                continue
            # A market nobody has tested. Absent fields must not reject --
            # treating a data gap as the worst case would silently drop
            # signals, the same reasoning as bidSize above.
            volume = opt.get("totalVolume")
            age    = _trade_age_min(opt)
            if (volume is not None and int(volume) == 0
                    and age is not None and age > MAX_UNTRADED_AGE_MIN):
                _tally(reasons, "untraded")
                continue   # no volume today and no print for a full session:
                           # the mid has nothing behind it
            mid = (bid + ask) / 2
            if mid <= 0 or (ask - bid) / mid > MAX_SPREAD_PCT:
                _tally(reasons, "spread")
                continue   # unquotable — selling at the bid here gives away
                           # most of the premium
            limit = order_limit_for(
                bid, ask,
                bid_size=int(bid_size) if bid_size is not None else None,
                ask_size=(int(opt["askSize"])
                          if opt.get("askSize") is not None else None))
            # Would we be asking a price this contract has not traded at today?
            # Only meaningful once enough contracts stand behind the range; with
            # no print at all there is nothing to check and this abstains.
            session_hi = max(float(opt.get("highPrice", 0) or 0),
                             float(opt.get("last", 0) or 0))
            if (session_hi > 0
                    and volume is not None and int(volume) >= MIN_RANGE_VOLUME
                    and limit > session_hi * (1 + MAX_LIMIT_OVER_SESSION)):
                _tally(reasons, "limit_over_session")
                continue   # a limit above a well-traded session high will rest
                           # unfilled all day — GOOGL 11-06 $335 at 7.55
            # Rank by distance to target strike first, then to target DTE
            rank = (abs(strike - target_strike), abs(dte - target_dte))
            if best is None or rank < best[0]:
                best = (rank, {
                    "strike":        strike,
                    # Expected fill. A sell limit fills at the limit or better,
                    # so the mid is the fair estimate; the bid is only a floor.
                    "premium":       round(mid, 4),
                    "mid":           round(mid, 4),
                    # Where the order actually goes in.
                    "order_limit":   limit,
                    "bid":           bid,
                    # Liquidity context for the LLM. `last` is deliberately NOT
                    # a pricing input: lastSize is 1-4 contracts in practice and
                    # trade_age_min has been measured anywhere from 0.2 to 5863
                    # minutes. It is here so the model can judge whether a quote
                    # is real, which is judgement rather than arithmetic.
                    "last":          float(opt.get("last", 0) or 0),
                    "last_size":     int(opt.get("lastSize", 0) or 0),
                    "volume":        int(volume) if volume is not None else 0,
                    "bid_size":      int(bid_size) if bid_size is not None else None,
                    "ask_size":      (int(opt["askSize"])
                                      if opt.get("askSize") is not None else None),
                    "trade_age_min": age,
                    "ask":           ask,
                    "dte":           dte,
                    "expiry":        exp_str,
                    "iv":            float(opt.get("volatility", 0) or 0) / 100,
                    "delta":         float(opt.get("delta", 0) or 0),
                    "open_interest": oi,
                })
    return best[1] if best else None


def requote_signal(client, s: dict, target_dte: int | None = None) -> dict | None:
    """
    Replace a Scavenger signal's model premium with a real chain quote.

    Returns the signal (mutated in place) tagged quote_source="schwab_chain"
    or "model" on fallback. Returns None when the *real* premium falls below
    the SCAV_MIN_PREMIUM_PCT floor — the trade the model priced does not
    actually exist at that yield, so the signal is dropped.

    The floor is judged on the mid, which is a fair fill estimate now that
    MAX_SPREAD_PCT rejects unquotable contracts. Before that guard existed the
    mid of a 250% spread passed a 0.5% floor as "1.10%" and filled at 0.52%.
    Non-option signals pass through untouched.
    """
    if s.get("signal") not in ("SELL_PUT", "SELL_CALL"):
        return s

    option_type = "CALL" if s["signal"] == "SELL_CALL" else "PUT"
    target_dte  = target_dte or int(s.get("dte", 30) or 30)
    model_close  = float(s.get("close", 0) or 0)
    model_strike = float(s.get("strike", 0) or 0)
    # Set the label from the MODEL strike up front so the key always exists,
    # including on the fallback path below. live_scanner prints
    # s.get("otm_pct", "5") / ("otm_pct", "8"), and those hardcoded defaults are
    # how a 2.1% OTM XOM put displayed as "(5% OTM)" on 2026-10-08.
    if model_close:
        s["otm_pct"] = round((abs(model_close - model_strike) / model_close)
                             * 100, 1)
    rejects: dict[str, int] = {}
    q = fetch_chain_quote(client, s["symbol"], option_type,
                          target_strike=model_strike,
                          target_dte=target_dte, reasons=rejects)

    if q is None or q["premium"] <= 0:
        # No eligible strike anywhere in the DTE window. Name the filters that
        # did it -- otherwise this symbol just silently stops producing signals
        # and nobody can tell a dead market from an over-tight threshold.
        if rejects:
            tally = "  ".join(f"{k} {v}" for k, v in
                              sorted(rejects.items(), key=lambda kv: -kv[1]))
            print(f"  [Quote] ⊘ {s['symbol']} {s['signal']} — no eligible "
                  f"strike near ${model_strike:.2f}: {tally}")
        s["quote_source"] = "model"
        return s

    close = float(s.get("close", 0) or 0)
    premium_pct = q["premium"] / close * 100 if close else 0.0
    if close and q["premium"] / close < SCAV_MIN_PREMIUM_PCT:
        print(f"  [Quote] ⊘ {s['symbol']} {s['signal']} ${q['strike']} dropped — "
              f"real premium {premium_pct:.2f}% of close, floor "
              f"{SCAV_MIN_PREMIUM_PCT * 100:.2f}%")
        return None   # real premium too thin — the modeled trade doesn't exist

    # How far OTM the chain's strike actually sits, versus how far the strategy
    # asked for. Measured from the ORIGINAL s["strike"] (still the model's
    # target — s.update below is what overwrites it), so this works for puts and
    # for both covered-call widths without importing any strategy parameter.
    is_put       = s.get("signal") == "SELL_PUT"
    realized_otm = ((close - q["strike"]) / close if is_put
                    else (q["strike"] - close) / close) if close else 0.0
    target_otm   = (abs(close - float(s.get("strike", 0) or 0)) / close
                    if close else 0.0)
    if target_otm > 0 and realized_otm < target_otm * MIN_OTM_FRACTION:
        print(f"  [Quote] ⊘ {s['symbol']} {s['signal']} dropped — strike drift: "
              f"chain gave ${q['strike']} ({realized_otm * 100:.1f}% OTM) vs "
              f"{target_otm * 100:.1f}% target, floor "
              f"{target_otm * MIN_OTM_FRACTION * 100:.1f}%")
        return None   # drifted too far toward the money — skip, don't substitute

    s.update({
        "otm_pct":      round(realized_otm * 100, 1),
        "strike":       q["strike"],
        "premium":      q["premium"],
        "premium_pct":  round(premium_pct, 2),
        "dte":          q["dte"],
        "expiry":       q["expiry"],
        "iv":           round(q["iv"] * 100, 1),      # store as % like hv
        "delta":        q["delta"],
        "order_limit":  q["order_limit"],
        "bid":          q["bid"],
        "ask":          q["ask"],
        "last":          q["last"],
        "last_size":     q["last_size"],
        "volume":        q["volume"],
        "bid_size":      q["bid_size"],
        "ask_size":      q["ask_size"],
        "trade_age_min": q["trade_age_min"],
        "quote_source": "schwab_chain",
    })
    if s.get("signal") == "SELL_PUT":
        s["max_loss"] = round((q["strike"] - q["premium"]) * 100, 2)
    return s
