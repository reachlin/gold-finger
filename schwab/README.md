# Schwab Trading Automation

Pullback-in-trend strategy for US stocks via the Charles Schwab API,
with Claude Code as the live signal verification layer.

---

## Strategy

**Long (BUY) signal conditions — all must be true:**
1. EMA20 > EMA50, price > EMA50, EMA50 rising (confirmed uptrend)
2. RSI dipped into 28–50 zone within last 5 bars (pullback occurred)
3. RSI recovering + green candle + price above EMA20 + volume > 0.9× avg (entry signal)

**Exit rules:**
- Target: entry + 5× ATR (~20%)
- Stop: entry − 2× ATR (~8%)
- Also exit if EMA20 crosses below EMA50 (trend ended)
- Max hold: 30 days

---

## Files

| File | Purpose |
|------|---------|
| `trend_scanner.py` | Core signal logic — `detect_trend`, `detect_pullback`, `detect_entry` |
| `backtest_strategy.py` | Walk-forward backtest on historical NVDA data |
| `live_scanner.py` | **Live loop** — runs in tmux, pauses on signals for Claude verification |
| `signal_verifier.py` | Data utilities — VIX, earnings proximity, headlines (used by Claude during live review) |
| `daily_signal.py` | One-shot daily scan + Slack notification |
| `market_intel.py` | News sentiment + technical analysis helpers |
| `nvda_trader.py` | Schwab API wrappers — auth, price history, order placement |
| `schwab_account.py` | Account info queries |

**Test files:** `test_trend_scanner.py`, `test_backtest_strategy.py`, `test_signal_verifier.py`

---

## Live Trading Workflow (via `/schwab` skill)

### One-command start

In Claude Code CLI:
```
/schwab start    # launches live_scanner.py in schwab-screen tmux
/schwab watch    # Claude enters monitoring loop, auto-approves or skips signals
/schwab stop     # kills the session
```

The scanner exits automatically at 4pm ET (market close). No need to stop it manually.

### Manual start (without skill)

```bash
tmux new-session -d -s schwab-screen
tmux send-keys -t schwab-screen \
  '/Users/lincai/anaconda3/envs/gold-finger/bin/python schwab/live_scanner.py' Enter
tmux attach -t schwab-screen
```

### Claude Code monitoring

When watching (`/schwab watch`), Claude:

Claude polls the pane every ~4.5 min (270s — keeps prompt cache warm). When it sees `===SIGNAL_START===`, it:
1. Reads signal details (symbol, entry, RSI, ADX, etc.)
2. Calls `signal_verifier.gather_signal_context()` for VIX + earnings + headlines
3. Uses its installed trading skills + web search to analyze macro conditions
4. Sends `y` (approve) or `n` (skip) via `tmux send-keys -t schwab-screen`

### 3. What Claude checks before approving

- **VIX ≥ 30** → auto-reject (extreme fear)
- **Earnings within 5 days** → auto-reject (high event risk)
- **Catastrophic news** → reject (factory fire, sanctions, crisis)
- **Elevated VIX 20–30** → approve with caution note
- **Routine news / analyst commentary** → approve

### 4. How to watch manually (if Claude isn't watching)

```bash
tmux attach -t schwab-screen
# When you see "Proceed? [y/N]:", type y or n
```

---

## One-Shot Daily Scan (no tmux needed)

```bash
/Users/lincai/anaconda3/envs/gold-finger/bin/python schwab/daily_signal.py
```

Scans watchlist, runs macro verification gate for each BUY signal, sends Slack notification.

---

## Backtest

```bash
# Uses saved data/nvda_history.csv
/Users/lincai/anaconda3/envs/gold-finger/bin/python schwab/backtest_strategy.py

# Re-fetch from Schwab first
/Users/lincai/anaconda3/envs/gold-finger/bin/python schwab/backtest_strategy.py --fetch
```

---

## Signal Verifier CLI

Quick data check for any symbol:

```bash
/Users/lincai/anaconda3/envs/gold-finger/bin/python schwab/signal_verifier.py NVDA
```

Shows: VIX level, earnings proximity, recent headlines, any auto-block conditions.

---

## Environment Setup

**Conda env:** `gold-finger` at `/Users/lincai/anaconda3/envs/gold-finger/`

**Secrets in `.env`:**
```
SCHWAB_CLIENT_ID=...
SCHWAB_CLIENT_SECRET=...
SLACK_WEBHOOK_URL=...
FINNHUB_API_KEY=...      # optional — headlines fallback to RSS if missing
```

**Token file:** `schwab/schwab_token.json` — created on first OAuth login, not in git.

**First-time auth:**
```bash
/Users/lincai/anaconda3/envs/gold-finger/bin/python schwab/nvda_trader.py --auth
```

---

## Token Budget (one trading day)

| Mode | Interval | Cache | Wakeups/day | Approx tokens |
|------|----------|-------|-------------|---------------|
| Watch (no signals) | 270s (4.5 min) | warm ✓ | 86 | ~100K new tokens |
| Watch (no signals) | 600s (10 min) | cold ✗ | 39 | ~2M+ tokens |
| Per signal analysis | — | — | rare | ~2–5K extra |

**Key insight:** 270s keeps the 5-min prompt cache warm → each check only pays for
the *new* pane content (~1K tokens), not the full conversation context. At 10 min
the cache expires and you re-read everything each time — much more expensive.

Stick with `270s` intervals. On a day with 0 signals the cost is minimal.

---

## Run Tests

```bash
/Users/lincai/anaconda3/envs/gold-finger/bin/python -m pytest schwab/ -v
```

---

## Watchlist

```
NVDA, AMD, TSLA, AAPL, AMZN, META, MSFT, GOOGL, TQQQ, SOXL
```

Edit `WATCHLIST` in `live_scanner.py` or `daily_signal.py` to change.

---

## Signal Flow Diagram

```
live_scanner.py (tmux: schwab-screen)
  │
  ├── every 5 min during market hours (9am–4pm ET)
  │   scan WATCHLIST via Schwab API
  │
  ├── BUY signal found?
  │   ├── print ===SIGNAL_START=== block
  │   └── wait for input: "Proceed? [y/N]:"
  │
Claude Code (watching pane)
  │
  ├── sees ===SIGNAL_START===
  ├── gather_signal_context(symbol) → VIX + earnings + headlines
  ├── analyze with skills (market_intel, web search, news)
  ├── hard-block? → send "n"
  └── all clear? → send "y" + reasoning printed to Claude terminal
  │
live_scanner.py
  ├── "y" → print ===APPROVED=== + order suggestion
  └── "n" → print ===SKIPPED===
```
