# Vault 76 Trading System — Fallout 76 Glossary

All system names are drawn from Fallout 76, a post-apocalyptic survival game set in
West Virginia 25 years after the nuclear war. Players emerge from Vault 76 on
Reclamation Day to rebuild Appalachia.

---

## System Architecture

| Trading Concept | Fallout 76 Term | Meaning in the Game |
|---|---|---|
| Overall trading system | **Vault 76** | Vault-Tec's control vault, #76 — designed to release its dwellers on July 4, 2076 to rebuild society |
| Market regime classifier | **The Overseer** | The person who runs the Vault; decides when dwellers deploy and which directives they follow |
| Collection of strategies | **The Armory** | The weapon workshop inside the Vault — where you craft and store your tools of survival |
| Individual trading strategy | **Role** | Cards you equip to your character granting special abilities; each builds on a different playstyle |
| The trader | **Dweller** | A person who lived in a Vault and emerges to face the Wasteland |

---

## Market Regimes (Overseer's Assessment)

| Market Condition | Fallout 76 Term | Trigger |
|---|---|---|
| Bull market | **Reclamation Day** | SPY above rising EMA50, VIX < 30. The day Vault 76 opened — new beginning, opportunity everywhere. Rebuilding is possible. |
| Bear / sideways market | **The Wasteland** | SPY below EMA50 or EMA50 falling, VIX < 30. The hostile surface world. Most creatures want you dead. Survival requires patience and cunning. |
| Crash / extreme volatility | **Nuked Zone** | VIX ≥ 30. A player-launched nuclear missile has landed. Radiation is extreme, enemies are legendary-tier, and the risk of catastrophic loss is real. All cards benched. |

---

## Roles

### The Raider — Role #001
**Fallout 76:** Raiders are aggressive wastelanders who attack when targets are vulnerable.
They raid settlements, strike fast, and retreat when outgunned.

**Trading:** Pullback-in-trend strategy. Enters when a strongly-trending stock dips
(shows vulnerability), then rides the bounce back up. Fast, aggressive, needs a clear target.

- Deploy in: Reclamation Day (primary), The Wasteland (opportunistic)
- Avoid: Nuked Zone — even Raiders know when to run

### The Maggie — Role #003
**Named for:** [Qullamaggie](https://qullamaggie.com/), the swing trader whose
Breakout setup this role implements.

**Trading:** Momentum breakout. Screens for stocks that already ran up 25%+
in the last ~90 bars, then coiled into a tight consolidation (higher lows,
contracting daily range, surfing the rising EMA20). Buys the range-expansion
breakout above the consolidation high on a volume surge. Stop is capped at
the tighter of ATR or ADR% of entry — never a wider stop than the stock's
own noise. First target takes profit and moves the stop to breakeven; the
rest trails on a 10-day EMA close-below.

- Deploy in: Reclamation Day only — Qullamaggie's setups "work best in
  bullish markets"; sit out corrections and bear markets
- Avoid: The Wasteland, Nuked Zone

### The Scavenger — Role #002
**Fallout 76:** Scavengers are patient survivors who find value in overlooked places.
They forage through ruined buildings, extract useful components, and never rush.

**Trading:** Wheel strategy. Sells options on sideways stocks to generate steady income.
Extracts premium from quiet markets others ignore.

- Phase 1 (SELL_PUT): Sell a cash-secured put 5% OTM when a stock is going nowhere
- Phase 2 (SELL_CALL): After assignment, sell a covered call 8% OTM to keep earning
- Deploy in: The Wasteland (primary for income), Reclamation Day (on sideways individual stocks)
- Avoid: Nuked Zone — IV looks attractive but assignment risk is catastrophic

---

## Economy & Resources

| Trading Concept | Fallout 76 Term | Meaning in the Game |
|---|---|---|
| Profits / returns | **Caps** | Bottle caps are the post-apocalyptic currency. Every Fallout fan knows you grind for caps. |
| Cash / buying power | **Stash** | Your personal storage box — what you have in reserve to deploy |
| Portfolio positions | **CAMP** | Your personal base; what you've built and what you're defending |
| P&L report | **Pip-Boy** | Wrist-mounted computer that shows all your stats, inventory, and map |
| Trading signals | **Threat Marked** | Enemies highlighted in VATS — you've identified a target worth engaging |

---

## System Tools

| Tool | Fallout 76 Term | Meaning in the Game |
|---|---|---|
| Regime targeting system (VIX + SPY) | **VATS** | Vault-Tec Assisted Targeting System — slows time, shows hit probability, picks the right target |
| Options pricing (Black-Scholes) | **Workbench** | Crafting station where you analyze components and build weapons |
| Backtesting environment | **The Whitespring** | A massive pre-war resort used as a simulation / safe zone for planning |
| Paper trading | **Simulation Mode** | Training against Scorched enemies before facing real threats |

---

## Threat Levels

| Situation | Fallout 76 Analogy |
|---|---|
| Normal scan, no signal | Exploring the map — finding junk, staying alert |
| Signal detected | Enemy spotted — VATS active, ready to engage |
| Signal approved | Pulling the trigger |
| Signal skipped | Retreating — lived to fight another day |
| Stop-loss hit | Got downed by a Deathclaw — used a Stimpak, respawned |
| Target hit | Legendary loot dropped |
| VIX ≥ 30 | Scorchbeast Queen spawned — all Dwellers retreat to Vault |

---

## Future Roles (Planned) 

| Codename | Strategy | Optimal Regime |
|---|---|---|
| **The Settler** | Dividend + value investing — builds lasting positions | Any regime |
