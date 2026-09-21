# Remote control — operating the overseer while away

Lets the overseer on this Mac be driven from anywhere by dropping files into a
Google Drive folder. No inbound network, no open ports, no VPN, and **no
dependency on a running Claude session** — a plain launchd daemon polls a synced
folder and executes a fixed set of lifecycle commands.

Decided 2026-09-21. Supersedes the earlier sketch that included
`claude --remote-control`; that layer is parked (see "Not in scope").

## Drop-in folder

```
~/Library/CloudStorage/GoogleDrive-reachlin@gmail.com/My Drive/gold-finger/
├── schwab_token.json          drop to auto-install + restart, then auto-deleted
└── control/
    ├── command.json           drop to run one command, then auto-deleted
    ├── result-<nonce>.json    written back by the daemon
    └── status.txt             human-readable latest overseer state
```

Override the location with `GOLDFINGER_DRIVE_DIR` (used by the tests).

## Commands

`control/command.json`:

```json
{"cmd": "restart", "nonce": "any-unique-string", "issued_at": "2026-09-21T13:00:00Z"}
```

| cmd | what it does |
|-----|--------------|
| `restart` | `launchctl kickstart -k gui/<uid>/com.goldfinger.overseer` |
| `stop` | `launchctl bootout gui/<uid>/com.goldfinger.overseer` — stays down; `KeepAlive` cannot resurrect an unloaded job |
| `start` | `launchctl bootstrap gui/<uid> ~/Library/LaunchAgents/com.goldfinger.overseer.plist` |
| `status` | runs `overseer_status.py`, returns the reconciled report |
| `reconcile` | pulls fresh truth from Schwab (same reconciled report). Read-only — the overseer corrects its own ledger on its next scan; use `restart` to force that now |

`issued_at` is ISO-8601 UTC. `nonce` is any string unique per command.

## Token refresh

Drop `schwab_token.json` (from the reauth on the other computer) into the
`gold-finger/` folder — no command file needed. The daemon validates it, archives
the outgoing local token as `schwab_token.json.revoked-<timestamp>`, installs it
at `schwab/schwab_token.json` with mode 600, deletes the Drive copy, and
restarts the overseer.

A Schwab reauth on another machine revokes this machine's refresh token
server-side, so the local token dies with TTL apparently remaining — this is the
recovery path for that.

> Drive deletions land in Drive trash for 30 days. Empty it if you want a
> retired credential unrecoverable.

## Design decisions

**Poll every 90s.** Worst case ~2 min from drop to action once Drive sync
latency is added. Lifecycle ops are not second-sensitive.

**Lifecycle only — no trading commands.** No place, close, or cancel. A bug in
this path can bounce the scanner; it cannot move money. The GTC covers already
rest broker-side and fill without this daemon or the overseer running.

**No authentication beyond Drive access.** That folder already carries the OAuth
token, so anyone holding the Drive account can already reauth into the brokerage
account — a signature would guard a door that is already open.

**Idempotency guards anyway** — these fix a *correctness* problem, not an auth
one. Drive re-syncs files on reconnect, device re-link and conflict resolution,
so an old `command.json` can reappear with nobody touching it:

1. **Consume-on-read** — the command file is read and deleted from Drive
   *before* it executes, invalid ones included, so nothing can loop.
2. **Freshness** — `issued_at` older than 10 min is ignored (and more than
   2 min in the future, for clock skew).
3. **Nonce ledger** — `data/remote_control_seen.json` records processed nonces
   for 24h and skips repeats.

**Materialization guard.** macOS Drive is a virtual filesystem; a file can appear
before its bytes do. A file is acted on only once its size is stable across
consecutive polls *and* it parses as JSON.

**Separate launchd agent.** `com.goldfinger.remote` is its own job with its own
`KeepAlive`, so `stop` on the overseer cannot kill the thing that would restart
it.

## Files

- `schwab/remote_control.py` — the daemon
- `schwab/test_remote_control.py` — tests
- `schwab/com.goldfinger.remote.plist` — LaunchAgent (version-controlled here,
  symlinked/copied into `~/Library/LaunchAgents/`)
- `data/remote_control_seen.json` — nonce ledger
- `data/remote_control.log` — daemon log

## Install

```bash
cp schwab/com.goldfinger.remote.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.goldfinger.remote.plist
launchctl list | grep goldfinger      # expect overseer + remote
```

Stop with `launchctl bootout gui/$(id -u)/com.goldfinger.remote`.

## Not in scope

- `claude --remote-control` — exists and works for conversational control, but
  putting a live Claude session on the critical path for real-money ops means a
  sleeping Mac or a dead tmux session takes the restart button with it. Parked.
- Any order placement or position management.
