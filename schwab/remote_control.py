"""
remote_control.py — operate the overseer from anywhere via a Google Drive folder.

Drop a file in the synced Drive folder; this daemon picks it up within ~90s and
acts on it. No inbound network, no open ports, no VPN, and no dependency on a
running Claude session.

  <Drive>/gold-finger/schwab_token.json   -> validated, installed, overseer restarted
  <Drive>/gold-finger/control/command.json -> one lifecycle command

Commands (allowlist — never a shell passthrough, this machine holds live
brokerage credentials):

    restart | stop | start | status | reconcile

Results are written back to the control/ folder and pushed to Slack/Bark.

Guards, and why they exist (see REMOTE_CONTROL.md for the full rationale):

  * Consume-on-read — the command file is deleted from Drive BEFORE it runs,
    invalid ones included, so nothing can loop.
  * Freshness + nonce ledger — Drive re-syncs files on reconnect, device
    re-link and conflict resolution, so a week-old command.json can reappear
    with nobody touching it. A stale or repeated one is ignored.
  * Stability check — macOS Drive is a virtual filesystem and lists a file
    before its bytes materialise; a file is read only once its size settles
    and it parses as JSON.

There is deliberately NO authentication beyond Drive access: that folder already
carries the OAuth token, so anyone holding the Drive account can already reauth
into the brokerage account.

Run:  python schwab/remote_control.py            (daemon)
      python schwab/remote_control.py --once     (single pass, for debugging)
"""

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv

HERE       = os.path.dirname(os.path.abspath(__file__))
PROJECT    = os.path.dirname(HERE)
DATA_DIR   = os.path.join(PROJECT, "data")

load_dotenv(os.path.join(PROJECT, ".env"))
load_dotenv(os.path.expanduser("~/.claude/.env"))      # BARK_DEVICE_KEY

# --------------------------------------------------------------------- #
# Configuration                                                         #
# --------------------------------------------------------------------- #

DEFAULT_DRIVE_DIR = os.path.expanduser(
    "~/Library/CloudStorage/GoogleDrive-reachlin@gmail.com/My Drive/gold-finger")

POLL_INTERVAL_S    = 90        # drop -> action is ~2 min with Drive sync on top
COMMAND_MAX_AGE_S  = 600       # ignore anything older than 10 minutes
COMMAND_MAX_SKEW_S = 120       # tolerate this much clock skew into the future
NONCE_TTL_S        = 86400     # remember processed nonces for a day
REFRESH_TOKEN_TTL_S = 7 * 86400

ALLOWED_COMMANDS = ("restart", "stop", "start", "status", "reconcile")

OVERSEER_LABEL = "com.goldfinger.overseer"
OVERSEER_PLIST = os.path.expanduser(
    f"~/Library/LaunchAgents/{OVERSEER_LABEL}.plist")

# Overridable so an end-to-end test can exercise the install path without
# touching the live token (which would restart the overseer mid-session).
TOKEN_PATH   = os.environ.get("GOLDFINGER_TOKEN_PATH",
                              os.path.join(HERE, "schwab_token.json"))
STATUS_TOOL  = os.path.join(HERE, "overseer_status.py")
SEEN_PATH    = os.path.join(DATA_DIR, "remote_control_seen.json")
PYTHON       = sys.executable

SLACK_MENTION = "<@U02DQJ9KKFZ>"


def drive_dir() -> str:
    return os.environ.get("GOLDFINGER_DRIVE_DIR", DEFAULT_DRIVE_DIR)


def control_dir() -> str:
    return os.path.join(drive_dir(), "control")


def log(msg: str):
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {msg}", flush=True)


# --------------------------------------------------------------------- #
# Notifications                                                         #
# --------------------------------------------------------------------- #

def _send_slack(msg: str):
    import requests
    webhook = os.getenv("SLACK_WEBHOOK_URL") or os.getenv("SLACK_WEB_HOOK")
    if not webhook:
        log(f"  [Slack] no webhook configured — {msg}")
        return
    try:
        requests.post(webhook, json={"text": msg}, timeout=5).raise_for_status()
    except Exception as e:
        log(f"  [Slack] failed: {e}")


def _send_bark(title: str, body: str):
    key = os.getenv("BARK_DEVICE_KEY")
    if not key:
        return
    import urllib.parse

    import requests
    try:
        url = (f"https://api.day.app/{key}/"
               f"{urllib.parse.quote(title)}/{urllib.parse.quote(body)}")
        requests.get(url, timeout=5)
    except Exception as e:
        log(f"  [Bark] failed: {e}")


def notify(title: str, body: str):
    _send_slack(f"{SLACK_MENTION} *{title}*\n{body}")
    _send_bark(title, body)


# --------------------------------------------------------------------- #
# Pure validation helpers (clock + filesystem injected, so testable)    #
# --------------------------------------------------------------------- #

def parse_command(raw: str):
    """JSON -> dict, or None if it is malformed or not a JSON object."""
    try:
        data = json.loads(raw)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _parse_iso(stamp: str):
    """Parse ISO-8601, tolerating a trailing Z and fractional seconds."""
    if not isinstance(stamp, str):
        return None
    try:
        return datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError:
        return None


def validate_command(cmd, *, now: datetime, seen) -> tuple:
    """(ok, reason). `now` is tz-aware UTC; `seen` is processed nonces."""
    if not isinstance(cmd, dict):
        return False, "command is not a JSON object"

    name = cmd.get("cmd")
    if not name:
        return False, "missing 'cmd' field"
    if name not in ALLOWED_COMMANDS:
        return False, (f"command {name!r} is not allowed "
                       f"(allowed: {', '.join(ALLOWED_COMMANDS)})")

    nonce = cmd.get("nonce")
    if not nonce:
        return False, "missing 'nonce' field"
    if nonce in seen:
        return False, f"nonce {nonce!r} was already processed"

    issued = _parse_iso(cmd.get("issued_at"))
    if issued is None:
        return False, "missing or unparseable 'issued_at'"
    if issued.tzinfo is None:
        issued = issued.replace(tzinfo=timezone.utc)

    age = (now - issued).total_seconds()
    if age > COMMAND_MAX_AGE_S:
        return False, (f"stale command — issued {age / 60:.1f} min ago "
                       f"(limit {COMMAND_MAX_AGE_S // 60} min)")
    if age < -COMMAND_MAX_SKEW_S:
        return False, f"issued_at is {-age:.0f}s in the future"

    return True, "ok"


def validate_token(data, *, now: float) -> tuple:
    """(ok, reason) for a dropped schwab_token.json."""
    if not isinstance(data, dict):
        return False, "token file is not a JSON object"

    ct = data.get("creation_timestamp")
    if not isinstance(ct, (int, float)):
        return False, "token file has no creation_timestamp"

    ttl_h = (ct + REFRESH_TOKEN_TTL_S - now) / 3600
    if ttl_h <= 0:
        return False, f"token already expired ({-ttl_h:.1f}h ago)"

    inner = data.get("token") if isinstance(data.get("token"), dict) else data
    if not inner.get("refresh_token"):
        return False, "token file has no refresh_token"

    return True, f"valid, {ttl_h:.1f}h remaining"


def prune_nonces(seen, *, now: float, ttl_s: int = NONCE_TTL_S) -> list:
    kept = []
    for e in seen or []:
        if not isinstance(e, dict) or "nonce" not in e:
            continue
        try:
            if now - float(e.get("ts", 0)) < ttl_s:
                kept.append(e)
        except (TypeError, ValueError):
            continue
    return kept


def read_stable_json(path, *, stat_fn=None, read_fn=None, sleep_fn=None,
                     attempts: int = 3, settle_s: float = 1.0):
    """Read JSON only once the file has stopped growing.

    Drive lists a file before its bytes land, so an eager read returns a
    truncated document. Returns None if it never settles, does not parse, or
    disappears mid-read (Drive does that while resolving conflicts).
    """
    stat_fn  = stat_fn or (lambda p: os.stat(p).st_size)
    read_fn  = read_fn or (lambda p: open(p).read())
    sleep_fn = sleep_fn or time.sleep

    last_size = None
    for _ in range(attempts):
        try:
            size = stat_fn(path)
        except (OSError, FileNotFoundError):
            return None
        if size == last_size:
            try:
                return parse_command(read_fn(path))
            except (OSError, FileNotFoundError):
                return None
        last_size = size
        # Consume a read even when unsettled so callers can model progress.
        try:
            read_fn(path)
        except (OSError, FileNotFoundError):
            return None
        sleep_fn(settle_s)
    return None


# --------------------------------------------------------------------- #
# Nonce ledger                                                          #
# --------------------------------------------------------------------- #

def _load_seen() -> list:
    if not os.path.exists(SEEN_PATH):
        return []
    try:
        with open(SEEN_PATH) as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _record_nonce(nonce: str):
    seen = prune_nonces(_load_seen(), now=time.time())
    seen.append({"nonce": nonce, "ts": time.time()})
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(SEEN_PATH, "w") as f:
        json.dump(seen, f, indent=2)


# --------------------------------------------------------------------- #
# Command execution                                                     #
# --------------------------------------------------------------------- #

def _run(argv, timeout=180) -> tuple:
    # GOLDFINGER_RC_DRY_RUN lets the full drop -> validate -> execute path be
    # exercised without actually bouncing the live overseer.
    if os.environ.get("GOLDFINGER_RC_DRY_RUN"):
        return True, f"DRY RUN (not executed): {' '.join(argv)}"
    try:
        p = subprocess.run(argv, capture_output=True, text=True,
                           timeout=timeout, cwd=PROJECT)
        out = (p.stdout or "") + (p.stderr or "")
        return p.returncode == 0, out.strip()
    except subprocess.TimeoutExpired:
        return False, f"timed out after {timeout}s: {' '.join(argv)}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _gui_target() -> str:
    return f"gui/{os.getuid()}/{OVERSEER_LABEL}"


def cmd_restart() -> tuple:
    return _run(["launchctl", "kickstart", "-k", _gui_target()])


def cmd_stop() -> tuple:
    ok, out = _run(["launchctl", "bootout", _gui_target()])
    if not ok and ("No such process" in out or "not find" in out.lower()):
        return True, "overseer was already stopped"
    return ok, out or "overseer stopped (stays down until 'start')"


def cmd_start() -> tuple:
    ok, out = _run(["launchctl", "bootstrap", f"gui/{os.getuid()}",
                    OVERSEER_PLIST])
    if not ok and "already" in out.lower():
        return True, "overseer was already running"
    return ok, out or "overseer started"


def cmd_status() -> tuple:
    return _run([PYTHON, STATUS_TOOL], timeout=240)


def cmd_reconcile() -> tuple:
    # overseer_status.py is read-only: it pulls live positions, cash and
    # resting orders straight from Schwab. The overseer corrects its own
    # ledger on its next scan — use 'restart' to force that immediately.
    ok, out = cmd_status()
    return ok, out


COMMANDS = {
    "restart":   cmd_restart,
    "stop":      cmd_stop,
    "start":     cmd_start,
    "status":    cmd_status,
    "reconcile": cmd_reconcile,
}


# --------------------------------------------------------------------- #
# Result reporting                                                      #
# --------------------------------------------------------------------- #

def _write_result(nonce: str, name: str, ok: bool, output: str,
                  executed: bool = True):
    """Write result-<nonce>.json always; refresh status.txt only for commands
    that actually ran. A rejected duplicate is not overseer state, and letting
    it clobber the last good report is how you end up reading 'FAILED' on a
    perfectly healthy account."""
    os.makedirs(control_dir(), exist_ok=True)
    payload = {
        "nonce":       nonce,
        "cmd":         name,
        "ok":          ok,
        "completed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "output":      output[-8000:],
    }
    try:
        with open(os.path.join(control_dir(), f"result-{nonce}.json"), "w") as f:
            json.dump(payload, f, indent=2)
        if executed:
            with open(os.path.join(control_dir(), "status.txt"), "w") as f:
                f.write(f"{payload['completed_at']}  {name}  "
                        f"{'OK' if ok else 'FAILED'}\n\n{output}\n")
    except Exception as e:
        log(f"  could not write result: {e}")


# --------------------------------------------------------------------- #
# Handlers                                                              #
# --------------------------------------------------------------------- #

def handle_command_file(path: str) -> bool:
    """Consume-on-read: archive + delete from Drive BEFORE executing."""
    cmd = read_stable_json(path)
    if cmd is None:
        log("  command file not stable/parseable yet — leaving for next pass")
        return False

    # Delete from Drive first so a bad file can never loop.
    try:
        os.makedirs(os.path.join(DATA_DIR, "remote_commands"), exist_ok=True)
        shutil.copy2(path, os.path.join(
            DATA_DIR, "remote_commands",
            f"{int(time.time())}-{os.path.basename(path)}"))
    except Exception as e:
        log(f"  archive failed (continuing): {e}")
    try:
        os.remove(path)
    except OSError as e:
        log(f"  could not delete {path}: {e} — skipping to avoid a loop")
        return False

    seen = {e["nonce"] for e in prune_nonces(_load_seen(), now=time.time())}
    ok, why = validate_command(cmd, now=datetime.now(timezone.utc), seen=seen)
    name  = cmd.get("cmd", "?")
    nonce = cmd.get("nonce") or f"invalid-{int(time.time())}"

    if not ok:
        log(f"  REJECTED {name!r}: {why}")
        _write_result(nonce, name, False, f"rejected: {why}", executed=False)
        notify("Remote command rejected", f"{name} — {why}")
        return True

    _record_nonce(nonce)
    log(f"  executing {name!r} (nonce={nonce})")
    ran_ok, output = COMMANDS[name]()
    log(f"  {name} -> {'OK' if ran_ok else 'FAILED'}")
    _write_result(nonce, name, ran_ok, output)
    notify(f"Remote: {name} {'✅' if ran_ok else '❌'}",
           output[:600] or ("done" if ran_ok else "failed"))
    return True


def handle_token_file(path: str) -> bool:
    data = read_stable_json(path)
    if data is None:
        log("  token file not stable/parseable yet — leaving for next pass")
        return False

    ok, why = validate_token(data, now=time.time())
    if not ok:
        log(f"  REJECTED token: {why}")
        notify("Remote token rejected", why)
        try:
            os.remove(path)
        except OSError:
            pass
        return True

    try:
        if os.path.exists(TOKEN_PATH):
            stamp = datetime.now().strftime("%Y%m%d-%H%M")
            shutil.copy2(TOKEN_PATH, f"{TOKEN_PATH}.revoked-{stamp}")
        shutil.copy2(path, TOKEN_PATH)
        os.chmod(TOKEN_PATH, 0o600)
    except Exception as e:
        log(f"  token install FAILED: {e}")
        notify("Remote token install failed", str(e))
        return True

    try:
        os.remove(path)             # don't leave a live credential in Drive
    except OSError as e:
        log(f"  installed, but could not delete the Drive copy: {e}")

    log(f"  token installed ({why}) — restarting overseer")
    ran_ok, output = cmd_restart()
    _write_result(f"token-{int(time.time())}", "update_token", ran_ok,
                  f"{why}\n{output}")
    notify(f"Remote: token installed {'✅' if ran_ok else '❌'}",
           f"{why}\nOverseer restart: {'OK' if ran_ok else output[:300]}")
    return True


# --------------------------------------------------------------------- #
# Main loop                                                             #
# --------------------------------------------------------------------- #

def run_once() -> bool:
    """One poll. Returns True if anything was handled."""
    acted = False
    token_drop = os.path.join(drive_dir(), "schwab_token.json")
    if os.path.exists(token_drop):
        log(f"token drop detected: {token_drop}")
        acted = handle_token_file(token_drop) or acted

    cdir = control_dir()
    if os.path.isdir(cdir):
        for fn in sorted(os.listdir(cdir)):
            if fn.startswith("command") and fn.endswith(".json"):
                log(f"command file detected: {fn}")
                acted = handle_command_file(os.path.join(cdir, fn)) or acted
    return acted


def main():
    once = "--once" in sys.argv
    log(f"remote_control starting — watching {drive_dir()} "
        f"every {POLL_INTERVAL_S}s (commands: {', '.join(ALLOWED_COMMANDS)})")

    if not os.path.isdir(drive_dir()):
        log(f"⚠ drive folder not present yet: {drive_dir()}")

    try:
        os.makedirs(control_dir(), exist_ok=True)
    except Exception as e:
        log(f"⚠ could not create control dir: {e}")

    if once:
        run_once()
        return

    while True:
        try:
            run_once()
        except Exception as e:
            log(f"poll error (continuing): {type(e).__name__}: {e}")
        time.sleep(POLL_INTERVAL_S)


if __name__ == "__main__":
    main()
