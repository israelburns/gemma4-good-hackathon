#!/usr/bin/env python3
"""
Vast.ai training launcher — Ace Burns Protocol build.
Mac side. THIS IS THE MASTER. The VM watchdog is layer-2 only.

ARCHITECTURE (Kimi Stage 2 + Gemini Stage 3):
- A100 40GB, reliability2>0.98, inet_up/down>500 Mbps, disk>50GB
- Pre-flight: kill any orphan instances first
- Hard caps: 45-min Mac wallclock, 12-min GPU-idle, $2/hr max rental price
- HF token injected via SSH env channel — never written to VM disk
- destroy_and_verify(): DELETE then poll GET /instances/ until ID is ABSENT
  ("stopped" still bills storage — Vast docs lie about this)
- try/finally on main(): DELETE fires no matter what (KeyboardInterrupt, OOM,
  exceptions, SystemExit). Mac launcher is supreme authority.
- Run with `caffeinate -i` so Mac can't sleep during the run.

REQUIRED ENV: VAST_API_KEY, HF_TOKEN.
"""
import os, sys, time, json, signal, shlex, subprocess
import urllib.request, urllib.error
from urllib.parse import quote   # Kimi #9 — urllib.request.quote does not exist
from pathlib import Path

# ───────────────────── CONFIG ─────────────────────
VAST_API_KEY = os.environ["VAST_API_KEY"]
HF_TOKEN     = os.environ["HF_TOKEN"]
SSH_KEY      = os.path.expanduser("~/.ssh/vastai_key")
TRAIN_SCRIPT = "/tmp/vast_launcher/train_on_vm.sh"
PY_SCRIPT    = "/tmp/vast_launcher/train.py"

# CAPS — defense in depth
MAX_WALLCLOCK_S   = 30 * 60        # 30 min — fastest GPU finishes in ~10 min train
MAX_PRICE_USD_HR  = 6.00           # allow B200/H100 — user wants fastest, not cheapest
MAX_IDLE_S        = 12 * 60        # GPU util <3% for this long → destroy
SSH_WAIT_S        = 600            # cold docker pull can take 5–8 min on slow hosts
POLL_INTERVAL_S   = 30
LOG_TAIL_LINES    = 20

# Vast offer filters — current API wants JSON-formatted operators per field.
# Probed actual market: B200 $5.18/hr (over cap), RTX 5090 $0.33/hr,
# RTX 4090 $0.14/hr, A100 PCIE >$2/hr at the moment.
def _q(gpu_name, dph_max, inet=200):
    return json.dumps({
        "gpu_name":     {"eq": gpu_name},
        "num_gpus":     {"eq": 1},
        "rentable":     {"eq": True},
        "reliability2": {"gt": 0.95},
        "dph_total":    {"lt": dph_max},
        "inet_up":      {"gt": inet},
        "inet_down":    {"gt": inet},
        "disk_space":   {"gt": 50},
    })

# Run #8 FAILED on B200 with bnb sm_100 kernel gap (cdequantize_blockwise_fp32
# missing). cu124 nvcc can't compile bnb for sm_100 (needs cu128+). Pivoting to
# RTX 4090 (sm_89 Ada) — every bnb wheel ever released supports it cleanly.
# Ada is 30% slower than Blackwell on small models; a guaranteed completion
# beats another $1.50 of failed Blackwell rentals.
SEARCH_QUERIES = [
    ("RTX 4090",        _q("RTX 4090", 0.6)),
    ("RTX 5090",        _q("RTX 5090", 1.0)),  # backup, Blackwell — likely fails too
]

# N=12 parallel sweep: each cell sets CELL_ID env var. Distinct LABEL per cell
# so pre-flight only kills THIS cell's orphans, not other live cells'.
CELL_ID = os.environ.get("CELL_ID", "default")
LABEL = f"jeremy-gemma4-train-{CELL_ID}"
# OFFER_INDEX: which-cheapest-offer to pick (0=cheapest, 1=2nd, ...). Each
# cell uses a different index so 12 launchers don't all rent the same offer.
OFFER_INDEX = int(os.environ.get("OFFER_INDEX", "0"))
# Pivoted to RTX 4090 (sm_89 Ada) — pytorch 2.5.1 cu124 is rock-solid.
IMAGE = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"

# ───────────────────── HELPERS ─────────────────────
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def vast(method, path, body=None):
    """Vast REST helper — Authorization: Bearer (Gemini Stage 3: query-param auth deprecated)."""
    url = f"https://console.vast.ai/api/v0{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {VAST_API_KEY}")
    if body is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"_http_error": e.code, "_body": e.read().decode(errors="replace")}
    except Exception as e:
        return {"_exception": str(e)}

def list_instances():
    """Kimi #1 BLOCKER: return None (NOT []) on any failure so callers
    can distinguish "API said no instances" from "API call failed"."""
    r = vast("GET", "/instances/")
    if not isinstance(r, dict) or "_http_error" in r or "_exception" in r:
        return None
    if "instances" not in r:
        return None
    return r["instances"]

def destroy(iid):
    return vast("DELETE", f"/instances/{iid}/")

def destroy_and_verify(iid):
    """Kimi #1/#2/#3 BLOCKERs:
       - never trust an API error as 'instance gone' (None vs [])
       - loop FOREVER (or until wallclock SIGALRM) — never give up
       - cast both sides to str (rent returns int, API returns str)
       'stopped' still bills storage — DELETE + verify-absent is the only safe signal."""
    target = str(iid)
    log(f"DESTROY {target} → calling DELETE (will loop until verified ABSENT)")
    destroy(iid)
    t0 = time.time()
    backoff = 5
    consecutive_api_errors = 0
    while True:
        inst = list_instances()
        if inst is None:                       # API call failed — keep polling
            consecutive_api_errors += 1
            log(f"  ... GET /instances/ FAILED (#{consecutive_api_errors}) — retrying, "
                f"NOT treating as success")
        else:
            consecutive_api_errors = 0
            ids = {str(i.get("id")) for i in inst}
            if target not in ids:
                log(f"DESTROY {target} → verified ABSENT (took {int(time.time()-t0)}s)")
                return True
            log(f"  ... still present (other instances: {len(ids)}), retrying DELETE")
            destroy(iid)
        time.sleep(backoff)
        backoff = min(backoff * 1.5, 60)       # exponential backoff capped at 60s

# ───────────────────── PRE-FLIGHT ─────────────────────
def preflight():
    log("=== PRE-FLIGHT ===")
    if not Path(SSH_KEY).exists():
        sys.exit(f"SSH key not found: {SSH_KEY}")
    sk = Path(SSH_KEY).stat().st_mode & 0o777
    if sk != 0o600:
        log(f"  fixing SSH key perms ({oct(sk)} → 0600)")
        Path(SSH_KEY).chmod(0o600)
    inst = list_instances()
    if inst is None:
        sys.exit("PRE-FLIGHT: GET /instances/ failed — refusing to rent (API may be down)")
    log(f"  current instances: {len(inst)}")
    # Kimi #11: aggressive — destroy ANY instance with our LABEL regardless of status
    for i in inst:
        iid, label, status = i.get("id"), i.get("label", ""), i.get("actual_status", "?")
        log(f"    id={iid} label={label!r} status={status} dph={i.get('dph_total')}")
        if label == LABEL:
            log(f"    → LABEL MATCH (orphan), destroying regardless of status")
            destroy_and_verify(iid)
        elif status in ("running", "stopped", "loading", "scheduling"):
            log(f"    → other billing instance ({status}); leaving alone, but flagged")
    log("  pre-flight OK")

# ───────────────────── SEARCH + RENT ─────────────────────
def search_offers(skip_offers=None, skip_hosts=None):
    """Return ALL offers across all tiers, cheapest-first, under price cap.
    skip_offers: set of offer ids to skip (failed previously this run).
    skip_hosts: set of host_ids to skip (host failed to boot)."""
    skip_offers = skip_offers or set()
    skip_hosts  = skip_hosts  or set()
    out = []
    for label, q in SEARCH_QUERIES:
        log(f"=== SEARCH: {label}")
        r = vast("GET", f"/bundles/?q={quote(q)}")
        offers = r.get("offers", []) if isinstance(r, dict) else []
        log(f"  {len(offers)} raw offers for {label}")
        offers.sort(key=lambda o: float(o.get("dph_total", 999)))
        for o in offers:
            oid  = o.get("id")
            host = o.get("host_id") or o.get("machine_id")
            dph  = float(o.get("dph_total", 99))
            if oid in skip_offers:    continue
            if host in skip_hosts:    continue
            if dph > MAX_PRICE_USD_HR: continue
            out.append((label, o))
    return out

def rent(offer):
    log(f"=== RENT offer={offer['id']}")
    body = {
        "client_id": "me",
        "image": IMAGE,
        "disk": 50,
        "label": LABEL,
        "runtype": "ssh",                 # SSH-based, not jupyter
        "onstart": "echo BOOT_OK > /tmp/boot_ok\n",
    }
    r = vast("PUT", f"/asks/{offer['id']}/", body)
    if not r.get("success"):
        sys.exit(f"RENT FAILED: {r}")
    iid = r["new_contract"]
    log(f"  rented: instance_id={iid}")
    return iid

# ───────────────────── SSH READY + INJECT ─────────────────────
def wait_for_ssh(iid):
    log(f"=== WAIT_FOR_SSH iid={iid} (up to {SSH_WAIT_S}s)")
    t0 = time.time()
    ssh_host = ssh_port = None
    target = str(iid)
    while time.time() - t0 < SSH_WAIT_S:
        inst = list_instances()
        if inst is None:                            # Kimi R2: API hiccup → wait, retry
            log(f"  ... GET /instances/ failed, retrying")
            time.sleep(POLL_INTERVAL_S)
            continue
        for i in inst:
            if str(i.get("id")) != target:
                continue
            status = i.get("actual_status")
            ssh_host, ssh_port = i.get("ssh_host"), i.get("ssh_port")
            if status == "running" and ssh_host and ssh_port:
                # try actual SSH handshake
                rc = subprocess.run(
                    ["ssh", "-o", "StrictHostKeyChecking=no",
                     "-o", "UserKnownHostsFile=/dev/null",
                     "-o", "ConnectTimeout=8",
                     "-i", SSH_KEY, "-p", str(ssh_port),
                     f"root@{ssh_host}", "echo SSH_OK"],
                    capture_output=True, timeout=20).returncode
                if rc == 0:
                    log(f"  SSH up: root@{ssh_host}:{ssh_port} (took {int(time.time()-t0)}s)")
                    return ssh_host, ssh_port
            log(f"  ... status={status} ssh={ssh_host}:{ssh_port} elapsed={int(time.time()-t0)}s")
            break
        time.sleep(POLL_INTERVAL_S)
    raise RuntimeError(f"SSH never came up within {SSH_WAIT_S}s")

def ssh_run(host, port, cmd, env=None, capture=True, timeout=60):
    # Kimi #10: shell-quote env values so secrets/spaces can't split the command
    full_env = ""
    if env:
        full_env = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in env.items()) + " "
    full = ["ssh", "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-i", SSH_KEY, "-p", str(port),
            f"root@{host}", full_env + cmd]
    return subprocess.run(full, capture_output=capture, text=True, timeout=timeout)

def scp_to(host, port, local, remote):
    return subprocess.run(
        ["scp", "-o", "StrictHostKeyChecking=no",
         "-o", "UserKnownHostsFile=/dev/null",
         "-i", SSH_KEY, "-P", str(port),
         local, f"root@{host}:{remote}"],
        capture_output=True, text=True, timeout=120,
    )

def inject_and_start(iid, host, port):
    log(f"=== INJECT iid={iid}")
    scp_to(host, port, PY_SCRIPT, "/root/train.py")
    scp_to(host, port, TRAIN_SCRIPT, "/root/train_on_vm.sh")
    ssh_run(host, port, "chmod +x /root/train_on_vm.sh")
    log("  starting training (nohup, env-injected token)…")
    # Token via SSH env channel — never on disk on the VM
    ssh_run(host, port,
            "nohup /root/train_on_vm.sh > /tmp/train.log 2>&1 < /dev/null &",
            env={"HF_TOKEN": HF_TOKEN, "VAST_API_KEY": VAST_API_KEY,
                 "VAST_INSTANCE_ID": str(iid)},
            capture=True)
    log("  training launched")

# ───────────────────── MONITOR ─────────────────────
# Run #7: even 60s SSH probe timed out during trainer.train()'s CPU/GPU spike
# on the VM (sshd starved). Bump to 180s, AND collapse the per-cycle SSH calls
# from 3 (is_complete + tail + gpu) to 1 (tail) — completion is detected by
# pattern in the tail content, not by a separate SSH `test -f` probe.
SSH_PROBE_TIMEOUT = 180

# Substrings in the log that mean "training script finished cleanly enough"
COMPLETION_MARKERS = (
    "=== COMPLETE",
    "Push complete.",
    "HF: https://huggingface.co/",
    "TRAIN EXITED rc=0",
)

def tail_log(host, port, n=LOG_TAIL_LINES):
    r = ssh_run(host, port, f"tail -n {n} /tmp/train.log 2>/dev/null",
                timeout=SSH_PROBE_TIMEOUT)
    return r.stdout if r.returncode == 0 else ""

def gpu_util(host, port):
    r = ssh_run(host, port,
                "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null",
                timeout=SSH_PROBE_TIMEOUT)
    try:    return int(r.stdout.strip().split()[0])
    except: return -1

def is_complete_from_tail(tail_text):
    return any(m in tail_text for m in COMPLETION_MARKERS)

def monitor(iid, host, port):
    log("=== MONITOR (Mac is master — single SSH per cycle)")
    t0 = time.time()
    last_progress_t = time.time()
    last_paste = ""
    consecutive_ssh_fails = 0
    while True:
        elapsed = time.time() - t0
        if elapsed > MAX_WALLCLOCK_S:
            log(f"!! WALLCLOCK CAP {MAX_WALLCLOCK_S}s exceeded — killing")
            return "wallclock_cap"
        # SINGLE SSH call per cycle: tail. Completion + progress both come from it.
        try:
            tail = tail_log(host, port)
            consecutive_ssh_fails = 0
        except subprocess.TimeoutExpired:
            consecutive_ssh_fails += 1
            log(f"  SSH probe timeout #{consecutive_ssh_fails} (host busy with training)")
            # 3 consecutive timeouts = host genuinely dead, not just busy
            if consecutive_ssh_fails >= 3:
                log("!! 3 consecutive SSH timeouts — host unreachable, killing")
                return "ssh_dead"
            time.sleep(POLL_INTERVAL_S)
            continue
        if tail and is_complete_from_tail(tail):
            log("  ✓ training complete signal in log tail")
            return "complete"
        if tail and tail != last_paste:
            new_lines = tail[len(last_paste):] if tail.startswith(last_paste) else tail
            for line in new_lines.strip().splitlines()[-8:]:
                log(f"  vm: {line}")
            last_paste = tail
            last_progress_t = time.time()
        elif time.time() - last_progress_t > MAX_IDLE_S:
            try:
                util = gpu_util(host, port)
            except subprocess.TimeoutExpired:
                util = -1
            log(f"  no log progress {int(time.time()-last_progress_t)}s, gpu_util={util}")
            if util >= 0 and util < 3:
                log(f"!! IDLE CAP — gpu={util}% no progress {MAX_IDLE_S}s — killing")
                return "idle_cap"
        log(f"  elapsed={int(elapsed)}s / cap={MAX_WALLCLOCK_S}s")
        time.sleep(POLL_INTERVAL_S)

MAX_BOOT_ATTEMPTS = 4   # cap rental retries when SSH never comes up

# ───────────────────── MAIN ─────────────────────
def main():
    iid = None
    try:
        preflight()
        offers = search_offers()
        if not offers:
            sys.exit("NO OFFERS under cap across all tiers — try again later")
        log(f"=== {len(offers)} offers under cap; will try up to {MAX_BOOT_ATTEMPTS}")
        # CELL_ID parallel sweep: skip the OFFER_INDEX cheapest offers so 12
        # parallel launchers don't all rent the same one. Cell 0 picks index 0,
        # cell N picks index N. Falls back gracefully if not enough offers.
        if OFFER_INDEX > 0 and OFFER_INDEX < len(offers):
            log(f"  cell={CELL_ID} skipping first {OFFER_INDEX} offers (parallel-safe)")
            offers = offers[OFFER_INDEX:] + offers[:OFFER_INDEX]
        skip_hosts = set()
        host = port = None
        for attempt, (label, offer) in enumerate(offers[:MAX_BOOT_ATTEMPTS], 1):
            log(f"=== ATTEMPT {attempt}/{MAX_BOOT_ATTEMPTS}: {label} offer={offer['id']} "
                f"host={offer.get('host_id')} dph=${offer.get('dph_total'):.3f}")
            iid = rent(offer)
            try:
                host, port = wait_for_ssh(iid)
                break  # SSH up — proceed to inject + train
            except RuntimeError as e:
                log(f"!! boot failed: {e} — destroying and trying next offer")
                destroy_and_verify(iid)
                iid = None
                bad_host = offer.get("host_id") or offer.get("machine_id")
                if bad_host:
                    skip_hosts.add(bad_host)
        if iid is None or host is None:
            sys.exit(f"All {MAX_BOOT_ATTEMPTS} boot attempts failed — bailing")
        inject_and_start(iid, host, port)
        result = monitor(iid, host, port)
        log(f"=== MONITOR EXIT: {result}")
        # final log dump
        final = tail_log(host, port, 60)
        log("=== FINAL TAIL ===\n" + final)
    except KeyboardInterrupt:
        log("!! Ctrl-C caught — destroying")
    except Exception as e:
        log(f"!! EXCEPTION: {type(e).__name__}: {e} — destroying")
    finally:
        # Kimi #8: shield the destructor — a 2nd Ctrl-C / SIGTERM during the
        # finally block must NOT interrupt destroy_and_verify(). Catch
        # BaseException (covers SystemExit + KeyboardInterrupt) and retry.
        if iid is not None:
            for attempt in range(5):
                try:
                    destroy_and_verify(iid)
                    break
                except BaseException as e:
                    log(f"!! destroy attempt {attempt+1} interrupted: {type(e).__name__}: {e}")
                    time.sleep(3)
        log("=== LAUNCHER EXIT ===")

if __name__ == "__main__":
    # signal hook in case the python process is asked to exit
    def _sigterm(signum, frame):
        log(f"!! SIG{signum} — letting finally block run")
        sys.exit(1)
    signal.signal(signal.SIGTERM, _sigterm)
    main()
