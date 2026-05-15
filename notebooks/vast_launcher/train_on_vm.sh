#!/usr/bin/env bash
# VM-side wrapper — layer-2 defense only. Mac launcher is master.
# Defenses: bash EXIT trap, dead-man watchdog (sleep+curl), GPU-idle killer.

# Kimi #6: do NOT use `set -u` here — if VAST_INSTANCE_ID/VAST_API_KEY are ever
# unset (SSH injection bug), `set -u` collapses the script AND the EXIT trap
# (which references the same vars) before destruction can fire.
# Defensive defaults instead:
: "${VAST_INSTANCE_ID:=}" "${VAST_API_KEY:=}" "${HF_TOKEN:=}"

LOG=/tmp/train.log
exec >>"$LOG" 2>&1

# ── Self-destruct via Vast API (Kimi #4: parse JSON properly; Kimi #5: loop forever) ──
self_destruct() {
    local reason="${1:-unknown}"
    echo "[VM] self_destruct triggered: $reason" >> "$LOG"
    if [ -z "${VAST_INSTANCE_ID:-}" ] || [ -z "${VAST_API_KEY:-}" ]; then
        echo "[VM] !! VAST_INSTANCE_ID or VAST_API_KEY missing — cannot self-destruct" >> "$LOG"
        return 1
    fi
    local i=0
    while true; do
        i=$((i + 1))
        out=$(curl -sS -X DELETE "https://console.vast.ai/api/v0/instances/${VAST_INSTANCE_ID}/" \
                   -H "Authorization: Bearer ${VAST_API_KEY}" 2>&1)
        echo "[VM] destroy try $i: $out" >> "$LOG"
        # Kimi #4: parse JSON — `grep '"success": true'` breaks if Vast returns no space
        if python3 -c "import sys,json
try: d=json.loads(sys.argv[1])
except: sys.exit(1)
sys.exit(0 if d.get('success') else 1)" "$out" 2>/dev/null; then
            echo "[VM] destroy CONFIRMED on try $i" >> "$LOG"
            break
        fi
        # Kimi #5: never give up. Cap backoff at 60s. Storage cost of looping is
        # negligible vs leaving a live GPU billing.
        sleep $(( i < 6 ? 8 : 60 ))
    done
    sync
}

# Bash EXIT trap — fires on normal exit, error, or signal.
# Mac launcher is master, but if Mac is also dead, this is the floor.
trap 'self_destruct "EXIT_TRAP_$?"' EXIT
trap '' HUP   # ignore SIGHUP so SSH drop doesn't kill us

# ── Hard dead-man timer (45 min ceiling, matches Mac launcher) ───────────────
# Kimi #12: removed `shutdown -h now` — does NOT stop Vast billing, only kills
# the container. Only the API DELETE actually stops the contract.
( sleep 2700 && self_destruct "deadman_timer_45min" ) </dev/null >/dev/null 2>&1 &
disown

# ── GPU-idle killer: util <3% for 12 min after 5-min grace → destroy ────────
(
    sleep 300                                # grace period for setup
    idle=0
    while true; do
        util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
               | head -1 | tr -d ' ')
        if [[ "$util" =~ ^[0-9]+$ ]] && [[ "$util" -lt 3 ]]; then
            idle=$((idle + 30))
            if [[ "$idle" -ge 720 ]]; then
                self_destruct "gpu_idle_12min_util=${util}"
                exit
            fi
        else
            idle=0
        fi
        sleep 30
    done
) </dev/null >/dev/null 2>&1 &
disown

echo "[VM] === SETUP $(date -u +%FT%TZ) ==="
echo "[VM] HF_TOKEN set: $([ -n "${HF_TOKEN:-}" ] && echo yes || echo NO)"
echo "[VM] INSTANCE_ID: $VAST_INSTANCE_ID"

# ── Run the training ─────────────────────────────────────────────────────────
cd /root
HF_TOKEN="$HF_TOKEN" python3 train.py
RC=$?
echo "[VM] === TRAIN EXITED rc=$RC ==="

# Signal completion to Mac launcher BEFORE EXIT trap destroys
touch /tmp/train.done
sync
sleep 5

# EXIT trap will now fire and destroy.
exit $RC
