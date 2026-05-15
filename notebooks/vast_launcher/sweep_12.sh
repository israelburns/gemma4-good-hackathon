#!/usr/bin/env bash
# N=12 parallel sweep launcher — Jeremy Gemma-4 QLoRA hyperparam grid.
# 3 LoRA-r values × 4 LR values = 12 cells. Each cell = independent launcher
# process with distinct CELL_ID, OFFER_INDEX, LORA_R, LR. Pushes to its own
# HF repo: israelburns/jeremy-gemma4-{CELL_ID}.

set -uo pipefail

VAST_API_KEY="${VAST_API_KEY:?must be set}"
HF_TOKEN="${HF_TOKEN:?must be set}"

LAUNCHER=/tmp/vast_launcher/vast_train_launcher.py
LOG_DIR=/tmp/vast_launcher/sweep_logs
mkdir -p "$LOG_DIR"

# 3×4 grid (LoRA r × LR). Cell ID format: r{R}-lr{LR_TAG}
# OFFER_INDEX increments per cell so 12 launchers don't all rent same offer.
declare -a CELLS=(
    "r8-lr5e5    8    5e-5"
    "r8-lr1e4    8    1e-4"
    "r8-lr2e4    8    2e-4"
    "r8-lr5e4    8    5e-4"
    "r16-lr5e5  16    5e-5"
    "r16-lr1e4  16    1e-4"
    "r16-lr2e4  16    2e-4"
    "r16-lr5e4  16    5e-4"
    "r32-lr5e5  32    5e-5"
    "r32-lr1e4  32    1e-4"
    "r32-lr2e4  32    2e-4"
    "r32-lr5e4  32    5e-4"
)

echo "=== Launching ${#CELLS[@]} parallel cells ==="
i=0
PIDS=()
for spec in "${CELLS[@]}"; do
    read -r cell_id lora_r lr <<< "$spec"
    log_file="$LOG_DIR/${cell_id}.log"
    echo "[$cell_id] r=$lora_r lr=$lr offer_index=$i log=$log_file"
    CELL_ID="$cell_id" \
    OFFER_INDEX="$i" \
    LORA_R="$lora_r" \
    LR="$lr" \
    VAST_API_KEY="$VAST_API_KEY" \
    HF_TOKEN="$HF_TOKEN" \
        nohup python3 "$LAUNCHER" > "$log_file" 2>&1 &
    PIDS+=($!)
    i=$((i + 1))
    sleep 2     # stagger by 2s — avoid thundering herd on Vast API
done

echo "=== All ${#CELLS[@]} cells launched ==="
echo "PIDs: ${PIDS[@]}"
echo "$(date)" > "$LOG_DIR/started.txt"
printf '%s\n' "${PIDS[@]}" > "$LOG_DIR/pids.txt"
echo "Logs: $LOG_DIR/"
echo ""
echo "Monitor:"
echo "  cat $LOG_DIR/started.txt"
echo "  ls -la $LOG_DIR/"
echo "  tail -f $LOG_DIR/r16-lr1e4.log    # baseline cell"
echo "  for p in \$(cat $LOG_DIR/pids.txt); do ps -p \$p -o pid,stat,etime 2>/dev/null; done"
echo ""
echo "Kill ALL:"
echo "  kill \$(cat $LOG_DIR/pids.txt)"
