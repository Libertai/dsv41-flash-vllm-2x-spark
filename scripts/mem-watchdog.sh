#!/usr/bin/env bash
# Kill the vLLM container if host memory gets dangerously low.
#
# On unified-memory hardware an over-commit does NOT raise: it starves the OS, and the
# kernel's OOM killer cannot reclaim CUDA mappings. The box keeps answering ping and port
# 22 keeps accepting while sshd can no longer fork -- only a power cycle recovers it.
# Run this alongside any experiment that changes the memory envelope (CUDA graphs, a
# larger GMU, an extra drafter).
#
# usage: FLOOR_GB=8 ./mem-watchdog.sh <container-name>
set -uo pipefail
NAME="${1:?usage: mem-watchdog.sh <container>}"
FLOOR_GB="${FLOOR_GB:-8}"
INTERVAL="${INTERVAL:-2}"
LOG="${LOG:-$HOME/mem-watchdog.log}"

echo "$(date -Is) watchdog armed on $NAME, floor ${FLOOR_GB} GiB" | tee -a "$LOG"
while true; do
  docker ps --format '{{.Names}}' | grep -qx "$NAME" || { echo "$(date -Is) $NAME gone, exiting" | tee -a "$LOG"; exit 0; }
  avail_gb=$(( $(awk '/MemAvailable/{print $2}' /proc/meminfo) / 1048576 ))
  if [ "$avail_gb" -lt "$FLOOR_GB" ]; then
    echo "$(date -Is) MemAvailable ${avail_gb} GiB < ${FLOOR_GB} -- KILLING $NAME to save the host" | tee -a "$LOG"
    docker kill "$NAME" >/dev/null 2>&1 || true
    exit 1
  fi
  sleep "$INTERVAL"
done
