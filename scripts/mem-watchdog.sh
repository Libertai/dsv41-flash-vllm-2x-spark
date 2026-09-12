#!/usr/bin/env bash
# Kill the vLLM container if host memory gets dangerously low.
#
# On unified-memory hardware an over-commit does NOT raise: it starves the OS, and the
# kernel's OOM killer cannot reclaim CUDA mappings. The box keeps answering ping and port
# 22 keeps accepting while sshd can no longer fork -- only a power cycle recovers it.
# Run this alongside any experiment that changes the memory envelope (CUDA graphs, a
# larger GMU, an extra drafter).
#
# usage: FLOOR_MB=250 ./mem-watchdog.sh <container-name>
#
# MEASURED 2026-09-12: this lane's NORMAL working point is 1.5-2.0 GiB MemAvailable --
# weights alone are 94 GiB/rank of a 121.7 GiB unified pool (77%). A floor of 8 GiB
# therefore kills the container on EVERY boot, during kernel warmup. Keep the floor
# at 1 GiB: it still catches a real starve (the wedge bottoms out near 0) without
# firing on the steady state.
set -uo pipefail
NAME="${1:?usage: mem-watchdog.sh <container>}"
FLOOR_MB="${FLOOR_MB:-250}"
INTERVAL="${INTERVAL:-2}"
LOG="${LOG:-$HOME/mem-watchdog.log}"

echo "$(date -Is) watchdog armed on $NAME, floor ${FLOOR_MB} MB" | tee -a "$LOG"
while true; do
  docker ps --format '{{.Names}}' | grep -qx "$NAME" || { echo "$(date -Is) $NAME gone, exiting" | tee -a "$LOG"; exit 0; }
  avail_mb=$(( $(awk '/MemAvailable/{print $2}' /proc/meminfo) / 1024 ))
  if [ "$avail_mb" -lt "$FLOOR_MB" ]; then
    echo "$(date -Is) MemAvailable ${avail_mb} MB < ${FLOOR_MB} -- KILLING $NAME to save the host" | tee -a "$LOG"
    docker kill "$NAME" >/dev/null 2>&1 || true
    exit 1
  fi
  sleep "$INTERVAL"
done
