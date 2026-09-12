#!/usr/bin/env bash
# dsv41-2xspark.sh <rank>  -- DeepSeek-V4.1-Flash (REAP-256E) on TWO DGX Spark / GB10 boxes,
# vLLM TP=2 over RoCE, Engram tables served from NVMe.
#
# Derived from tonyd2wild's 4x-Spark dsv41-tp4.sh (same fabric/NCCL shape, same patch manifest
# mechanism, same CUDA-graph rules). Differences for a 2-box cluster:
#   * the 475 GiB stock checkpoint does NOT fit on 2x120 GB -- this serves a REAP-pruned
#     256-expert checkpoint (LibertAIDAI/DeepSeek-V4.1-Flash-REAP-256E, ~94 GiB/rank)
#   * TP=2 means num_heads=32, so FlashInfer needs its own sparse-MLA instantiations
#     (see flashinfer-dsv41-sm120-tp2.patch + build-fi.sh)
#
# Knobs (export the SAME values on both nodes):
#   WORK         host dir holding the checkpoint + caches, mounted at /w   (required)
#   PATCH_DIR    vLLM patch dir with mounts.txt                            (required)
#   FIPATCH      dir with the patched FlashInfer sources                   (required)
#   CKPT         checkpoint dir under /w (default hf-k256)
#   IMAGE        default vllm-dsv41:base
#   HEAD_IP / IB_HCA / IB_GID / FABRIC_IF / ADDR_RANGE   fabric addressing (required)
#   GMU          --gpu-memory-utilization (default 0.90; 0.80 leaves NO room for KV here,
#                and above ~0.91 vLLM's own free-memory check trips)
#   MAXLEN       default 32768     SEQS default 8     MAX_BATCHED default 4096
#   EAGER        1 => --enforce-eager (DEFAULT, and the only config that fits at TP=2).
#                0 => CUDA graphs: needs the prestage patch AND more memory than a
#                2-box cluster has at the GMU that yields usable KV. See README.
#                ** EAGER=0 with ENGRAM_DISK=1 REQUIRES the prestage patch **
#   CG_SIZES     capture sizes, comma list. Default with dspark: every multiple of k and k+1
#                up to SEQS*(k+1), so each decode batch gets an exact FULL graph and no
#                padded rows ever reach SM12x sparse MLA (FlashInfer #5015).
#   SPEC         dspark (default) | none        SPEC_K default 5 (= config dspark_block_size)
#   SPEC_ADAPT   adaptive verification, default false (it forces varlen FULL graphs = padded rows)
#   ENGRAM_DISK  1 (default) => DSV41_ENGRAM_DISK=1 + disk-backed Engram patch
#   TEXT_ONLY    1 (default) => --language-model-only
#   THINKING     false (default) => --default-chat-template-kwargs '{"thinking": false}'
#   PARSERS      1 (default) => reasoning + tool parsers (both register as deepseek_v41)
set -euo pipefail
NODE_RANK="${1:?usage: dsv41-2xspark.sh <0|1>}"

WORK="${WORK:?set WORK to the host dir holding the checkpoint}"
PATCH_DIR="${PATCH_DIR:?set PATCH_DIR to the vLLM patch dir containing mounts.txt}"
FIPATCH="${FIPATCH:?set FIPATCH to the patched FlashInfer source dir}"
CKPT="${CKPT:-hf-k256}"
IMAGE="${IMAGE:-vllm-dsv41:base}"
EXP_NAME="${EXP_NAME:-boot1}"
HEAD_IP="${HEAD_IP:?head node IP on the fast fabric}"
IB_HCA="${IB_HCA:?RoCE device, e.g. from ibv_devices}"
IB_GID="${IB_GID:-3}"
FABRIC_IF="${FABRIC_IF:?fabric interface name}"
ADDR_RANGE="${ADDR_RANGE:?fabric subnet, e.g. a /30 between the two boxes}"
MPORT="${MPORT:-29801}"; PORT="${PORT:-8888}"
GMU="${GMU:-0.90}"; MAXLEN="${MAXLEN:-32768}"; SEQS="${SEQS:-8}"; MAX_BATCHED="${MAX_BATCHED:-4096}"
EAGER="${EAGER:-1}"; CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-FULL_AND_PIECEWISE}"; CG_SIZES="${CG_SIZES:-}"
SPEC="${SPEC:-dspark}"; SPEC_K="${SPEC_K:-5}"
ENGRAM_DISK="${ENGRAM_DISK:-1}"; TEXT_ONLY="${TEXT_ONLY:-1}"
THINKING="${THINKING:-false}"; PARSERS="${PARSERS:-1}"
VLLM_EXTRA="${VLLM_EXTRA:-}"; NCCL_EXTRA="${NCCL_EXTRA:-}"

NAME="dsv41-vllm-r$NODE_RANK"
SITE="/usr/local/lib/python3.12/dist-packages/vllm"
FI="/usr/local/lib/python3.12/dist-packages/flashinfer"
AOT="/usr/local/lib/python3.12/dist-packages/flashinfer_jit_cache/jit_cache/sparse_mla_sm120"
case "$NODE_RANK" in 0) HEADLESS="" ;; 1) HEADLESS="--headless" ;; *) echo "rank must be 0 or 1" >&2; exit 2 ;; esac

# ---- preflight ----
test -f "$WORK/$CKPT/config.json" || { echo "MODEL MISSING at $WORK/$CKPT" >&2; exit 3; }
test -e "$WORK/$CKPT/model-00048-of-00048.safetensors" || { echo "Engram shard 48 missing (fetch it from the base repo)" >&2; exit 3; }
test -f "$PATCH_DIR/mounts.txt" || { echo "no mounts.txt in $PATCH_DIR" >&2; exit 3; }
test -f "$WORK/fi-cache/0.6.18.post1/121a/cached_ops/sparse_mla_sm120/sparse_mla_sm120.so" \
  || { echo "patched FlashInfer module not built -- run build-fi.sh first" >&2; exit 3; }

PATCH_MOUNTS=""
while read -r f rel; do
  [ -z "$f" ] && continue
  test -f "$PATCH_DIR/$f" || { echo "PATCH FILE MISSING: $PATCH_DIR/$f" >&2; exit 3; }
  PATCH_MOUNTS="$PATCH_MOUNTS -v $PATCH_DIR/$f:$SITE/$rel:ro"
done < "$PATCH_DIR/mounts.txt"

if [ "$ENGRAM_DISK" = "1" ]; then
  ENGRAM_ENV="-e DSV41_ENGRAM_DISK=1 -e DSV41_ENGRAM_DISK_THREADS=${ENGRAM_THREADS:-32} -e DSV41_ENGRAM_DISK_CHUNK=${ENGRAM_CHUNK:-16}"
else
  ENGRAM_ENV="-e DSV41_ENGRAM_DISK=0"
fi

docker rm -f "$NAME" 2>/dev/null || true
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
AVAIL_GB=$(( $(grep MemAvailable /proc/meminfo | awk '{print $2}') / 1048576 ))
if [ "$AVAIL_GB" -lt 100 ]; then
  # The usual cause is another model server holding the pool. On unified memory there is
  # no second tier, so "something else is resident" means "this cannot start" -- and the
  # error vLLM would give you is a cryptic CUDA free-memory complaint 8 minutes later.
  echo "MemAvailable ${AVAIL_GB} GiB < 100 GiB, refusing to boot" >&2
  OTHERS=$(docker ps --format '{{.Names}}' | grep -v "^${NAME}$" || true)
  if [ -n "$OTHERS" ]; then
    echo "containers currently running (one of these is probably holding it):" >&2
    docker ps --format '   {{.Names}}  {{.Image}}  {{.Status}}' | grep -v "  ${NAME}  " >&2
    echo "also check for boot-persistent units that restart a model server:" >&2
    echo "   systemctl list-unit-files --state=enabled | grep -iE 'vllm|sglang|stack'" >&2
  fi
  exit 4
fi

# Arm the memory watchdog if we are about to change the memory envelope. CUDA-graph
# capture allocates OUTSIDE the gpu-memory-utilization bound; without this, an overrun
# starves the OS and the box needs a power cycle (ping answers, sshd cannot fork).
WD="$(cd "$(dirname "$0")" && pwd)/scripts/mem-watchdog.sh"
if [ "$EAGER" != "1" ] && [ "${WATCHDOG:-1}" = "1" ] && [ -x "$WD" ]; then
  FLOOR_GB="${WATCHDOG_FLOOR_GB:-8}" setsid nohup "$WD" "$NAME" \
    >>"${WATCHDOG_LOG:-$HOME/mem-watchdog.log}" 2>&1 < /dev/null &
  echo "mem-watchdog armed on $NAME (floor ${WATCHDOG_FLOOR_GB:-8} GiB)"
fi

GRAPH_ENV=""
if [ "$EAGER" = "1" ]; then
  GRAPH_ARGS=(--enforce-eager)
else
  # Without the prestage patch the host-side Engram gather sits INSIDE the captured
  # graph and replays a stale staging buffer -> the model emits pure NaN, silently.
  if [ "$ENGRAM_DISK" = "1" ] && ! grep -q '^model_state.py ' "$PATCH_DIR/mounts.txt"; then
    echo "EAGER=0 + ENGRAM_DISK=1 needs the Engram prestage patch (model_state.py in mounts.txt)" >&2; exit 3
  fi
  if [ -z "$CG_SIZES" ]; then
    if [ "$SPEC" = "dspark" ]; then
      CG_SIZES=$( { seq "$SPEC_K" "$SPEC_K" $((SPEC_K * SEQS)); seq $((SPEC_K + 1)) $((SPEC_K + 1)) $(((SPEC_K + 1) * SEQS)); } | sort -n -u | paste -sd, - )
    else
      CG_SIZES=$(seq 1 "$SEQS" | paste -sd, -)
    fi
  fi
  GRAPH_ARGS=(--compilation-config "{\"cudagraph_mode\":\"$CUDAGRAPH_MODE\",\"cudagraph_capture_sizes\":[$CG_SIZES]}")
  GRAPH_ENV="-e VLLM_USE_BREAKABLE_CUDAGRAPH=1"
fi
if [ "$EAGER" = "1" ]; then SPEC_ADAPT=false; else SPEC_ADAPT="${SPEC_ADAPT:-false}"; fi
if [ "$SPEC" = "dspark" ]; then
  SPEC_ARGS="--speculative-config {\"method\":\"dspark\",\"num_speculative_tokens\":$SPEC_K,\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\":\"block\",\"enable_adaptive_verification\":$SPEC_ADAPT}"
else SPEC_ARGS=""; fi
if [ "$TEXT_ONLY" = "1" ]; then TEXT_ARGS="--language-model-only"; else TEXT_ARGS=""; fi
if [ "$PARSERS" = "1" ]; then PARSER_ARGS="--reasoning-parser deepseek_v41 --enable-auto-tool-choice --tool-call-parser deepseek_v41"; else PARSER_ARGS=""; fi

mkdir -p "$WORK/cache" "$WORK/fi-aot-empty"

# shellcheck disable=SC2086
exec docker run --rm --name "$NAME" \
  --runtime nvidia --gpus all \
  --network host --ipc host --shm-size 32g --memory 112g --memory-swap 112g \
  --ulimit memlock=-1:-1 --cap-add IPC_LOCK --device /dev/infiniband:/dev/infiniband \
  --oom-score-adj 500 \
  -v "$WORK:/w" \
  $PATCH_MOUNTS \
  -v "$FIPATCH/sparse_mla_sm120_decode_dsv4.cu:$FI/data/csrc/sparse_mla_sm120_decode_dsv4.cu:ro" \
  -v "$FIPATCH/sparse_mla_sm120_prefill.cu:$FI/data/csrc/sparse_mla_sm120_prefill.cu:ro" \
  -v "$FIPATCH/_sparse_mla_sm120.py:$FI/mla/_sparse_mla_sm120.py:ro" \
  -v "$WORK/fi-aot-empty:$AOT:ro" \
  -v "$WORK/fi-cache:/root/.cache/flashinfer" \
  -e HF_HOME=/w/cache/huggingface -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_CACHE_ROOT="/w/cache/vllm-$EXP_NAME" \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_HAS_FLASHINFER_CUBIN=1 -e VLLM_DEEP_GEMM_WARMUP=skip -e PYTHONUNBUFFERED=1 \
  $ENGRAM_ENV $GRAPH_ENV \
  -e CUTE_DSL_ARCH=sm_121a -e TILELANG_CACHE_DIR=/w/tl-cache \
  -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA="$IB_HCA" \
  -e NCCL_IB_GID_AUTO=0 -e NCCL_IB_GID_INDEX="$IB_GID" \
  -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET -e NCCL_IB_ADDR_RANGE="$ADDR_RANGE" \
  -e NCCL_SOCKET_IFNAME="$FABRIC_IF" -e GLOO_SOCKET_IFNAME="$FABRIC_IF" \
  -e TP_SOCKET_IFNAME="$FABRIC_IF" -e MN_IF_NAME="$FABRIC_IF" \
  -e NCCL_NVLS_ENABLE=0 -e NCCL_CROSS_NIC=1 -e NCCL_IB_MERGE_NICS=0 -e NCCL_CUMEM_ENABLE=0 \
  -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  $NCCL_EXTRA \
  --entrypoint vllm "$IMAGE" serve "/w/$CKPT" \
    --served-model-name deepseek-v4.1-flash-reap256 deepseek-v4.1-flash \
    --host 0.0.0.0 --port "$PORT" --trust-remote-code \
    --tensor-parallel-size 2 --gpu-memory-utilization "$GMU" --max-model-len "$MAXLEN" \
    --max-num-seqs "$SEQS" --max-num-batched-tokens "$MAX_BATCHED" \
    --engram-config '{"cpu_offload": false}' \
    --kernel-config '{"enable_flashinfer_autotune": false, "enable_cutedsl_warmup": false, "enable_jit_warmup": false}' \
    --default-chat-template-kwargs "{\"thinking\": $THINKING}" \
    $TEXT_ARGS $PARSER_ARGS $SPEC_ARGS "${GRAPH_ARGS[@]}" \
    --distributed-executor-backend mp --nnodes 2 --node-rank "$NODE_RANK" \
    --master-addr "$HEAD_IP" --master-port "$MPORT" $HEADLESS $VLLM_EXTRA
