#!/usr/bin/env bash
# Prebuild the patched FlashInfer sparse_mla_sm120 module into a persistent cache.
#
# Why this exists: the wheel ships a prebuilt AOT sparse_mla_sm120.so and
# JitSpec.is_aot short-circuits the JIT build, so a patched .cu is never compiled.
# We mask the AOT directory with an empty bind-mount to force the JIT path, and build
# into $WORK/fi-cache so that at serve time nvcc never competes with the engine for
# unified memory. Takes ~40 s.
set -euo pipefail
WORK="${WORK:?set WORK}"
FIPATCH="${FIPATCH:?set FIPATCH}"
IMAGE="${IMAGE:-vllm-dsv41:base}"
FI=/usr/local/lib/python3.12/dist-packages/flashinfer
AOT=/usr/local/lib/python3.12/dist-packages/flashinfer_jit_cache/jit_cache/sparse_mla_sm120
HERE="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$WORK/fi-cache" "$WORK/fi-aot-empty"
docker run --rm --runtime nvidia --gpus all --name fi-build \
  -v "$WORK:/w" -v "$HERE:/s:ro" \
  -v "$FIPATCH/sparse_mla_sm120_decode_dsv4.cu:$FI/data/csrc/sparse_mla_sm120_decode_dsv4.cu:ro" \
  -v "$FIPATCH/sparse_mla_sm120_prefill.cu:$FI/data/csrc/sparse_mla_sm120_prefill.cu:ro" \
  -v "$FIPATCH/_sparse_mla_sm120.py:$FI/mla/_sparse_mla_sm120.py:ro" \
  -v "$WORK/fi-aot-empty:$AOT:ro" \
  -v "$WORK/fi-cache:/root/.cache/flashinfer" \
  -e CUTE_DSL_ARCH=sm_121a -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a \
  -e MAX_JOBS="${MAX_JOBS:-10}" -e PYTHONUNBUFFERED=1 \
  --entrypoint python3 "$IMAGE" /s/fi_build.py

OBJ=$(find "$WORK/fi-cache" -name 'csrc_sparse_mla_sm120_decode_dsv4.cuda.o' | head -1)
[ -n "$OBJ" ] || { echo "no object built" >&2; exit 3; }
docker run --rm -v "$WORK/fi-cache:/c:ro" --entrypoint bash "$IMAGE" -c \
  "nm -C '${OBJ/$WORK\/fi-cache//c}' | grep -c 'sparse_mla_decode_dsv4_kernel<(ModelType)1, 32, 1152, 64>'" \
  | { read -r n; echo "(32,1152) symbols in object: $n"; [ "$n" -gt 0 ] || { echo "INSTANTIATION MISSING" >&2; exit 3; }; }
echo "flashinfer module prebuilt in $WORK/fi-cache"
