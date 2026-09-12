#!/usr/bin/env bash
# Extract the three stock FlashInfer sparse-MLA files from the image and apply the
# TP=2 (num_heads=32) instantiation patch. Writes $FIPATCH, which the launcher mounts.
set -euo pipefail
IMAGE="${IMAGE:-vllm-dsv41:base}"
FIPATCH="${FIPATCH:?set FIPATCH to the output directory}"
PATCH="${PATCH:-$(dirname "$0")/../patches/flashinfer-dsv41-sm120-tp2.patch}"
test -f "$PATCH" || { echo "patch not found: $PATCH" >&2; exit 2; }

mkdir -p "$FIPATCH"; cd "$FIPATCH"
docker run --rm -v "$PWD:/o" --entrypoint bash "$IMAGE" -c '
  S=/usr/local/lib/python3.12/dist-packages/flashinfer
  cp $S/data/csrc/sparse_mla_sm120_decode_dsv4.cu /o/
  cp $S/data/csrc/sparse_mla_sm120_prefill.cu /o/
  cp $S/mla/_sparse_mla_sm120.py /o/
  chmod 666 /o/*'

patch -p1 --no-backup-if-mismatch < "$PATCH"
grep -q 'DSV4_DISPATCH(32, 1152)' sparse_mla_sm120_decode_dsv4.cu \
  && grep -q '(32, 1152)' _sparse_mla_sm120.py \
  || { echo "patch did not apply as expected" >&2; exit 3; }
python3 -c "import py_compile,sys; py_compile.compile('_sparse_mla_sm120.py', doraise=True)"
echo "fipatch ready in $FIPATCH"
