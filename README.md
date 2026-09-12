# DeepSeek-V4.1-Flash on two DGX Sparks

A complete, reproducible recipe for serving **DeepSeek-V4.1-Flash** on **two** DGX Spark /
GB10 boxes (120 GB unified memory each, sm_121) under vLLM with TP=2 over RoCE — OpenAI
API, thinking, tool calling, 32K+ context.

Everything here was measured on the hardware, not estimated. Where a number is not
measured it says so.

> **The problem in one line:** the stock checkpoint is **475 GiB**, and the non-Engram
> weights alone are **286 GiB** — 143 GiB per rank on a 2-box cluster, against ~111 GiB
> usable. It does not fit, and no amount of quantization helps because
> **the model already ships in 4-bit**. Two boxes need a *smaller model*, not a smaller
> dtype.

## What this serves, and what it costs

| | |
|---|---|
| checkpoint | [`LibertAIDAI/DeepSeek-V4.1-Flash-REAP-256E`](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-REAP-256E) — REAP-pruned, 384 → 256 experts/layer |
| weights resident | ~94 GiB/rank |
| Engram tables | **on NVMe** — 47.2 GiB/rank never allocated |
| quality cost | text perplexity +14.1%, caption perplexity +8.9% vs unpruned (see the [pruning repo](https://github.com/Libertai/deepseek-v41-flash-reap)) |

Pruning is a real quality cost and this recipe does not pretend otherwise. If you have
**four** Sparks, run the stock checkpoint with
[tonyd2wild's 4× recipe](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark)
instead — it is faster *and* lossless. This repo exists for people who have two.

## Measured performance

**Throughput is workload-dependent, and by a factor of three.** Quoting one number for
this lane is misleading, so here is the spread. Single stream (c=1), 600 tokens per case,
greedy drafting, DSpark k=5, eager, REAP-256E at TP=2, 32K context:

| workload | decode c=1 | mean acceptance length (max 6) |
|---|---:|---:|
| counting ("1 to 220") | **49.4 tok/s** | 5.90–6.00 |
| code generation | **30.4 tok/s** | 3.26–4.51 |
| reasoning (word problem) | 28.3 tok/s | 3.13–3.81 |
| chat / explanation | 27.0 tok/s | 3.23–3.59 |
| prose (400-word essay) | **19.2 tok/s** | 2.24–2.49 |

⚠️ **Do not benchmark this lane on counting prompts.** A degenerate sequence lets the
drafter accept every token (6.00 of 6, the k=5 ceiling) and reads ~2.6× the honest rate.
Earlier revisions of this file quoted **16.7 tok/s at acceptance 2.18–2.44**; that number
is reproducible, but it is the *prose* case specifically, not the lane's general speed.
Judge by code/chat/prose, and always report which.

Prefill measures **700–2,200 tok/s** depending on length (chunked, `max-num-batched-tokens
4096`). A 25K-token prompt takes ~22 s to first token.

### Speculative decoding: what helps, what does not

Every one of these was measured on the workloads above, not reasoned about:

| change | result |
|---|---|
| `draft_sample_method: greedy` (vs `probabilistic`) | **+1–3%**, and uses less GPU memory. Use it — it is also vLLM's default. |
| `num_speculative_tokens: 10` | **−24% to −38% on every workload.** Do not. |
| `num_speculative_tokens: 7` | **rejected at startup** — must be a multiple of `n_predict`. |
| CUDA graphs (`PIECEWISE` or `FULL_AND_PIECEWISE`) | **`CUDA error: an illegal memory access`** — see below. |
| `enable_adaptive_verification: true` | **impossible here** — it requires full CUDA graphs. |

⚠️ **`num_speculative_tokens` must be a MULTIPLE of `dspark_block_size` (5)**, not merely
≥ it. k=7 fails with `num_speculative_tokens:7 must be divisible by n_predict=5`.

⚠️ **k=10 is much slower, and this is structural.** A spec step verifies k+1 tokens, and
each token routes to its own 6-of-256 experts, so expert-weight traffic grows with the
block. On a bandwidth-bound box that cost outruns the extra accepted tokens — acceptance
per draft *falls* as well (counting drops 5.90 → 4.50). **k=5 is optimal. Deeper
speculation is counterproductive on a sparse MoE.**

## Context length: 32K is the honest ceiling on two boxes

The checkpoint declares 1M (YaRN). vLLM will happily start at `--max-model-len 204800` and
report a 1.27M-token KV pool. **It cannot serve it.** Weights are 94 GiB/rank of a 121.7 GiB
unified pool (77%), and long-prefill working memory has nowhere to live:

| max-model-len | prompt tested | host MemAvailable floor | result |
|---|---:|---:|---|
| 32K | 25K tokens | **913 MB** | ✅ needle retrieved |
| 64K | 35K tokens | 609 MB | ✅ needle retrieved |
| 64K | ~62K tokens | **68 MB** | ✗ container killed |
| 200K | ~150K tokens | 184 MB | ✗ container killed |
| 200K | ~150K tokens (`max-num-batched-tokens 2048`) | — | ✗ **host wedged, power cycle** |

Smaller prefill chunks do **not** fix it, which says the growth is not in the chunk
activations — the sparse-MLA indexer structures scale with context
(`candidate_topk_blocks: 2048`, `index_topk: 512`).

**Ship 32K.** It leaves a ~913 MB floor under load, roughly 3.6× the watchdog threshold.
64K works for prompts up to ~40K but has no margin left for a full-length one.

## Why Engram on disk, and why `cpu_offload` is a trap

V4.1-Flash's Engram tables are 189 GiB of the 475 — and the Engram lookup is a **hashed
n-gram gather**, not a matmul: 48 rows per token, ~12.4 KB. It never needs to be resident.

vLLM ships `--engram-config '{"cpu_offload": true}'`, which pins the tables in host RAM and
dereferences them over UVA. **On a Spark that is a no-op that costs you the memory anyway**,
because host memory *is* the accelerator pool — there is no separate tier to offload to.
So this recipe uses `cpu_offload: false` plus the disk-backed Engram patch
(`DSV41_ENGRAM_DISK=1`), and the page cache does the rest.

Measured during decode: **NVMe reads 2–3 MB/s**. Serving Engram from disk is effectively
free here.

## Prerequisites

- 2× DGX Spark / GB10 (sm_121, 120 GB unified each), Ubuntu 24.04, aarch64
- A fast fabric between them. This was built on **RoCE** (ConnectX, `/dev/infiniband`
  present, `ibv_devices` lists your HCA). Get `NCCL_IB_HCA`, the GID index, and the
  interface name before you start — guessing these wastes hours.
- ~420 GB free on the node-local NVMe of **each** box (211 GB pruned weights + 189 GiB
  Engram). The Engram shards are read from disk on both ranks, so both need a copy.
- Docker with the NVIDIA runtime.

## Step 1 — the checkpoint

```bash
WORK=$HOME/dsv41-work; mkdir -p $WORK && cd $WORK
hf download LibertAIDAI/DeepSeek-V4.1-Flash-REAP-256E --local-dir hf-k256
# shards 47-48 are the Engram tables: unchanged by pruning, so they live in the base repo
hf download deepseek-ai/DeepSeek-V4.1-Flash --local-dir hf-k256 \
   --include "model-00047-of-00048.safetensors" "model-00048-of-00048.safetensors"
python3 scripts/verify_shards.py hf-k256   # 0 bad shards of 48
```

`verify_shards.py` checks each shard's declared data extent against its file size. Run it.
A truncated download parses fine at the header level and then serves garbage.

## Step 2 — the image

vLLM gained `DeepseekV41ForCausalLM` in [#56214](https://github.com/vllm-project/vllm/pull/56214).
Build an aarch64 image from a nightly at or after that merge:

```bash
docker build -t vllm-dsv41:base -f Dockerfile .
```

Verified in this stack:

| | |
|---|---|
| base image | `vllm/vllm-openai:nightly-e7edf17cea217e52701f913cd8491fcacf2d9490` |
| vLLM | `0.1.1.dev16+gc191787a6` (wheel installed over the base) |
| torch | `2.13.0+cu130` |
| flashinfer | `0.6.18.post1` |
| CUDA / python | 13.0.2 / 3.12.3 |

⚠️ Install the nightly **by URL**, not with `--pre`: the nightly calls itself `0.1.0`, so
`pip install --pre vllm` silently prefers PyPI's release instead.

Check the architecture registered:

```bash
docker run --rm --entrypoint python3 vllm-dsv41:base -c \
  'from vllm.models.registry import ModelRegistry as R; print("DeepseekV41ForCausalLM" in R.get_supported_archs())'
```

## Step 3 — the vLLM patches

This model on sm_12x needs a patch set that is **not** upstream. It originates with
tonyd2wild's 4× Spark work; `patches/` here carries the manifest layout and the
**prestage** variant you need for CUDA graphs:

```
sparse_swa.py            v1/attention/backends/mla/sparse_swa.py
attention.py             models/deepseek_v4_1/attention.py
flashinfer_sparse.py     models/deepseek_v4_1/nvidia/flashinfer_sparse.py
engram.py                models/deepseek_v4_1/common/engram.py          <- prestage variant
weight_utils.py          model_executor/model_loader/weight_utils.py
model_state.py           models/deepseek_v4_1/nvidia/model_state.py     <- prestage variant
sparse_attn_indexer.py   model_executor/layers/sparse_attn_indexer.py
```

They are bind-mounted over the installed package, so there is nothing to rebuild.

**Use the prestage `engram.py` + `model_state.py` if you want CUDA graphs.** They move the
host-side Engram gather into `prepare_inputs`, *outside* the captured forward. Without
them, graphs replay a stale staging buffer and **the model emits pure NaN** — see
Troubleshooting.

## Step 4 — the FlashInfer kernel patch (this is the TP=2-specific part)

DSV4.1 mixes attention compress ratios, which gives sparse-MLA index widths of **1152**
(ratio 2) and **640** (ratio 1). FlashInfer instantiates its sm_120 sparse-MLA kernels per
`(num_heads, topk)` pair and ships neither width.

**`num_heads` is `64 / TP`.** So the fix depends on your parallelism: the 4× recipe needs
`(16, 1152)` and `(16, 640)`; **TP=2 needs `(32, …)`**. Copying someone else's patch for a
different TP gets you the same crash.

```bash
export WORK=$HOME/dsv41-work FIPATCH=$WORK/fipatch
./scripts/make-fipatch.sh     # pulls the 3 stock files out of the image, applies the patch
./scripts/build-fi.sh         # ~40 s; compiles and verifies the instantiation landed
```

`build-fi.sh` finishes by running `nm` on the object and failing loudly if
`sparse_mla_decode_dsv4_kernel<(ModelType)1, 32, 1152, 64>` is absent — do not skip that,
because a build that silently produced the *old* kernel set looks identical until the
engine refuses at profiling. Run both on **each** box.

The wheel ships a **prebuilt AOT `sparse_mla_sm120.so`**, and `JitSpec.is_aot`
short-circuits the JIT build before your source is read. `build-fi.sh` masks the AOT
directory with an empty bind-mount and builds into a persistent cache, which is then
mounted at serve time so nvcc never competes with the engine for unified memory.

## Step 5 — launch

Same environment on both boxes; rank 1 (worker) first, then rank 0 (head).

```bash
export WORK=$HOME/dsv41-work
export PATCH_DIR=$HOME/patches/dsv41-prestage FIPATCH=$WORK/fipatch
export HEAD_IP=<head-fabric-ip> IB_HCA=<your-hca> IB_GID=<your-gid> \
       FABRIC_IF=<your-fabric-if> ADDR_RANGE=<fabric-subnet>

./dsv41-2xspark.sh 1      # on the worker
./dsv41-2xspark.sh 0      # on the head
```

Defaults: `GMU=0.90 MAXLEN=32768 SEQS=8 EAGER=1 SPEC=dspark SPEC_K=5 PARSERS=1
JIT_WARMUP=true CUTEDSL_WARMUP=true EXEC_TIMEOUT=1800`.

Cold start is ~10–14 min. **Both warmups are on by default and they are what makes the
lane survive a real workload** — they move Triton and CuTeDSL compilation out of the
serving path, at the cost of a few minutes at boot. Do not turn them off to boot faster;
see the `EngineDeadError` entry in Troubleshooting.

This is the configuration this recipe recommends, and every part of it was A/B'd:

| knob | value | why |
|---|---|---|
| `EAGER` | `1` | CUDA graphs crash with an illegal memory access (both modes) |
| `SPEC_K` | `5` | must be a multiple of 5; k=10 is 24–38% slower |
| `draft_sample_method` | `greedy` | +1–3% over `probabilistic`, less GPU memory |
| `MAXLEN` | `32768` | 64K has no memory margin for a full-length prompt |
| `GMU` | `0.90` | 0.86 leaves no room for KV at all; >0.91 trips vLLM's own check |
| watchdog floor | `250 MB` | below the 1.5–2.0 GiB steady state and the ~500 MB warmup dip |

### `gpu-memory-utilization` has a narrow window

Weights are ~94 GiB/rank of a ~111 GiB usable pool:

| GMU | result |
|---|---|
| 0.80 | `No available memory for the cache blocks` — weights alone exceed the budget |
| **0.90** | **works** |
| 0.92 | `Free memory on device cuda:0 (111.07/121.69 GiB) ... is less than desired` |

Drop the page cache on both nodes before launching (`sync; echo 3 > /proc/sys/vm/drop_caches`)
— the launcher does this and refuses to boot below 100 GiB `MemAvailable`.

### Memory is the whole story at TP=2, and it is why CUDA graphs are hard here

A GB10 exposes ~121.7 GiB to CUDA. REAP-256E's non-Engram weights are ~94 GiB **per rank**
at TP=2 — **77% of the pool**. Compare the 4-box case with the stock checkpoint:
286 GiB / 4 = 71.5 GiB/rank, **59%**.

| | 2× Spark (REAP-256E) | 4× Spark (stock) |
|---|---:|---:|
| weights/rank | 94.0 GiB | 71.5 GiB |
| in-budget slack at GMU 0.80 | **3.4 GiB** | 25.9 GiB |
| slack needed to get usable KV | GMU 0.90 | GMU 0.80 |
| memory left outside the budget at the working GMU | **12.2 GiB** | 24.3 GiB |

`gpu-memory-utilization` bounds what vLLM allocates. **CUDA-graph capture takes memory
outside that bound**, and on unified memory "outside the bound" is also where the kernel and
`sshd` live. Push it and the box stops being reachable: ping still answers, port 22 still
accepts, but `sshd` can no longer fork, and the OOM killer cannot reclaim CUDA mappings.
Recovering means a power cycle.

**Measured outcome: CUDA graphs are not usable at TP=2 with this checkpoint** — but the
wall turned out to be *correctness*, not only memory. Both modes were tried at `GMU=0.90`
with prestage Engram and exact DSpark capture sizes in place:

| mode | capture sizes | result |
|---|---|---|
| `PIECEWISE` | 15 (multiples of k and k+1) | `CUDA error: an illegal memory access was encountered` |
| `FULL_AND_PIECEWISE` | 8 (`SEQS=4`) | `CUDA error: an illegal memory access was encountered` |

Both died in `Worker_TP0` shortly after serving started, not at capture. This is consistent
with the sm_120 sparse-MLA kernel receiving a batch shape it has no instantiation for — the
same family as FlashInfer #5015 and the DSpark illegal-access seen on the 4× RTX PRO 6000
box. **Reducing the capture set did not help**, so this is not the memory ceiling that an
earlier revision of this file blamed.

⚠️ Consequence: **`enable_adaptive_verification` is unavailable on this hardware.** It is
rejected at startup with *"Adaptive verification requires full CUDA graphs"*, and full
graphs crash. Do not spend time on it.

What is left untested: a harder prune (K=192 ≈ 71 GiB/rank, matching the 4-box footprint) to
buy back headroom, at a real accuracy cost.

**This recipe ships `EAGER=1`.** On bandwidth-bound hardware that is a small loss.

### Guard rails the launcher installs for you

- **It refuses to boot below 100 GiB `MemAvailable`** and, if something else is holding the
  pool, prints the running containers and reminds you to check boot-persistent units. On
  unified memory there is no second tier: "another model server is resident" means "this
  cannot start", and vLLM's own complaint about it arrives ~8 minutes later as a cryptic
  CUDA free-memory error. We lost a reboot to a GLM server that a `systemd` unit restarted
  at boot and quietly took 106 of 121 GB.
- **It arms `scripts/mem-watchdog.sh`** (floor **250 MB**), which kills the container before
  the host starves. That is the difference between a failed request and a power cycle — the
  kernel OOM killer cannot reclaim CUDA mappings, so without it the box becomes unreachable.
  Verified working: during an over-long prefill it fired at 115 MB and the host stayed up.
  ⚠️ **Arm it in eager mode too**, not only with graphs — eager wedges the box just as
  readily. ⚠️ **The floor must sit below the steady state (1.5–2.0 GiB) and below the
  warmup dip (~500 MB)**; 8 GiB, this script's former default, kills every boot. And make
  sure only one instance is running.

## Step 6 — verify, *then* benchmark

```bash
python3 scripts/verify_serving.py http://localhost:8888/v1 deepseek-v4.1-flash-reap256
```

This is not optional ceremony. **Two separate faults in this stack produce a server that
starts, answers, and benchmarks normally while being completely wrong.** The script asks
for `logprobs` (NaN detection), checks a known answer, exercises the **default** request
shape with no optional flags, and round-trips a tool call.

## Troubleshooting

Every failure below was hit on real hardware, with the exact string it produces.

### `Unsupported expert number: 272`
vLLM's fused MoE router dispatches on a fixed table:
`{1,2,4,8,16,32,64,128,192,256,320,384,448,512,576}`. A 272-expert checkpoint is refused
outright. **Check that table before choosing a prune ratio.**

### `SM120 sparse-MLA has no decode kernel for this shape: ... topk=1152 ...`
Step 4, and check you patched the `(num_heads, topk)` pairs for **your** TP degree.

### `decode-dsv4 launch failed (unsupported shape or kernel error)`
Subtly different from the above, and the give-away that your `.cu` patch never compiled:
the Python-side guard now passes while the C++ dispatch still refuses. The prebuilt AOT
`.so` is shadowing your source — Step 4's `build-fi.sh`.

### Fluent-looking output that is actually garbage; `logprobs` returns `nan`
```
Out of range float values are not JSON compliant: nan
```
The model is emitting NaN. **It does not crash and the throughput number it reports is
real** — we measured 14.5 tok/s from a model producing pure NaN. Two causes:

1. **CUDA graphs without the prestage Engram patch.** The host-side gather ends up inside
   the captured graph and replays a stale staging buffer. Use the prestage variants
   (Step 3), or set `EAGER=1`.
2. **Padded decode rows reaching SM12x sparse MLA** (FlashInfer #5015). Capture sizes must
   match the decode batch sizes *exactly*. With DSpark at k=5 the batch is a multiple of
   k or k+1, so the launcher captures every multiple of 5 and 6 up to `SEQS*(k+1)`. Leave
   `SPEC_ADAPT=false`: adaptive verification forces varlen FULL graphs, which reintroduces
   padded rows.

### Raw chain-of-thought inside `content`, with a bare `</think>`
No reasoning parser. Both parsers register as **`deepseek_v41`**:
`--reasoning-parser deepseek_v41 --enable-auto-tool-choice --tool-call-parser deepseek_v41`.
The response field is **`reasoning`**, not `reasoning_content`. Thinking is ON by default —
a smoke test that always sends `thinking: false` never exercises this path.

### `num_speculative_tokens:7 must be divisible by n_predict=5`
k must be a **multiple** of `dspark_block_size` (5), not merely ≥ it — so 5, 10, 15.
k=5 is also the **fastest**: k=10 is 24–38% slower on every workload (see above).

### The box becomes unreachable (ping answers, SSH hangs)
Unified memory: an over-commit starves the OS, and the OOM killer cannot reclaim CUDA
mappings. `sshd` can no longer fork while ping keeps answering; `:8033` (nginx) may still
answer while `:8888` never binds. Sometimes it recovers after ~15 min, sometimes it needs a
power cycle.

⚠️ **This is NOT caused by CUDA graphs.** It reproduces with `--enforce-eager`. The cause is
the memory working point itself: at GMU 0.90 the steady state leaves only **1.5–2.0 GiB**
of host headroom, so anything extra — a long prefill, a mid-serve JIT compile — goes over.

⚠️ **`--memory 112g --memory-swap 112g` made this worse, not better.** The intent (make the
container the OOM victim instead of the host) is right, but a hard cgroup cap forces reclaim
*inside* the cgroup against ~197 GB of mmap'd checkpoint shards, and the boot dies at the
load→profile transition. The **memory watchdog is the mechanism that actually works**: it
killed the container at 115 MB during an over-long prefill and the host stayed up.

⚠️ **Set the watchdog floor at ~250 MB, not 8 GiB.** This lane's normal working point is
1.5–2.0 GiB available and it legitimately dips to ~500 MB during warmup. A floor of 8 GiB
kills the container on every boot; earlier revisions of `mem-watchdog.sh` shipped that
default. Also make sure only **one** watchdog is running — a stale one from a previous
experiment will fire at its own old floor.

### One rank dies and the other hangs forever, with `/health` still returning 200
If rank1's container goes away, rank0 blocks indefinitely on the shm broadcast
(`No available shared memory broadcast block found in 60 seconds`) while the API server
keeps answering `/health` with 200 and requests just never return. **Do not trust `/health`
as a liveness check on this lane** — check that *both* containers are up.

### `TimeoutError: RPC call to sample_tokens timed out` / `EngineDeadError` mid-serve
The engine serves fine for a few minutes, then dies during a real workload. The worker log
shows the cause just before it: **`Triton kernel JIT compilation during inference`** /
**`CuTeDSL JIT compilation during inference`**, each warning *"causes a latency spike;
consider extending warmup"*. The compile blows past the execute-model RPC deadline, or
(worse) spikes memory while there is none left.

**Fix — both warmups, plus the timeout:**

```
--kernel-config '{"enable_jit_warmup": true, "enable_cutedsl_warmup": true, ...}'
-e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
```

⚠️ **`enable_jit_warmup` covers Triton only.** CuTeDSL kernels (e.g.
`MXFP8QuantizeSwizzledKernel`) are a separate warmup flag, and some of them are reached only
on the **long-prefill** path — so short smoke tests will not expose them. Earlier revisions
of this recipe disabled both warmups and set no timeout; that combination is the most likely
explanation for the unexplained `EngineDeadError` this cluster carried for months.

### A second job OOM-kills the first
Anything memory-hungry run next to the server (an upload that hashes hundreds of GB, a
conversion job) will take it out. On a 120 GB unified box there is no headroom — run one
thing at a time.

## What's in here

| file | what it does |
|---|---|
| `dsv41-2xspark.sh` | the launcher — run it on both boxes, worker first |
| `Dockerfile` | serving image (vLLM nightly with `DeepseekV41ForCausalLM`) |
| `patches/flashinfer-dsv41-sm120-tp2.patch` | sparse-MLA decode + prefill instantiations for topk 640/1152 at **32** heads (TP=2) |
| `scripts/make-fipatch.sh` | extract the stock FlashInfer sources from the image and apply that patch |
| `scripts/build-fi.sh` + `scripts/fi_build.py` | prebuild the patched module into a persistent cache, past the AOT `.so` that would otherwise shadow it |
| `scripts/verify_serving.py` | **run this before any benchmark** — NaN/logprobs, known answer, default request shape, tool round trip |
| `scripts/verify_shards.py` | each shard's declared extent vs its file size (catches a truncated download) |
| `scripts/bench.py` | single-stream decode by workload category, counting usage tokens not SSE chunks |
| `scripts/mem-watchdog.sh` | kills the container before the host starves; arm it for any memory-envelope experiment |

## Credits

The hard part of this — the sm_12x vLLM patch set, the disk-backed Engram, the CUDA-graph
prestage fix, the FlashInfer #5015 capture-size rule, and the DSpark configuration — is
**[tonyd2wild's 4-box recipe](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark)**,
a 4× Spark recipe. This repo adapts it to two boxes: a pruned checkpoint that fits, and the
TP=2 sparse-MLA instantiations.

Pruning recipe and the quality measurements:
**[Libertai/deepseek-v41-flash-reap](https://github.com/Libertai/deepseek-v41-flash-reap)**.

## License

MIT for the scripts here. The vLLM patches carry vLLM's Apache-2.0; the model and its
bundled reference implementation keep their own license.
