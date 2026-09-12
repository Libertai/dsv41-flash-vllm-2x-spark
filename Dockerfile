# DeepSeek-V4.1-Flash serving image for DGX Spark / GB10 (aarch64, sm_121).
#
# vLLM gained DeepseekV41ForCausalLM in https://github.com/vllm-project/vllm/pull/56214.
# Any aarch64 nightly at or after that merge works. The exact pair this recipe was
# verified against, read back off the built image:
#
#   base   vllm/vllm-openai:nightly-e7edf17cea217e52701f913cd8491fcacf2d9490
#   wheel  vllm 0.1.1.dev16+gc191787a6   (installed over the base)
#   torch 2.13.0+cu130 | flashinfer 0.6.18.post1 | CUDA 13.0.2 | python 3.12.3
#
# ⚠️ Install the nightly wheel BY URL. The nightly calls itself "0.1.0", so
#    `pip install --pre vllm` silently prefers PyPI's stable release instead.
# ⚠️ On wheels.vllm.ai the index hrefs are relative ("../../<sha>/..."), so there is no
#    "/vllm/" path segment to add, and the "+" in the filename must stay percent-encoded.
ARG BASE=vllm/vllm-openai:nightly-e7edf17cea217e52701f913cd8491fcacf2d9490
FROM ${BASE}

# Optional: install a newer nightly wheel over the base (what this build did).
# Omit it if your base already contains the DSV4.1 registry entry.
ARG VLLM_WHEEL_URL=""
RUN if [ -n "$VLLM_WHEEL_URL" ]; then pip install --no-cache-dir "$VLLM_WHEEL_URL"; fi

# Fail the build now rather than 8 minutes into a serve attempt.
RUN python3 -c "\
from vllm.models.registry import ModelRegistry as R; \
archs = R.get_supported_archs(); \
assert 'DeepseekV41ForCausalLM' in archs, 'DSV4.1 not registered in this build'; \
import vllm, torch, flashinfer; \
print('vllm', vllm.__version__, '| torch', torch.__version__, '| flashinfer', flashinfer.__version__)"
